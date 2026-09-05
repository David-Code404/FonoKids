"""
train_landmarks_transformer.py
-----------------------------------
PARTE 3 del pipeline de landmarks puros (SIN CNN/ResNet -- entra directo la
secuencia de landmarks normalizados por frame, dimensión detectada
automáticamente del dataset).

Arquitectura: TCN + CONFORMER.
    Entrada -> proyección lineal -> TCN (extracción de features locales por
    convoluciones dilatadas) -> Conformer (self-attention + módulo
    convolucional propio, captura relaciones de largo alcance) -> masked
    pooling -> clasificador (Softmax, lista cerrada de frases -- no CTC,
    porque no es transcripción de texto libre).

Es la misma familia que usan los modelos VSR de investigación (tipo
LIP-RTVE): extracción de landmarks con una red de alta fidelidad (acá
MediaPipe o HRNet/FAN, ver extraer_landmarks_npy.py) + normalización
espacial + red de secuencia temporal + clasificación.

Uso:
    python train_landmarks_transformer.py

Requiere haber corrido antes: python extraer_landmarks_npy.py
"""
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from lip_reading_dataset import build_dataloaders

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# CAMBIÁ ESTA RUTA si querés entrenar con otro dataset de landmarks (ej. uno
# que sacaste con extraer_landmarks_npy.py apuntando a otra carpeta). Por
# defecto entrena con el dataset normal de esta PC (data/landmarks_npy).
# =====================================================================
RUTA_DATASET = os.path.join(BASE_DIR, "data", "landmarks_npy")

MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_landmarks_conformer.pth")

SEED = 42
torch.manual_seed(SEED)


# =====================================================================
# Positional Encoding (igual que antes -- el Conformer también la necesita,
# la self-attention por sí sola no sabe el orden de los frames)
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# =====================================================================
# TCN (Temporal Convolutional Network) -- extracción de features locales
# ANTES del Conformer. Convoluciones 1D con dilatación creciente (1, 2, 4...)
# para agrandar el campo receptivo temporal sin agrandar el kernel, con
# conexión residual en cada bloque (igual que el TCN original de Bai et al.).
# =====================================================================
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # padding simétrico -- no causal, mantiene largo T
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, C)
        residual = x
        x = x.transpose(1, 2)  # (B, C, T) -- Conv1d espera el tiempo al final
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T, C)
        return self.relu(residual + self.dropout(x))


class TCNFeatureExtractor(nn.Module):
    def __init__(self, channels, n_blocks=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])

    def forward(self, x, key_padding_mask=None):
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x


# =====================================================================
# BLOQUE CONFORMER
# =====================================================================
class ConformerFeedForward(nn.Module):
    """Feed-forward "macaron" -- se usa DOS veces por bloque, cada una con
    la mitad de peso (residual * 0.5), a diferencia de un Transformer
    normal que tiene una sola FFN de peso completo."""

    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),  # Swish -- la activación estándar de Conformer
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    """El módulo que distingue a Conformer de un Transformer plano: una
    convolución depthwise a lo largo del tiempo, que captura patrones
    LOCALES (ej. la transición rápida entre dos formas de boca) que la
    self-attention por sí sola modela peor.

    Pointwise conv -> GLU -> Depthwise conv -> BatchNorm -> Swish -> Pointwise conv
    """

    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, d_model)
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, d_model, T) -- Conv1d espera el tiempo al final

        if key_padding_mask is not None:
            # Poné a cero los frames de padding ANTES de convolucionar, para
            # que no contaminen con ruido a sus vecinos reales.
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        return residual + x


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.ff1 = ConformerFeedForward(d_model, dim_feedforward, dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)

        self.ff2 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        # 1. FFN macaron (mitad de peso)
        x = x + 0.5 * self.ff1(x)

        # 2. Self-attention
        residual = x
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask, need_weights=False
        )
        x = residual + self.attn_dropout(attn_out)

        # 3. Módulo convolucional (lo distintivo de Conformer)
        x = self.conv_module(x, key_padding_mask=src_key_padding_mask)

        # 4. FFN macaron (mitad de peso)
        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward,
                 conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dim_feedforward, conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


# =====================================================================
# MODELO: TCN + Conformer (VSR real -- reemplaza al Transformer plano
# y al ResNet del pipeline viejo). input_dim se detecta del dataset, no
# viene hardcodeado.
# =====================================================================
class LipReadingConformer(nn.Module):
    # Defaults pensados para dataset CHICO (pocas clases / pocas muestras por
    # clase, como ahora mismo con 3-19 de las 57 palabras). Con ~4M de
    # parámetros el Conformer memorizaba el train set en vez de generalizar
    # (Train Loss bajaba, Val Loss no). Cuando tengas el dataset completo
    # (57 clases, ~500 clips c/u) subí hidden_dim=256, n_layers=4,
    # dim_feedforward=512 -- ahí sí va a hacer falta esa capacidad.
    def __init__(self, num_classes, input_dim, hidden_dim=128,
                 n_layers=2, n_heads=8, dim_feedforward=256,
                 conv_kernel_size=15, tcn_blocks=3, tcn_kernel_size=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Extracción de features locales (TCN) ANTES de pasarle la secuencia
        # al Conformer -- ver TCNFeatureExtractor más arriba.
        self.tcn = TCNFeatureExtractor(
            hidden_dim, n_blocks=tcn_blocks, kernel_size=tcn_kernel_size, dropout=dropout
        )

        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.conformer_encoder = ConformerEncoder(
            d_model=hidden_dim, n_layers=n_layers, n_heads=n_heads,
            dim_feedforward=dim_feedforward, conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, input_dim)
        x = self.input_proj(x)                 # (B, T, hidden_dim)
        x = self.tcn(x, key_padding_mask=src_key_padding_mask)
        x = self.pos_encoder(x)
        x = self.conformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        pooled = masked_mean_pool(x, src_key_padding_mask)
        return self.classifier(pooled)


def masked_mean_pool(x, src_key_padding_mask):
    """Global Average Pooling que IGNORA los frames de padding.

    x.mean(dim=1) directo diluye la representación con los frames en cero
    del padding -- acá se promedia solo sobre los frames reales, usando la
    máscara invertida (src_key_padding_mask es True en el padding).
    """
    if src_key_padding_mask is None:
        return x.mean(dim=1)

    real_mask = (~src_key_padding_mask).unsqueeze(-1).float()   # (B, T, 1), 1=real, 0=padding
    summed = (x * real_mask).sum(dim=1)                          # (B, hidden_dim)
    counts = real_mask.sum(dim=1).clamp(min=1.0)                 # (B, 1) -- evita /0 si T real=0
    return summed / counts


def make_padding_mask(lengths, max_len, device):
    """True donde hay padding (para que el encoder -- y el masked pooling --
    lo ignoren)."""
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


# =====================================================================
# EARLY STOPPING (basado en val_loss, no en val_accuracy)
# =====================================================================
class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# =====================================================================
# GRAFICAS (mismo estilo que el resto del proyecto)
# =====================================================================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r--", label="Validation Loss", linewidth=2)
    ax1.set_title("Convergencia de Pérdida")
    ax1.set_xlabel("Épocas"); ax1.set_ylabel("Pérdida"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, train_accs, "b-", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, val_accs, "g-", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Precisión del Modelo")
    ax2.set_xlabel("Épocas"); ax2.set_ylabel("Precisión (%)"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "outputs", "landmarks_conformer_curvas.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"-> Gráfico guardado: {out_path}")


# =====================================================================
# EVALUACIÓN
# =====================================================================
def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for sequences, labels, lengths in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            mask = make_padding_mask(lengths, sequences.shape[1], device)

            if train:
                optimizer.zero_grad()

            outputs = model(sequences, src_key_padding_mask=mask)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                # Gradient clipping -- los Transformer/Conformer entrenados desde
                # cero son propensos a picos de gradiente que hacen "rebotar" la
                # pérdida en vez de converger suavemente (justo lo que se veía:
                # Val Loss subiendo y bajando sin mejorar). Esto lo estabiliza.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * sequences.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc


def build_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs=5):
    """LR sube linealmente desde ~0 hasta el valor pedido durante los
    primeros `warmup_epochs`, y después decae en coseno.

    Sin warmup, un Transformer/Conformer arrancado desde cero con AdamW y
    LR alto (1e-3) suele desestabilizarse en las primeras épocas -- gradientes
    grandes contra pesos todavía aleatorios -- lo que se vio como Val Loss
    "rebotando" en vez de bajar. Empezar suave evita ese quilombo inicial."""
    warmup_epochs = max(1, min(warmup_epochs, epochs - 1)) if epochs > 1 else 1

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# ENTRENAMIENTO PRINCIPAL
# =====================================================================
def train(epochs=50, batch_size=32, max_frames=60, lr=1e-3, weight_decay=1e-3,
          val_ratio=0.2, early_stopping_patience=15, warmup_epochs=5,
          label_smoothing=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    dataset, train_loader, val_loader = build_dataloaders(
        data_dir=RUTA_DATASET, max_frames=max_frames, batch_size=batch_size,
        val_ratio=val_ratio, seed=SEED, augment_train=True,
    )
    num_classes = len(dataset.class_names)
    input_dim = dataset.landmark_dim  # detectado del .npy real, no hardcodeado
    print(f"Clases ({num_classes}): {dataset.class_names}")
    print(f"Dimensión de entrada detectada: {input_dim}")
    print(f"Total muestras: {len(dataset)} | Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)}")

    model = LipReadingConformer(num_classes=num_classes, input_dim=input_dim).to(device)
    # Label smoothing: en vez de pedirle al modelo 100% de confianza en la
    # clase correcta y 0% en el resto, le pide un poco menos (ej. 90%/10%
    # repartido) -- reduce sobreconfianza y sobreajuste, más notorio cuantas
    # más clases hay (acá van a ser 57).
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs)
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_val_loss = float("inf")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\nIniciando entrenamiento (Conformer)...")
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_actual = optimizer.param_groups[0]["lr"]
        print(f"Época [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr_actual:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": dataset.class_names,
                "max_frames": max_frames,
                "input_dim": input_dim,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, MODEL_PATH)
            print(f"  -> Nuevo mejor modelo guardado (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"  -> Early stopping activado en la época {epoch+1} "
                  f"(sin mejora en Val Loss por {early_stopping_patience} épocas).")
            break

    plot_training_curves(history["train_loss"], history["val_loss"],
                          history["train_acc"], history["val_acc"])

    print(f"\nListo. Mejor Val Loss: {best_val_loss:.4f} | Modelo en: {MODEL_PATH}")


if __name__ == "__main__":
    train()
