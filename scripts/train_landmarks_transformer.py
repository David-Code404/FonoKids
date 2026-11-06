import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    import seaborn as sns
except ImportError:
    sns = None

from lip_reading_dataset import build_dataloaders

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUTA_DATASET = os.path.join(BASE_DIR, "data", "landmarks_npy")

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")

SEED = 42
torch.manual_seed(SEED)


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


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = x.transpose(1, 2)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1, 2)
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


class ConformerFeedForward(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):

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
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
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
        x = x + 0.5 * self.ff1(x)

        residual = x
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask, need_weights=False
        )
        x = residual + self.attn_dropout(attn_out)

        x = self.conv_module(x, key_padding_mask=src_key_padding_mask)

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


class LipReadingConformer(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim=128,
                 n_layers=2, n_heads=8, dim_feedforward=256,
                 conv_kernel_size=15, tcn_blocks=3, tcn_kernel_size=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

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

    def forward_features(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.tcn(x, key_padding_mask=src_key_padding_mask)
        x = self.pos_encoder(x)
        x = self.conformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return masked_mean_pool(x, src_key_padding_mask)

    def forward(self, x, src_key_padding_mask=None):
        pooled = self.forward_features(x, src_key_padding_mask=src_key_padding_mask)
        return self.classifier(pooled)


def masked_mean_pool(x, src_key_padding_mask):
    if src_key_padding_mask is None:
        return x.mean(dim=1)

    real_mask = (~src_key_padding_mask).unsqueeze(-1).float()
    summed = (x * real_mask).sum(dim=1)
    counts = real_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def make_padding_mask(lengths, max_len, device):
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


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
    warmup_epochs = max(1, min(warmup_epochs, epochs - 1)) if epochs > 1 else 1

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _es_clase_correcta(nombre_clase):
    return nombre_clase.endswith("_correcto")


def evaluate_final_report(model, val_loader, class_names, device):
    model.eval()
    all_preds, all_labels, all_top3_hit, all_top_probs = [], [], [], []
    latencies_ms = []

    with torch.no_grad():
        for sequences, labels, lengths in val_loader:
            sequences = sequences.to(device)
            lengths_dev = lengths.to(device)
            mask = make_padding_mask(lengths_dev, sequences.shape[1], device)

            outputs = model(sequences, src_key_padding_mask=mask)
            probs = torch.softmax(outputs, dim=1)

            top1 = probs.argmax(dim=1).cpu()
            top3 = probs.topk(min(3, probs.shape[1]), dim=1).indices.cpu()
            top_prob = probs.max(dim=1).values.cpu()

            all_preds.extend(top1.tolist())
            all_labels.extend(labels.tolist())
            all_top_probs.extend(top_prob.tolist())
            for i in range(labels.size(0)):
                all_top3_hit.append(labels[i].item() in top3[i].tolist())

            for i in range(sequences.size(0)):
                start = time.perf_counter()
                _ = model(sequences[i : i + 1], src_key_padding_mask=mask[i : i + 1])
                latencies_ms.append((time.perf_counter() - start) * 1000)

    total = len(all_labels)
    top1_acc = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / total
    top3_acc = 100.0 * sum(all_top3_hit) / total

    cm_path = os.path.join(BASE_DIR, "outputs", "matriz_confusion.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(max(8, len(class_names) * 0.35), max(6, len(class_names) * 0.35)))
    if sns is not None:
        sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    else:
        plt.imshow(cm, cmap="Blues")
        plt.xticks(range(len(class_names)), class_names)
        plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusion (validacion)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    es_correcta_real = [_es_clase_correcta(class_names[l]) for l in all_labels]
    es_correcta_pred = [_es_clase_correcta(class_names[p]) for p in all_preds]
    binaria_acc = 100.0 * sum(r == p for r, p in zip(es_correcta_real, es_correcta_pred)) / total

    umbral = 0.40
    con_confianza = sum(1 for p in all_top_probs if p >= umbral)
    sin_confianza = total - con_confianza

    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)

    secciones = [
        f"## Resultados\n\n**Muestras de validacion:** {total}",
        f"### Accuracy\n\n- Top-1: **{top1_acc:.2f}%**\n- Top-3: **{top3_acc:.2f}%**",
        f"### Matriz de confusion\n\nGuardada en: `{os.path.basename(cm_path)}`",
        f"### Accuracy correcta/incorrecta (binaria)\n\n- **{binaria_acc:.2f}%** -- ¿el modelo distingue bien si la palabra se dijo bien o mal, más allá de si acierta el subtipo exacto de error?",
        (
            f"### Confianza del modelo (umbral {umbral*100:.0f}%)\n\n"
            f"- Predicciones con confianza suficiente para mostrarle un resultado al chico: **{con_confianza}/{total}** "
            f"({100.0*con_confianza/total:.2f}%)\n"
            f"- Sin confianza suficiente (el server pediría repetir la toma): **{sin_confianza}** "
            f"({100.0*sin_confianza/total:.2f}%)"
        ),
        f"### Latencia de inferencia\n\n- Promedio: **{avg_latency_ms:.2f} ms** (batch=1, {device})",
    ]

    for seccion in secciones:
        print(seccion + "\n")

    report_path = os.path.join(BASE_DIR, "outputs", "reporte_resultados.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(secciones) + "\n")
    print(f"-> Reporte guardado: {report_path}")


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
    input_dim = dataset.landmark_dim
    print(f"Clases ({num_classes}): {dataset.class_names}")
    print(f"Dimensión de entrada detectada: {input_dim}")
    print(f"Total muestras: {len(dataset)} | Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)}")

    model = LipReadingConformer(num_classes=num_classes, input_dim=input_dim).to(device)

    train_labels = [dataset._label_from_path(p) for p in train_loader.dataset.file_paths]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    class_weights = class_counts.sum() / (num_classes * class_counts)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("Pesos por clase (loss):")
    for name, w, c in zip(dataset.class_names, class_weights, class_counts):
        print(f"  {name:28s} {int(c):4d} muestras -- peso {w:.3f}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights_t)
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

    try:
        best_checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        evaluate_final_report(model, val_loader, dataset.class_names, device)
    except Exception as e:
        print(f"\n[AVISO] No se pudo generar el reporte de resultados: {e}")
        print("El modelo entrenado SÍ se guardó bien, esto solo afecta al reporte de métricas.")


if __name__ == "__main__":
    train()
