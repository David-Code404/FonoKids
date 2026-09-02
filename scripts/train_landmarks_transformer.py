"""
train_landmarks_transformer.py
-----------------------------------
PARTE 3 del pipeline Transformer-puro: define LipReadingTransformer (SIN
CNN/ResNet -- entra directo la secuencia de 80 landmarks normalizados por
frame) y entrena con los .npy generados por extraer_landmarks_npy.py.

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

from lip_reading_dataset import LANDMARK_DIM, build_dataloaders

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_landmarks_transformer.pth")

SEED = 42
torch.manual_seed(SEED)


# =====================================================================
# MODELO: LipReadingTransformer (sin CNN -- el input ya son los 80
# landmarks normalizados por frame, no imágenes)
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


class LipReadingTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=LANDMARK_DIM, hidden_dim=256,
                 n_layers=4, n_heads=8, dim_feedforward=512, dropout=0.3):
        super().__init__()
        # Proyección de entrada: 80 landmarks -> hidden_dim (reemplaza al CNN)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, 80)
        x = self.input_proj(x)                 # (B, T, hidden_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        pooled = x.mean(dim=1)                  # promedio temporal simple
        return self.classifier(pooled)


def make_padding_mask(lengths, max_len, device):
    """True donde hay padding (para que el Transformer lo ignore)."""
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


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
    out_path = os.path.join(BASE_DIR, "outputs", "landmarks_transformer_curvas.png")
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
            mask = make_padding_mask(lengths, sequences.shape[1], device)

            if train:
                optimizer.zero_grad()

            outputs = model(sequences, src_key_padding_mask=mask)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * sequences.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc


# =====================================================================
# ENTRENAMIENTO PRINCIPAL
# =====================================================================
def train(epochs=50, batch_size=32, max_frames=60, lr=1e-4, weight_decay=1e-3,
          val_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    dataset, train_loader, val_loader = build_dataloaders(
        max_frames=max_frames, batch_size=batch_size, val_ratio=val_ratio, seed=SEED,
    )
    num_classes = len(dataset.class_names)
    print(f"Clases ({num_classes}): {dataset.class_names}")
    print(f"Total muestras: {len(dataset)} | Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)}")

    model = LipReadingTransformer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_val_acc = -1.0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\nIniciando entrenamiento...")
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Época [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": dataset.class_names,
                "max_frames": max_frames,
                "val_acc": val_acc,
            }, MODEL_PATH)
            print(f"  -> Nuevo mejor modelo guardado (Val Acc: {val_acc:.2f}%)")

    plot_training_curves(history["train_loss"], history["val_loss"],
                          history["train_acc"], history["val_acc"])

    print(f"\nListo. Mejor Val Acc: {best_val_acc:.2f}% | Modelo en: {MODEL_PATH}")


if __name__ == "__main__":
    train()
