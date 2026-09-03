import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 0. REPRODUCIBILIDAD
# =====================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# CAMBIÁ ESTA RUTA si querés entrenar con otra carpeta de dataset .pt
# (recortes de boca). Por defecto usa el dataset normal de esta PC
# (data/dataset_pt).
# =====================================================================
RUTA_DATASET = os.path.join(BASE_DIR, "data", "dataset_pt")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =====================================================================
# 1. FUNCIONES DE VISUALIZACIÓN MATPLOTLIB (Las 3 mejores opciones)
# =====================================================================
def plot_roi_scatter(dataloader):
    """
    1. Gráfico de Dispersión + Imagen: chequeo visual rápido del recorte (ROI) de boca.

    IMPORTANTE: los puntos (centro y comisuras) son posiciones FIJAS de referencia,
    no landmarks reales detectados. Sirven solo para confirmar visualmente que el
    crop está centrado y con el tamaño esperado (112x112), NO para validar la
    precisión de la detección de landmarks en sí.
    """
    sequences, _ = next(iter(dataloader))
    first_frame = sequences[0][0]  # Batch 0, Frame 0

    img = first_frame.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    h, w, _ = img.shape

    plt.figure(figsize=(5, 5))
    plt.imshow(img)

    plt.scatter([w / 2], [h / 2], color='red', marker='x', s=100, label='Centro ROI (referencia)')
    plt.scatter([w / 4, 3 * w / 4], [h / 2, h / 2], color='yellow', marker='o', s=30, label='Comisuras (referencia)')

    plt.title("Chequeo visual del ROI (posiciones de referencia, no landmarks reales)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, "01_validacion_espacial.png"))
    plt.close()
    print("-> Gráfico generado: 01_validacion_espacial.png")


def plot_training_curves(train_losses, val_losses, val_accs, has_validation=True):
    """
    2. Gráficos de Línea: Monitorea la convergencia temporal y detecta overfitting.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if has_validation:
        ax1.plot(epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Convergencia de Pérdida (Loss)')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True)

    if has_validation:
        ax2.plot(epochs, val_accs, 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Precisión del Modelo (Accuracy)')
    else:
        ax2.text(0.5, 0.5, 'Sin set de validación\n(dataset demasiado chico)',
                  ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Precisión del Modelo (no disponible)')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Precisión (%)')
    ax2.legend() if has_validation else None
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "02_curvas_entrenamiento.png"))
    plt.close()
    print("-> Gráfico generado: 02_curvas_entrenamiento.png")


def plot_confusion_matrix(model, dataloader, device, class_names):
    """
    3. Mapas de Calor (Matriz de Confusión): Evalúa las predicciones finales.
    """
    num_classes = len(class_names)
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.float().to(device)
            labels = labels.cpu().numpy()
            outputs = model(sequences)
            _, preds = outputs.max(1)
            preds = preds.cpu().numpy()

            for t, p in zip(labels, preds):
                confusion_matrix[t, p] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = confusion_matrix.max() / 2. if confusion_matrix.max() > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "03_matriz_confusion.png"))
    plt.close()
    print("-> Gráfico generado: 03_matriz_confusion.png")


# =====================================================================
# 2. DATASET, EARLY STOPPING Y NÚCLEO IA
# =====================================================================
def safe_torch_load(path):
    """
    Carga robusta ante diferencias de versión de PyTorch: en PyTorch >= 2.6
    el default de `torch.load` cambió a weights_only=True, lo que puede
    romper la carga de archivos .pt guardados con estructuras más viejas.
    """
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class FolderLipReadingDataset(Dataset):
    # Estadísticas de ImageNet, necesarias porque el backbone (ResNet18) viene
    # preentrenado con ellas. Sin esto, el modelo preentrenado pierde gran
    # parte de su ventaja de transfer learning.
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, data_dir, max_frames=25):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.pt"), recursive=True))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sample = safe_torch_load(self.file_paths[idx])
        sequence = sample["sequence"]
        label = sample["label"]

        # --- Redimensionar espacialmente a 112x112 si hace falta ---
        if sequence.shape[-2:] != (112, 112):
            sequence = F.interpolate(sequence, size=(112, 112), mode='bilinear', align_corners=False)

        # --- Asegurar 3 canales (RGB) para que entre al ResNet18 ---
        if sequence.dim() == 3:
            sequence = sequence.unsqueeze(1)
        if sequence.shape[1] == 1:
            sequence = sequence.repeat(1, 3, 1, 1)
        elif sequence.shape[1] != 3:
            raise ValueError(
                f"Se esperaban 1 o 3 canales, se encontraron {sequence.shape[1]} "
                f"en el archivo: {self.file_paths[idx]}"
            )

        # --- Normalizar a [0,1] si viene en rango 0-255, y aplicar mean/std de ImageNet ---
        sequence = sequence.float()
        if sequence.max() > 1.5:  # heurística: valores tipo 0-255
            sequence = sequence / 255.0
        sequence = (sequence - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # --- Ajustar cantidad de frames (padding o truncado) ---
        T = sequence.shape[0]
        if T < self.max_frames:
            padding = torch.zeros((self.max_frames - T,) + sequence.shape[1:], dtype=sequence.dtype)
            sequence = torch.cat([sequence, padding], dim=0)
        elif T > self.max_frames:
            sequence = sequence[:self.max_frames]

        return sequence, torch.as_tensor(label, dtype=torch.long)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VisualSpeechTransformer(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, n_layers=4, n_heads=8):
        super(VisualSpeechTransformer, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.spatial_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.projection = nn.Linear(512, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_classes)
        )

        self.projection.apply(initialize_weights)
        self.classifier.apply(initialize_weights)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        features = self.spatial_extractor(x).view(b, t, 512)
        features = self.pos_encoder(self.projection(features))

        transformer_output = self.transformer_encoder(features)
        return self.classifier(torch.mean(transformer_output, dim=1))


# =====================================================================
# 3. SCRIPT PRINCIPAL DE ENTRENAMIENTO
# =====================================================================
def load_class_names(data_dir, dataset):
    """
    Usa dataset_pt/label_map.json (palabra -> índice) para poder mostrar
    los nombres de las palabras en la matriz de confusión, en vez de números.
    Si no existe, arma los nombres a partir de los labels realmente presentes.
    """
    label_map_path = os.path.join(data_dir, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        num_classes = max(label_map.values()) + 1
        class_names = ["?"] * num_classes
        for word, idx in label_map.items():
            class_names[idx] = word
        return class_names

    labels_present = sorted({int(dataset[i][1]) for i in range(len(dataset))})
    return [str(l) for l in labels_present]


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    dataset = FolderLipReadingDataset(data_dir=RUTA_DATASET, max_frames=25)
    if len(dataset) == 0:
        print(f"No hay archivos .pt en {RUTA_DATASET}. Este pipeline (recortes de "
              "boca) ya no tiene script de extracción propio -- usá el dataset .pt que ya "
              "tengas, o pasate al pipeline de landmarks puros (extraer_landmarks_npy.py + "
              "train_landmarks_transformer.py).")
        return

    class_names = load_class_names(RUTA_DATASET, dataset)
    num_classes = len(class_names)
    print(f"Clases detectadas ({num_classes}): {class_names}")

    labels_con_datos = sorted({int(dataset[i][1]) for i in range(len(dataset))})
    if len(labels_con_datos) < 2:
        print("\n[AVISO] Solo hay muestras reales de UNA clase todavía "
              f"({class_names[labels_con_datos[0]]}). El modelo no puede aprender a distinguir "
              "nada así -- va a 'acertar' siempre trivialmente. Grabá al menos otra palabra "
              "distinta antes de sacar conclusiones de accuracy/matriz de confusión.\n")

    has_validation = len(dataset) >= 2
    if has_validation:
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    else:
        print("AVISO: dataset muy chico (<2 muestras), se entrena sin validación.")
        train_dataset = dataset
        val_loader = None

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    print("\nGenerando pre-visualización espacial de los datos...")
    plot_roi_scatter(train_loader)

    model = VisualSpeechTransformer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=4, min_delta=0.005) if has_validation else None

    epochs = 50
    best_loss = float('inf')
    model_path = os.path.join(MODELS_DIR, "mejor_modelo_speakshadow.pt")

    history_train_loss = []
    history_val_loss = []
    history_val_acc = []

    print("\nIniciando Bucle de Entrenamiento...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.float().to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sequences), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        if has_validation:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.float().to(device), labels.to(device)
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
            history_val_loss.append(avg_val_loss)
            history_val_acc.append(val_acc)

            scheduler.step(avg_val_loss)
            monitored_loss = avg_val_loss
            print(f"Época [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        else:
            scheduler.step(avg_train_loss)
            monitored_loss = avg_train_loss
            print(f"Época [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}")

        if monitored_loss < best_loss:
            best_loss = monitored_loss
            torch.save(model.state_dict(), model_path)

        if early_stopping is not None:
            early_stopping(monitored_loss)
            if early_stopping.early_stop:
                print(f"-> Early Stopping activado (Época {epoch+1}).")
                break

    print("\nGenerando gráficas de evaluación finales...")

    plot_training_curves(history_train_loss, history_val_loss, history_val_acc, has_validation=has_validation)

    if has_validation:
        model.load_state_dict(safe_torch_load(model_path))
        plot_confusion_matrix(model, val_loader, device, class_names=class_names)
    else:
        print("-> Se omite la matriz de confusión: no hay set de validación.")

    print(f"\nListo. Modelo guardado en: {model_path}")
    print("Pipeline de entrenamiento finalizado.")


if __name__ == "__main__":
    run_training()
