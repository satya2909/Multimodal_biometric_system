import torch
import torch.nn as nn
import numpy as np
import wfdb
import scipy.signal as sp_signal
import joblib
import os

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# === ECG Transformer Model ===
class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, seq_len=500, d_model=64, nhead=4, num_layers=4, num_classes=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)  # [1, 1, 500, 1] → [1, 500, 1]
        x = self.input_proj(x)         # [1, 500, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# === Load ECG Model + Encoder ===
def load_ecg_model(device, model_path="weights/transformer_model_shared.pth", encoder_path="weights/shared_label_encoder.pkl"):
    label_encoder = joblib.load(encoder_path)
    num_classes = len(label_encoder.classes_)
    
    model = ECGTransformer(num_classes=num_classes).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    # Optional: remove classifier if class mismatch
    if any(k.startswith("classifier.") for k in state_dict.keys()):
        current_classifier_shape = model.classifier[-1].weight.shape[0]
        loaded_classifier_shape = state_dict["classifier.3.weight"].shape[0]
        if current_classifier_shape != loaded_classifier_shape:
            for key in list(state_dict.keys()):
                if key.startswith("classifier."):
                    del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, label_encoder

# === Preprocess ECG Segment ===
def preprocess_ecg_input(file_path):
    record_path = file_path.replace(".dat", "")
    ecg, _ = wfdb.rdsamp(record_path)
    ecg_signal = ecg[:, 0].flatten()

    if len(ecg_signal) < 1000:
        raise ValueError("ECG signal too short")

    ecg_signal /= np.max(np.abs(ecg_signal))
    fs = 500
    pks, _ = sp_signal.find_peaks(ecg_signal, height=np.mean(ecg_signal), distance=fs // 2)

    if len(pks) >= 2:
        segment = ecg_signal[pks[0]:pks[1]]
        if len(segment) < 500:
            segment = np.pad(segment, (0, 500 - len(segment)), mode='constant')
        else:
            segment = segment[:500]
    else:
        # Fallback to fixed window
        segment = ecg_signal[:500]
        if len(segment) < 500:
            segment = np.pad(segment, (0, 500 - len(segment)), mode='constant')

    tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Shape: [1, 500, 1]
    return tensor
