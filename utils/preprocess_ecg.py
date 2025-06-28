def preprocess_ecg_input(file_path):
    import wfdb
    import numpy as np
    import torch
    import scipy.signal as sp_signal

    # Automatically strip .dat if needed
    record_path = file_path.replace(".dat", "")  
    ecg, _ = wfdb.rdsamp(record_path)

    # Assume channel 0 is the lead of interest
    ecg_signal = ecg[:, 0].flatten()

    if len(ecg_signal) < 1000:
        raise ValueError("ECG signal too short to extract R-R interval.")

    # Normalize safely
    ecg_signal = ecg_signal / (np.max(np.abs(ecg_signal)) + 1e-8)

    fs = 500  # Sampling frequency
    pks, _ = sp_signal.find_peaks(ecg_signal, height=np.mean(ecg_signal), distance=fs // 2)

    # Use R-R segment if possible, else fallback to fixed window
    if len(pks) >= 2:
        segment = ecg_signal[pks[0]:pks[1]]
    else:
        segment = ecg_signal[:500]

    # Ensure fixed size: 500 samples
    if len(segment) < 500:
        segment = np.pad(segment, (0, 500 - len(segment)), mode='constant')
    else:
        segment = segment[:500]

    # Final shape: [1, 500, 1] — batch size 1, sequence length 500, 1 channel
    tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return tensor
