from flask import Flask, render_template, request
import os
import torch
import numpy as np
import traceback

from models.fingerprint_model import load_fingerprint_model, preprocess_fingerprint_image
from models.ecg_model import load_ecg_model, preprocess_ecg_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models and label encoders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fingerprint_model, fp_label_encoder = load_fingerprint_model(device)
ecg_model, ecg_label_encoder = load_ecg_model(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        fingerprint_file = request.files.get('fingerprint')
        ecg_dat_file = request.files.get('ecg_dat')
        ecg_hea_file = request.files.get('ecg_hea')

        # === File Validations ===
        if not fingerprint_file or not ecg_dat_file or not ecg_hea_file:
            return render_template('error.html', message="Please upload fingerprint and both ECG .dat and .hea files.")

        if not fingerprint_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('error.html', message="Invalid fingerprint file format. Use .png, .jpg, or .jpeg.")
        if not ecg_dat_file.filename.endswith('.dat') or not ecg_hea_file.filename.endswith('.hea'):
            return render_template('error.html', message="Invalid ECG file format. Please upload both .dat and .hea files.")

        # === Save Uploaded Files ===
        fingerprint_path = os.path.join(app.config['UPLOAD_FOLDER'], fingerprint_file.filename)
        ecg_dat_path = os.path.join(app.config['UPLOAD_FOLDER'], ecg_dat_file.filename)
        ecg_hea_path = os.path.join(app.config['UPLOAD_FOLDER'], ecg_hea_file.filename)

        fingerprint_file.save(fingerprint_path)
        ecg_dat_file.save(ecg_dat_path)
        ecg_hea_file.save(ecg_hea_path)

        # === Preprocessing ===
        ecg_base_name = os.path.splitext(ecg_dat_file.filename)[0]
        ecg_base_path = os.path.join(app.config['UPLOAD_FOLDER'], ecg_base_name)

        try:
            # === Fingerprint Prediction ===
            fp_tensor = preprocess_fingerprint_image(fingerprint_path).to(device)
            with torch.no_grad():
                fp_output = fingerprint_model(fp_tensor.unsqueeze(0))
                fp_probs = torch.softmax(fp_output, dim=1).cpu().numpy().flatten()
                fp_pred_idx = int(np.argmax(fp_probs))
                top5_fp = np.argsort(fp_probs)[-5:][::-1]
                top5_fp_labels = fp_label_encoder.inverse_transform(top5_fp)

            # === ECG Prediction ===
            ecg_tensor = preprocess_ecg_input(ecg_base_path).to(device)
            if ecg_tensor.dim() == 4:
                ecg_tensor = ecg_tensor.squeeze(1)

            print("✅ ECG Tensor Shape:", ecg_tensor.shape)
            with torch.no_grad():
                ecg_output = ecg_model(ecg_tensor)
                ecg_probs = torch.softmax(ecg_output, dim=1).cpu().numpy().flatten()
                ecg_pred_idx = int(np.argmax(ecg_probs))
                top5_ecg = np.argsort(ecg_probs)[-5:][::-1]
                top5_ecg_labels = ecg_label_encoder.inverse_transform(top5_ecg)

            # === Match Check ===
            print(f"📌 Fingerprint Predicted Index: {fp_pred_idx}")
            print(f"📌 ECG Predicted Index:        {ecg_pred_idx}")
            print("🔍 Top-5 Fingerprint Predictions:", top5_fp_labels)
            print("🔍 Top-5 ECG Predictions:", top5_ecg_labels)

            if fp_pred_idx == ecg_pred_idx:
                identity = str(fp_label_encoder.inverse_transform([fp_pred_idx])[0])
                confidence = float((fp_probs[fp_pred_idx] + ecg_probs[ecg_pred_idx]) / 2)
                print(f"✅ Matched Identity: {identity}")
                print(f"🔒 Authentication Confidence: {confidence:.4f}")

                return render_template(
                    'result.html',
                    identity=identity,
                    confidence=round(confidence * 100, 2),
                    top5_fp=top5_fp_labels,
                    top5_ecg=top5_ecg_labels
                )
            else:
                return render_template('error.html', message="❌ Authentication failed: Fingerprint and ECG don't match.")

        except Exception as e:
            traceback.print_exc()
            return render_template('error.html', message=f"🔥 Prediction error: {str(e)}")

    return render_template('index.html')

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', message="500 Internal server error.")

if __name__ == '__main__':
    app.run(debug=True)
