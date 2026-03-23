import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

files = sorted(os.listdir(folder_path))
print("Files found:", files)

anomaly_counts = []
anomaly_scores = []
all_anomaly_indices = []

for file in files:

    print(f"\nProcessing: {file}")

    # -------- LOAD FILE --------
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path)
    signals = data[:, :8]

    # -------- SEGMENTATION --------
    segments = []
    for start in range(0, len(signals) - window_size + 1, window_size):
        segment = signals[start:start + window_size, :]
        segments.append(segment)

    # -------- FEATURE EXTRACTION --------
    X = []
    for segment in segments:
        X.append(extract_features(segment, Fs))

    X = np.array(X)

    print("Feature variance:", np.var(X))  # debug

    # -------- SCALING --------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------- MODELS --------
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_labels = iso.fit_predict(X_scaled)

    svm = OneClassSVM(nu=0.1)
    svm_labels = svm.fit_predict(X_scaled)

    # -------- AUTOENCODER --------
    ae = build_autoencoder(X_scaled.shape[1])
    ae.fit(X_scaled, X_scaled, epochs=10, batch_size=16, verbose=0)

    recon = ae.predict(X_scaled)
    mse = np.mean((X_scaled - recon)**2, axis=1)

    threshold_ae = np.percentile(mse, 95)
    ae_labels = np.where(mse > threshold_ae, -1, 1)

    # -------- ENSEMBLE --------
    iso_bin = (iso_labels == -1).astype(int)
    svm_bin = (svm_labels == -1).astype(int)
    ae_bin  = (ae_labels == -1).astype(int)

    print("ISO:", np.sum(iso_bin),
          "SVM:", np.sum(svm_bin),
          "AE:", np.sum(ae_bin))

    # -------- ADAPTIVE WEIGHTS --------
    iso_conf = np.mean(iso_bin)
    svm_conf = np.mean(svm_bin)
    ae_conf  = np.mean(ae_bin)

    total_conf = iso_conf + svm_conf + ae_conf + 1e-6

    w_iso = iso_conf / total_conf
    w_svm = svm_conf / total_conf
    w_ae  = ae_conf  / total_conf

    # -------- WEIGHTED SCORE --------
    weighted_score = (
        w_iso * iso_bin +
        w_svm * svm_bin +
        w_ae  * ae_bin
    )

    final_anomalies = (weighted_score >= 0.5).astype(int)

    # -------- RESULTS --------
    anomaly_indices = np.where(final_anomalies == 1)[0]
    anomaly_count = np.sum(final_anomalies)
    anomaly_score = np.sum(weighted_score)

    print(f"Count: {anomaly_count}")
    print(f"Score: {anomaly_score:.2f}")
    print(f"Indices: {anomaly_indices}")

    anomaly_counts.append(anomaly_count)
    anomaly_scores.append(anomaly_score)
    all_anomaly_indices.append(anomaly_indices)
