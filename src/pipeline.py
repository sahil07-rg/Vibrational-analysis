import os
Fs = 20480
folder_path = "/filedata" # we have created one folder named "filedata" locally which consists of 38 files.

files = sorted(os.listdir(folder_path))

print("Files found:", files)

anomaly_counts = []

for file in files:
    print("\nProcessing:", file)

    # Load data
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path)

    signal = data[:, 2]  # channel 2 only

    # Segmentation
    segments = create_segments(signal, Fs)

    # Feature extraction
    X = []
    for seg in segments:
        X.append(extract_features(seg, Fs))

    X = np.array(X)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Models ---

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_labels = iso.fit_predict(X_scaled)

    # One-Class SVM
    svm = OneClassSVM(nu=0.05, gamma='scale')
    svm_labels = svm.fit_predict(X_scaled)

    # Autoencoder
    ae = build_autoencoder(X_scaled.shape[1])
    ae.fit(X_scaled, X_scaled, epochs=20, batch_size=16, verbose=0)

    recon = ae.predict(X_scaled)
    mse = np.mean((X_scaled - recon)**2, axis=1)

    threshold = np.percentile(mse, 95)
    ae_labels = np.where(mse > threshold, -1, 1)

    # --- Ensemble ---
    combined = (iso_labels == -1).astype(int) + \
               (svm_labels == -1).astype(int) + \
               (ae_labels == -1).astype(int)

    strong_anomalies = np.where(combined >= 2)[0]

    count = len(strong_anomalies)
    anomaly_counts.append(count)

    print(f"Strong anomalies: {count}")
