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
