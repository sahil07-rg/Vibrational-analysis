import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis
def extract_features(segment, Fs):
    all_features = []

    # Loop through all channels
    for ch in range(segment.shape[1]):
        signals = segment[:, ch]

        # ---- Time Domain ----
        rms = np.sqrt(np.mean(signals**2))
        std = np.std(signals)
        kurt = kurtosis(signals, fisher=False)

        # ---- Frequency Domain ----
        freqs, psd = welch(signals, fs=Fs)

        centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
        bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * psd) / (np.sum(psd) + 1e-12))
        flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)

        psd_norm = psd / (np.sum(psd) + 1e-12)
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))

        # ---- Shape Features ----
        peak = np.max(np.abs(signals))
        mean_abs = np.mean(np.abs(signals))
        mean_sqrt = np.mean(np.sqrt(np.abs(signals)))

        crest = peak / (rms + 1e-12)
        shape = rms / (mean_abs + 1e-12)
        impulse = peak / (mean_abs + 1e-12)
        clearance = peak / (mean_sqrt**2 + 1e-12)

        features = [
            rms, std, kurt,
            centroid, bandwidth, flatness, entropy,
            crest, shape, impulse, clearance
        ]

        all_features.extend(features)

    return np.array(all_features)

