import numpy as np
from scipy.signal import welch
sig2=cleaned_data[:,2]
Fs=20480
def extract_features(sig2, Fs):
  rms=np.sqrt(np.mean(sig2**2))
  std=np.std(sig2) #creating std for fecture vector
  from scipy.stats import kurtosis # importing kurtosis library
  kurt=kurtosis(sig2,fisher=False) #using false kurtosis
  freqs, psd = welch(sig2, fs=Fs, nperseg=2048)
  def band_power(freqs, psd, f_low, f_high): #defining band power function. IMP- always PSD must be defined before using band power function.
      idx = (freqs >= f_low) & (freqs <= f_high)
      return np.trapezoid(psd[idx], freqs[idx])
  bp_1k = band_power(freqs, psd, 800, 1200)
  centroid=np.sum(freqs*psd)/np.sum(psd) # calculating centroid for fecture extration.
  bandwidth=np.sqrt(np.sum(((freqs-centroid)**2)*psd)/np.sum(psd)) # created for bandwidth feature.
  flatness=np.exp(np.mean(np.log(psd)))/np.mean (psd) #created flatness for feature extration.
  psd_norm=psd/np.sum(psd) #creating spectral entropy
  entropy=-np.sum(psd_norm*np.log(psd_norm))
  #we are adding advanced features of vibrations for this dataset.
  peak=np.max(np.abs(sig2))
  mean_abs=np.mean(np.abs(sig2))
  mean_sqrt=np.mean(np.sqrt(np.abs(sig2)))
  crest_factor=peak/rms
  shape_factor=rms/mean_abs
  impulsive_factor=peak/mean_abs
  clearance_factor=peak/(mean_sqrt**2)

  #Now creating feature vector:
  feature_vector=[rms,
                  std,
                  kurt,
                  bp_1k,
                  centroid,
                  bandwidth,
                  flatness,
                  entropy,
                  crest_factor,
                  shape_factor,
                  impulsive_factor,
                  clearance_factor
                  ]
  return feature_vector


