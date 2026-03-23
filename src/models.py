from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

# Moved to our next model for cross verification of the anamolies.
from sklearn.svm import OneClassSVM
svm_model=OneClassSVM(
    kernel='rbf',
    nu=0.05,
    gamma='scale'
)
svm_labels=svm_model.fit_predict(X_scaled)
print(svm_labels)

# One of the best models for anomalies detection as per the previous research.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = X_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_scaled, X_scaled,
                epochs=50,
                batch_size=16,
                verbose=0)
# did reconstruction as a part of our AE.
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean((X_scaled - reconstructions)**2, axis=1)

threshold = np.percentile(mse, 95)

ae_labels = np.where(mse > threshold, -1, 1)
print(ae_labels) # printing anomalies found by AE.
