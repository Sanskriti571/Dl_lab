from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

# Dummy data
X = np.random.rand(100, 20)

input_dim = 20
latent_dim = 2

# Encoder
inputs = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(inputs)
latent = Dense(latent_dim)(encoded)

# Decoder
decoded = Dense(10, activation='relu')(latent)
outputs = Dense(input_dim, activation='sigmoid')(decoded)

# Model
vae = Model(inputs, outputs)
vae.compile(optimizer='adam', loss='mse')

# Train
vae.fit(X, X, epochs=5, batch_size=10)

print("VAE training complete")
