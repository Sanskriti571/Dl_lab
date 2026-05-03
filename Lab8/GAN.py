from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Generator
generator = Sequential([
    Dense(16, activation='relu', input_dim=5),
    Dense(1, activation='sigmoid')
])

# Discriminator
discriminator = Sequential([
    Dense(16, activation='relu', input_dim=1),
    Dense(1, activation='sigmoid')
])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN (combined)
discriminator.trainable = False

gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Dummy training
noise = np.random.rand(10, 5)
fake_data = generator.predict(noise)

print("GAN generated data:", fake_data[:5])
