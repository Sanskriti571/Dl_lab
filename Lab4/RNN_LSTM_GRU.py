import time
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding

# 1. Load dataset
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# 2. Preprocess (padding)
max_length = 100
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# 3. Model builder
def build_model(model_type):
    model = Sequential()
    
    model.add(Embedding(vocab_size, 32, input_length=max_length))
    
    if model_type == "RNN":
        model.add(SimpleRNN(32))
    elif model_type == "LSTM":
        model.add(LSTM(32))
    elif model_type == "GRU":
        model.add(GRU(32))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 4. Train & Compare
results = {}

for model_type in ["RNN", "LSTM", "GRU"]:
    print(f"\nTraining {model_type}...")
    
    model = build_model(model_type)
    
    start = time.time()
    
    model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)
    
    end = time.time()
    
    loss, accuracy = model.evaluate(X_test, y_test)
    
    results[model_type] = {
        "accuracy": accuracy,
        "time": end - start
    }

# 5. Results
print("\n Final Results:")
for model in results:
    print(f"{model} → Accuracy: {results[model]['accuracy']:.4f}, Time: {results[model]['time']:.2f} sec")
