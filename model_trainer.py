# Importing Libraries
import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Configurations
DATA_DIR = "dataset_sequences"
SEQUENCE_LENGTH = 30
N_FEATURES = 60
BATCH_SIZE = 32
EPOCHS = 60
MODEL_FILENAME = "sign_seq_translator_v2.h5"
RANDOM_SEED = 42

# Loads all .pkl sequences from the data directory
def load_dataset(data_dir: str):
    data_dir = Path(data_dir)
    labels = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    label2idx = {lbl: i for i, lbl in enumerate(labels)}
    
    X, y = [], []
    for lbl in labels:
        folder = data_dir / lbl
        for file in folder.glob("*.pkl"):
            try:
                with open(file, "rb") as f:
                    seq = pickle.load(f)
                
                seq = np.array(seq, dtype=np.float32)
                if seq.shape != (SEQUENCE_LENGTH, N_FEATURES):
                    print(f"Skipping {file}: Incorrect shape {seq.shape}. Expected ({SEQUENCE_LENGTH}, {N_FEATURES}).")
                    continue
                    
                X.append(seq)
                y.append(label2idx[lbl])
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")

    X = np.array(X)
    y = np.array(y, dtype=np.int32)
    return X, y, label2idx

# Defines the Bidirectional LSTM model architecture
def build_model(seq_len: int, n_features: int, n_classes: int):
    tf.random.set_seed(RANDOM_SEED)
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), input_shape=(seq_len, n_features)),
        Dropout(0.35),
        Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
        Dropout(0.35),
        Dense(32, activation='relu'),
        Dropout(0.25),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def main():
    print("Loading dataset from:", DATA_DIR)
    X, y, label2idx = load_dataset(DATA_DIR)
    
    if len(X) == 0:
        raise RuntimeError(f"No sequences found in '{DATA_DIR}'. Run translator_app.py in RECORD mode first.")

    n_classes = len(label2idx)
    print(f"Dataset: {X.shape[0]} samples, {n_classes} classes found: {list(label2idx.keys())}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    model = build_model(SEQUENCE_LENGTH, N_FEATURES, n_classes)

    ckpt = ModelCheckpoint(MODEL_FILENAME, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[ckpt, early],
        verbose=2
    )
    
    label_map_path = "label_map.pkl"
    with open(label_map_path, "wb") as f:
        pickle.dump(label2idx, f)
    print(f"Label map saved to {label_map_path}. Contents: {label2idx}")
    print(f"Training finished. Best model saved to: {MODEL_FILENAME}")

if __name__ == "__main__":
    main()