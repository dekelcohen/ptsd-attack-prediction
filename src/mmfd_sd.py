import numpy as np
import pandas as pd
import os
import zipfile
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, concatenate, GlobalAveragePooling1D, \
    BatchNormalization, ReLU


# Step 1: Download the WESAD Dataset
def download_wesad_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00470/WESAD.zip"
    dataset_path = "WESAD.zip"
    if not os.path.exists(dataset_path):
        os.system(f"wget {url} -O {dataset_path}")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall("WESAD")


# Step 2: Load and Preprocess the Data
def load_data():
    data_path = "WESAD/S2/S2.pkl"
    data = pd.read_pickle(data_path)
    return data


def preprocess_data(data):
    signals = np.hstack([data['signal']['wrist'][sensor] for sensor in ['ACC', 'EDA', 'TEMP', 'BVP']])
    labels = data['label']
    return signals, labels


def segment_data(signals, labels, window_size=60, overlap=0.5):
    segments = []
    segment_labels = []
    step = int(window_size * (1 - overlap))
    for start in range(0, len(signals) - window_size + 1, step):
        end = start + window_size
        segments.append(signals[start:end])
        segment_labels.append(labels[start:end])
    return np.array(segments), np.array(segment_labels)


def augment_data(signals, labels):
    smote = SMOTE()
    signals_res, labels_res = smote.fit_resample(signals, labels)
    return signals_res, labels_res


def extract_features(signals):
    time_features = np.mean(signals, axis=1)
    freq_features = np.array([welch(signal)[1] for signal in signals])
    return np.hstack((time_features.reshape(-1, 1), freq_features))


def scale_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


# Step 3: Build and Train the Model
def build_model(input_shape):
    input_signal = Input(shape=input_shape)

    # Time-domain branch
    x1 = Conv1D(64, kernel_size=3, activation='relu')(input_signal)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = GlobalAveragePooling1D()(x1)

    # Frequency-domain branch
    x2 = Conv1D(64, kernel_size=3, activation='relu')(input_signal)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = GlobalAveragePooling1D()(x2)

    # Concatenate branches
    x = concatenate([x1, x2])
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_signal, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, signals, labels):
    X_train, X_test, y_train, y_test = train_test_split(signals, labels, test_size=0.2, random_state=42)
    X_train = scale_features(X_train)
    X_test = scale_features(X_test)

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    return model


# Main function
def main():
    download_wesad_dataset()
    data = load_data()
    signals, labels = preprocess_data(data)
    segments, segment_labels = segment_data(signals, labels)
    features = extract_features(segments)
    features, labels = augment_data(features, segment_labels)
    model = build_model((features.shape[1], 1))
    trained_model = train_model(model, features, labels)
    return trained_model


# Run the process
trained_model = main()
