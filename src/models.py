import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout, Flatten, Input

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}

    def train_classical(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate Classical ML models.
        """
        # Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        self.evaluate(lr, X_test, y_test, "Logistic Regression")

        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        self.evaluate(rf, X_test, y_test, "Random Forest")

    def build_mlp(self, input_dim):
        """
        Simple Dense Neural Network for tabular data.
        """
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_cnn(self, input_dim):
        """
        1D CNN for tabular data (treating features as a sequence).
        """
        model = Sequential([
            Input(shape=(input_dim, 1)),
            Conv1D(32, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_lstm(self, input_dim):
        """
        LSTM for tabular data.
        """
        model = Sequential([
            Input(shape=(input_dim, 1)),
            LSTM(32),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_transformer(self, input_dim):
        """
        Transformer-based model for tabular data.
        Treats the feature vector as a sequence.
        """
        inputs = Input(shape=(input_dim, 1))
        
        # Self-Attention
        attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + inputs)
        
        # Feed Forward
        outputs = Dense(32, activation="relu")(attention)
        outputs = Dense(1, activation="relu")(outputs) # Keep dimensions
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention)
        
        # Global Average Pooling to flatten
        outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
        
        # Final classification
        outputs = Dense(1, activation="sigmoid")(outputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_dl(self, model, X_train, y_train, X_test, y_test, name="DL Model", epochs=10, batch_size=32):
        print(f"Training {name}...")
        # Reshape for CNN/LSTM if needed (add channel dimension)
        if len(model.input_shape) == 3 and len(X_train.shape) == 2:
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
            
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        
        # Evaluate
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        self.store_metrics(y_test, y_pred, name)

    def evaluate(self, model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        self.store_metrics(y_test, y_pred, name)

    def store_metrics(self, y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        self.results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }
        print(f"Results for {name}: {self.results[name]}")
