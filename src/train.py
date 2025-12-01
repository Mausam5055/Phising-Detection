import pandas as pd
import numpy as np
import os
import argparse
from preprocessing import load_data, preprocess_data
from models import ModelTrainer
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = '../data/dataset.csv'

def main():
    parser = argparse.ArgumentParser(description="Train Phishing Detection Models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for DL models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DL models")
    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}.")
        return

    df = load_data(DATA_PATH)
    if df is None:
        return

    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Preprocessing
    X, y = preprocess_data(df)
    if X is None:
        return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_dim = X_train.shape[1]
    print(f"Input features: {input_dim}")

    trainer = ModelTrainer()

    # 3. Classical ML
    print("\n--- Classical ML ---")
    trainer.train_classical(X_train, y_train, X_test, y_test)

    # 4. Deep Learning
    print("\n--- Deep Learning ---")
    
    # MLP
    mlp = trainer.build_mlp(input_dim)
    trainer.train_dl(mlp, X_train, y_train, X_test, y_test, name="MLP (Dense)")

    # CNN
    cnn = trainer.build_cnn(input_dim)
    trainer.train_dl(cnn, X_train, y_train, X_test, y_test, name="CNN")

    # LSTM
    lstm = trainer.build_lstm(input_dim)
    trainer.train_dl(lstm, X_train, y_train, X_test, y_test, name="LSTM")

    # 5. Transformer
    print("\n--- Transformer ---")
    transformer = trainer.build_transformer(input_dim)
    trainer.train_dl(transformer, X_train, y_train, X_test, y_test, name="Transformer", epochs=args.epochs, batch_size=args.batch_size)

    # 6. Summary
    print("\n--- Final Results ---")
    results_df = pd.DataFrame(trainer.results).T
    print(results_df)
    results_df.to_csv('../docs/results_summary.csv')

if __name__ == "__main__":
    main()
