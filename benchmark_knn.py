import pandas as pd
import argparse
from pathlib import Path
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def main():
    start_time_total = time.time()
    
    parser = argparse.ArgumentParser(description="Annotate cells using a k-NN classifier and output predictions in a standard format.")
    parser.add_argument('--train_matrix', required=True)
    parser.add_argument('--train_labels', required=True)
    parser.add_argument('--test_matrix', required=True)
    parser.add_argument('--test_labels', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--n_neighbors', type=int, default=10, help="The value of k in k-NN")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_predictions_path = output_dir / "knn_predictions.csv"
    report_path = output_dir / 'report_knn.txt'
    
    # --- 1. Data Loading ---
    print("--- Step 1/4: Loading Data ---")
    start_time_load = time.time()
    train_counts = pd.read_csv(args.train_matrix, index_col=0)
    train_labels = pd.read_csv(args.train_labels, index_col=0)
    test_counts = pd.read_csv(args.test_matrix, index_col=0)
    test_labels = pd.read_csv(args.test_labels, index_col=0)
    
    common_genes = train_counts.columns.intersection(test_counts.columns)
    train_counts = train_counts[common_genes]
    test_counts = test_counts[common_genes]
    
    X_train = train_counts.values
    y_train = train_labels.loc[train_counts.index, 'class'].values.ravel()
    X_test = test_counts.values
    y_test_true = test_labels.loc[test_counts.index, 'class'].values.ravel()
    
    load_time = time.time() - start_time_load
    print("Data loading and alignment complete.")

    # --- 2. Model Training ---
    print(f"\n--- Step 2/4: Training k-NN Model (k={args.n_neighbors}) ---")
    start_time_train = time.time()
    model = KNeighborsClassifier(n_neighbors=args.n_neighbors, n_jobs=-1)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time_train
    print("Model training complete.")

    # --- 3. Prediction and Saving Results ---
    print("\n--- Step 3/4: Predicting on the test set and saving results ---")
    start_time_pred = time.time()
    y_pred = model.predict(X_test)
    
    results_df = pd.DataFrame({
        'cell_id': test_counts.index,
        'true_label': y_test_true,
        'predicted_label_knn': y_pred
    })
    
    results_df.to_csv(output_predictions_path, index=False)
    pred_time = time.time() - start_time_pred
    print(f"Predictions in standard format have been saved to: {output_predictions_path}")

    # --- 4. Calculate Basic Metrics and Generate Report ---
    print("\n--- Step 4/4: Generating a simple report ---")
    accuracy = accuracy_score(y_test_true, y_pred)
    f1_macro = f1_score(y_test_true, y_pred, average='macro', zero_division=0)
    
    with open(report_path, 'w') as f:
        f.write("========== k-NN Performance Evaluation Report (Brief) ==========\n\n")
        f.write(f"Dataset: {Path(args.train_matrix).name.replace('_count_matrix.csv', '')}\n")
        f.write(f"k-value: {args.n_neighbors}\n")
        f.write("-" * 40 + "\n")
        f.write("Key Metrics:\n")
        f.write(f"  - Accuracy: {accuracy:.4f}\n")
        f.write(f"  - F1-Score (Macro): {f1_macro:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write("Execution Time:\n")
        f.write(f"  - Data Loading: {load_time:.2f} seconds\n")
        f.write(f"  - Model Training: {train_time:.2f} seconds\n")
        f.write(f"  - Prediction & Saving: {pred_time:.2f} seconds\n")
        f.write(f"  - Total: {time.time() - start_time_total:.2f} seconds\n")
    print(f"A simple report has been saved to: {report_path}")

if __name__ == "__main__":
    main()