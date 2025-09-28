import scanpy as sc
import scvi
import pandas as pd
import argparse
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, f1_score

def main():
    start_time_total = time.time()
    
    parser = argparse.ArgumentParser(description="Annotate cells using scANVI and output predictions in a standard format.")
    parser.add_argument('--train_matrix', required=True)
    parser.add_argument('--train_labels', required=True)
    parser.add_argument('--test_matrix', required=True)
    parser.add_argument('--test_labels', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_predictions_path = output_dir / "scanvi_predictions.csv"
    report_path = output_dir / 'report_scanvi.txt'
    
    # --- 1. Data Loading ---
    print("--- Step 1/4: Loading Data ---")
    start_time_load = time.time()
    
    train_counts = pd.read_csv(args.train_matrix, index_col=0)
    train_labels = pd.read_csv(args.train_labels, index_col=0)
    test_counts = pd.read_csv(args.test_matrix, index_col=0)
    test_labels = pd.read_csv(args.test_labels, index_col=0)
    
    adata_train = sc.AnnData(train_counts)
    adata_train.obs['label'] = train_labels.loc[adata_train.obs_names, 'class']
    
    adata_test = sc.AnnData(test_counts)
    adata_test.obs['label'] = test_labels.loc[adata_test.obs_names, 'class']
    
    # Place the objects to be merged into a dictionary, the keys will be used as batch names
    adata = sc.concat(
        {"train": adata_train, "test": adata_test},
        label="batch" # This parameter specifies the name of the obs column to store batch information
    )
    
    load_time = time.time() - start_time_load
    print(f"Data loading complete.")

    # --- 2. Model Setup and Training ---
    print("\n--- Step 2/4: Setting up and training the scANVI model ---")
    start_time_train = time.time()
    
    adata.obs['label_for_scanvi'] = adata.obs['label'].copy()
    adata.obs.loc[adata.obs['batch'] == 'test', 'label_for_scanvi'] = 'Unknown'
    
    scvi.model.SCVI.setup_anndata(adata, batch_key='batch', labels_key='label_for_scanvi')
    
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    vae.train()
    
    lvae = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown")
    lvae.train(max_epochs=20)
    
    train_time = time.time() - start_time_train
    print("Model training complete.")

    # --- 3. Prediction and Saving Results ---
    print("\n--- Step 3/4: Predicting on the test set and saving results ---")
    start_time_pred = time.time()
    
    adata_test_anvi = adata[adata.obs.batch == "test"].copy()
    y_pred = lvae.predict(adata_test_anvi)
    
    results_df = pd.DataFrame({
        'cell_id': adata_test_anvi.obs_names,
        'true_label': adata_test_anvi.obs['label'],
        'predicted_label_scanvi': y_pred
    })
    
    results_df.to_csv(output_predictions_path, index=False)
    pred_time = time.time() - start_time_pred
    print(f"Predictions in standard format have been saved to: {output_predictions_path}")

    # --- 4. Calculate Basic Metrics and Generate Report ---
    print("\n--- Step 4/4: Generating a simple report ---")
    y_true = results_df['true_label']
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    with open(report_path, 'w') as f:
        f.write("========== scANVI Performance Evaluation Report (Brief) ==========\n\n")
        f.write(f"Dataset: {Path(args.train_matrix).name.replace('_count_matrix.csv', '')}\n")
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