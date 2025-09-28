import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# --- 1. Import definitions from the correct training script ---
# Make sure this filename exactly matches the 3D training script
try:
    from train_classifier_3D import PointNetClassifier, CellGeneDataset, collate_fn
except ImportError:
    print("Error: Could not import necessary classes from 'train_classifier_3D.py'.")
    print("Please ensure that the 'train_classifier_3D.py' file is in the same directory as this script.")
    exit()

# --- 2. Prediction Function (Unchanged) ---
def predict(model, dataloader, device, idx_to_class):
    """
    Runs the model on the test set and returns the prediction results.
    
    Returns:
        A dictionary mapping cell_id to (predicted_class, true_class).
    """
    model.eval()
    results = {}
    
    # Since the DataLoader's shuffle=False, we can get the cell_ids in order
    cell_ids = dataloader.dataset.cell_ids
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Predicting")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Record the results for each sample in the batch
            batch_size = inputs.size(0)
            for j in range(batch_size):
                cell_idx_in_dataset = i * dataloader.batch_size + j
                if cell_idx_in_dataset < len(cell_ids):
                    cell_id = cell_ids[cell_idx_in_dataset]
                    true_class_name = idx_to_class[labels[j].item()]
                    pred_class_name = idx_to_class[preds[j].item()]
                    results[cell_id] = {
                        'true_class': true_class_name,
                        'predicted_class': pred_class_name
                    }
    return results

# --- 3. Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Uses a trained model to make predictions and saves the results integrated with 3D coordinates to a CSV file."
    )
    parser.add_argument('--model_path', required=True, help='Path to the model weights file (.pth)')
    parser.add_argument('--test_csv', required=True, help='Path to the test set CSV file')
    parser.add_argument('--train_csv', required=True, help='Path to the training set CSV file (for reconstructing mappings)')
    parser.add_argument('--metadata_csv', required=True, help='Cell 3D coordinate metadata file')
    parser.add_argument('--color_map', required=True, help='Path to the gene color map file')
    parser.add_argument(
        '--output_csv', 
        default='./prediction_results_with_coords.csv', 
        help='Path to the output CSV file containing prediction results and coordinates'
    )
    args = parser.parse_args()

    # Create the parent directory for the output file
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load mappings ---
    print("\nStep 1/4: Loading mappings...")
    gene_map_df = pd.read_csv(args.color_map)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_map_df['gene'])}
    num_genes = len(gene_to_idx)
    
    train_df_for_meta = pd.read_csv(args.train_csv)
    unique_classes = sorted(train_df_for_meta['class'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # --- Step 2: Load the model ---
    print("\nStep 2/4: Loading the trained model...")
    num_features = 3 + num_genes
    model = PointNetClassifier(num_features, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # --- Step 3: Perform prediction ---
    print("\nStep 3/4: Preparing test data and performing prediction...")
    test_dataset = CellGeneDataset(args.test_csv, class_to_idx, gene_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    predictions_dict = predict(model, test_loader, device, idx_to_class)
    predictions_df = pd.DataFrame.from_dict(predictions_dict, orient='index').reset_index().rename(columns={'index': 'cell_id'})

    # --- Step 4: Integrate data and save ---
    print("\nStep 4/4: Integrating data and saving to CSV...")
    spatial_df = pd.read_csv(args.metadata_csv, usecols=['cell_label', 'x', 'y', 'z'])
    spatial_df['cell_label'] = spatial_df['cell_label'].astype(str)

    final_df = pd.merge(predictions_df, spatial_df, left_on='cell_id', right_on='cell_label', how='inner')
    final_df = final_df.drop(columns=['cell_label'])
    
    if final_df.empty:
        print("Error: No intersection between prediction results and spatial metadata."); return
        
    print(f"Successfully merged prediction results and spatial coordinates for {len(final_df)} cells.")

    final_df.to_csv(output_path, index=False)
    print(f"\nDetailed prediction results have been successfully saved to: {output_path}")
    print("\n--- Prediction task complete! ---")

if __name__ == '__main__':
    main()