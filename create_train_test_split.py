import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def main():
    """
    Main function: Performs stratified sampling on the integrated gene data to create training and test sets.
    Addresses the sklearn stratified sampling error by preprocessing "orphan" data points.
    """
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Performs stratified sampling on MERFISH data to create training and test sets."
    )
    parser.add_argument(
        '--input_file', required=True, 
        help="Integrated gene data CSV file (e.g., 'final_sampled_genes_with_class.csv')."
    )
    parser.add_argument(
        '--output_dir', required=True, 
        help="Directory to save the training and test set CSV files."
    )
    parser.add_argument(
        '--train_size', type=int, default=2000, 
        help="Number of cells to include in the training set."
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help="Random seed for sampling to ensure reproducibility."
    )
    args = parser.parse_args()
    
    # Use the random seed
    np.random.seed(args.random_seed)

    # --- 2. Load data and extract cell metadata ---
    print(f"Step 1/5: Loading integrated gene data file: {args.input_file}")
    try:
        full_gene_df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}"); return
    
    print("File loaded successfully. Extracting unique cell information...")
    cell_metadata = full_gene_df[['cell_id', 'class', 'source_file']].drop_duplicates().reset_index(drop=True)
    total_cells = len(cell_metadata)
    print(f"Successfully extracted information for {total_cells} unique cells.")

    if args.train_size >= total_cells:
        print(f"Error: Requested training set size ({args.train_size}) is greater than or equal to the total number of cells ({total_cells})."); return

    # --- 3. Corrected stratified sampling logic ---
    print(f"\nStep 2/5: Performing stratified sampling to select {args.train_size} cells for the training set...")
    
    # Create the basis for stratification
    cell_metadata['strata'] = cell_metadata['class'].astype(str) + "_" + cell_metadata['source_file'].astype(str)
    
    # Calculate the number of members for each stratum
    strata_counts = cell_metadata['strata'].value_counts()
    
    # Identify "orphan" cells (their stratum has only one member)
    single_member_strata = strata_counts[strata_counts == 1].index
    orphan_cells = cell_metadata[cell_metadata['strata'].isin(single_member_strata)]
    
    # Identify "mainstream" cells (their stratum has >= 2 members)
    mainstream_cells = cell_metadata[~cell_metadata['strata'].isin(single_member_strata)]

    print(f"  - Found {len(orphan_cells)} 'orphan' cells (only 1 member in their class-source combination).")
    print(f"  - {len(mainstream_cells)} cells will undergo standard stratified sampling.")

    # Initialize the lists for training and test set cell IDs
    train_cell_ids = set()
    
    # 3.1 Handle "orphan" cells: random allocation by proportion
    # Calculate the proportion of the training set to the total
    train_ratio = args.train_size / total_cells
    
    for cell_id in orphan_cells['cell_id']:
        if len(train_cell_ids) < args.train_size and np.random.rand() < train_ratio:
            train_cell_ids.add(cell_id)

    print(f"  - {len(train_cell_ids)} 'orphan' cells were pre-allocated to the training set.")

    # 3.2 Perform stratified sampling on "mainstream" cells
    # Calculate how many more cells need to be drawn for the training set
    remaining_train_size = args.train_size - len(train_cell_ids)
    
    if remaining_train_size > 0 and not mainstream_cells.empty:
        # Perform stratified sampling on mainstream cells
        mainstream_train_df, _ = train_test_split(
            mainstream_cells,
            train_size=remaining_train_size,
            test_size=None,
            stratify=mainstream_cells['strata'],
            random_state=args.random_seed
        )
        train_cell_ids.update(mainstream_train_df['cell_id'])
    
    # The final set of test cell IDs
    all_cell_ids = set(cell_metadata['cell_id'])
    test_cell_ids = all_cell_ids - train_cell_ids

    print("Sampling complete.")
    print(f"  - Final training set cell count: {len(train_cell_ids)}")
    print(f"  - Final test set cell count: {len(test_cell_ids)}")

    # --- 4. Split gene data based on the selected cell IDs ---
    print("\nStep 3/5: Splitting gene data based on cell IDs...")
    train_gene_df = full_gene_df[full_gene_df['cell_id'].isin(train_cell_ids)]
    test_gene_df = full_gene_df[full_gene_df['cell_id'].isin(test_cell_ids)]
    print(f"  - Number of gene records in the training set: {len(train_gene_df)}")
    print(f"  - Number of gene records in the test set: {len(test_gene_df)}")

    # --- 5. Save the results to files ---
    print(f"\nStep 4/5: Checking and creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_output_path = os.path.join(args.output_dir, 'train_set_genes.csv')
    test_output_path = os.path.join(args.output_dir, 'test_set_genes.csv')
    
    print(f"\nStep 5/5: Saving files...")
    print(f"  - Saving training set to: {train_output_path}")
    train_gene_df.to_csv(train_output_path, index=False)
    
    print(f"  - Saving test set to: {test_output_path}")
    test_gene_df.to_csv(test_output_path, index=False)

    print("\n--- Processing complete ---")
    
    # (Optional) Validate the stratification effect
    print("\n--- Stratification Effect Validation (Class Distribution Percentage) ---")
    train_cells_df = cell_metadata[cell_metadata['cell_id'].isin(train_cell_ids)]
    test_cells_df = cell_metadata[cell_metadata['cell_id'].isin(test_cell_ids)]
    train_class_dist = train_cells_df['class'].value_counts(normalize=True).head(5)
    test_class_dist = test_cells_df['class'].value_counts(normalize=True).head(5)
    
    dist_df = pd.DataFrame({
        'Train Set %': train_class_dist * 100,
        'Test Set %': test_class_dist * 100
    }).fillna(0)
    print("Distribution of Top 5 cell classes in the training and test sets:")
    print(dist_df.to_string(float_format="%.2f%%"))

if __name__ == '__main__':
    main()