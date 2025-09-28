import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def main():
    """
    Main function: Performs stratified sampling on the integrated gene data to create training and test sets.
    This version stratifies based on SUBCLASS.
    """
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Performs stratified sampling on MERFISH data based on SUBCLASS."
    )
    parser.add_argument(
        '--input_file', required=True, 
        help="Integrated gene data CSV file."
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
        help="Random seed for sampling."
    )
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)

    # --- 2. Load data and extract cell metadata ---
    print(f"Step 1/5: Loading integrated gene data file: {args.input_file}")
    try:
        full_gene_df = pd.read_csv(args.input_file)
        # Ensure the subclass column exists and has the correct type
        if 'subclass' not in full_gene_df.columns:
            print("Error: 'subclass' column not found in the input file.")
            return
        full_gene_df['subclass'] = full_gene_df['subclass'].astype(str)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}"); return
    
    print("File loaded successfully. Extracting unique cell information...")
    cell_metadata = full_gene_df[['cell_id', 'class', 'subclass', 'source_file']].drop_duplicates().reset_index(drop=True)
    total_cells = len(cell_metadata)
    print(f"Successfully extracted information for {total_cells} unique cells.")

    if args.train_size >= total_cells:
        print(f"Error: Requested training set size ({args.train_size}) is greater than or equal to the total number of cells ({total_cells})."); return

    # --- 3. Corrected stratified sampling logic ---
    print(f"\nStep 2/5: Performing stratified sampling (by subclass) to select {args.train_size} cells for the training set...")
    
    cell_metadata['strata'] = cell_metadata['subclass'].astype(str) + "_" + cell_metadata['source_file'].astype(str)
    
    strata_counts = cell_metadata['strata'].value_counts()
    
    single_member_strata = strata_counts[strata_counts == 1].index
    orphan_cells = cell_metadata[cell_metadata['strata'].isin(single_member_strata)]
    
    mainstream_cells = cell_metadata[~cell_metadata['strata'].isin(single_member_strata)]

    print(f"  - Found {len(orphan_cells)} 'orphan' cells (only 1 member in their subclass-source combination).")
    print(f"  - {len(mainstream_cells)} cells will undergo standard stratified sampling.")

    train_cell_ids = set()
    
    train_ratio = args.train_size / total_cells
    
    for cell_id in orphan_cells['cell_id']:
        if len(train_cell_ids) < args.train_size and np.random.rand() < train_ratio:
            train_cell_ids.add(cell_id)

    print(f"  - {len(train_cell_ids)} 'orphan' cells were pre-allocated to the training set.")

    remaining_train_size = args.train_size - len(train_cell_ids)
    
    if remaining_train_size > 0 and not mainstream_cells.empty:
        # Check if stratified sampling is possible
        if remaining_train_size < mainstream_cells['strata'].nunique():
            print(f"Warning: The remaining training set size ({remaining_train_size}) is smaller than the number of strata in the mainstream cells ({mainstream_cells['strata'].nunique()}).")
            print("Switching to simple random sampling for all mainstream cells.")
            mainstream_train_df = mainstream_cells.sample(n=remaining_train_size, random_state=args.random_seed)
        else:
            mainstream_train_df, _ = train_test_split(
                mainstream_cells,
                train_size=remaining_train_size,
                test_size=None,
                stratify=mainstream_cells['strata'],
                random_state=args.random_seed
            )
        train_cell_ids.update(mainstream_train_df['cell_id'])
    
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
    train_gene_df.to_csv(train_output_path, index=False)
    test_gene_df.to_csv(test_output_path, index=False)
    print(f"  - Training set saved to: {train_output_path}")
    print(f"  - Test set saved to: {test_output_path}")

    print("\n--- Processing complete ---")
    
    # (Optional) Validate the stratification effect
    print("\n--- Stratification Effect Validation (Subclass Distribution Percentage) ---")
    train_cells_df = cell_metadata[cell_metadata['cell_id'].isin(train_cell_ids)]
    test_cells_df = cell_metadata[cell_metadata['cell_id'].isin(test_cell_ids)]
    
    train_subclass_dist = train_cells_df['subclass'].value_counts(normalize=True).head(5)
    test_subclass_dist = test_cells_df['subclass'].value_counts(normalize=True).head(5)
    
    dist_df = pd.DataFrame({
        'Train Set %': train_subclass_dist * 100,
        'Test Set %': test_subclass_dist * 100
    }).fillna(0)
    print("Distribution of Top 5 cell subclasses in the training and test sets:")
    print(dist_df.to_string(float_format="%.2f%%"))

if __name__ == '__main__':
    main()