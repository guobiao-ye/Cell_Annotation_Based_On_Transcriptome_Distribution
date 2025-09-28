import pandas as pd
import argparse
from pathlib import Path

def process_file_to_matrix_and_labels(input_csv_path, metadata_df, output_dir):
    """
    A general function to convert a gene spot list file into a count matrix and a labels file.
    
    Args:
        input_csv_path (Path): Path to the input gene spot CSV file.
        metadata_df (pd.DataFrame): DataFrame containing the ground truth labels.
        output_dir (Path): Directory to save the output files.
    """
    file_name_base = input_csv_path.stem.replace('_genes', '')
    print(f"\n--- Starting to process: {input_csv_path.name} ---")

    # 1. Load gene spot data
    print(f"  - Step 1/3: Loading gene spot data...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"    Error: File not found: {input_csv_path}")
        return

    # 2. Generate cell-gene count matrix
    print(f"  - Step 2/3: Generating count matrix...")
    # Using pd.crosstab, which is specialized for computing frequency tables and is very efficient
    # The result is a DataFrame where the index is cell_id, columns are target_gene, and values are the counts of each gene's occurrences
    count_matrix = pd.crosstab(df['cell_id'], df['target_gene'])
    
    # Save the count matrix to CSV
    matrix_output_path = output_dir / f"{file_name_base}_count_matrix.csv"
    count_matrix.to_csv(matrix_output_path)
    print(f"    Success! Count matrix saved to: {matrix_output_path}")
    print(f"    Matrix dimensions: {count_matrix.shape[0]} cells x {count_matrix.shape[1]} genes")

    # 3. Generate labels file
    print(f"  - Step 3/3: Generating labels file...")
    # Get all unique cell IDs from the current data
    cell_ids_in_data = pd.DataFrame({'cell_id': count_matrix.index.astype(str)})
    
    # Merge with metadata to get labels
    # Use a left join to keep all cells from the count matrix and match their labels
    labels_df = pd.merge(
        cell_ids_in_data,
        metadata_df,
        left_on='cell_id',
        right_on='cell_label',
        how='left'
    )
    
    # Select and clean up the final labels file
    # We only keep the ID and label columns, and remove the redundant cell_label column
    final_labels_df = labels_df[['cell_id', 'class', 'subclass', 'supertype', 'cluster']]
    
    # Save the labels file
    labels_output_path = output_dir / f"{file_name_base}_labels.csv"
    final_labels_df.to_csv(labels_output_path, index=False)
    print(f"    Success! Labels file saved to: {labels_output_path}")


def main():
    """
    Main function: Converts gene spot list files into cell-gene count matrices and corresponding label files.
    """
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Converts MERFISH gene spot data into count matrices and label files for benchmarking."
    )
    parser.add_argument('--train_csv', required=True, help="Path to the training set gene spot CSV file.")
    parser.add_argument('--test_csv', required=True, help="Path to the test set gene spot CSV file.")
    parser.add_argument('--metadata_csv', required=True, help="Metadata CSV file containing 'ground truth' cell labels.")
    parser.add_argument('--output_dir', required=True, help="Directory to save all output files.")
    args = parser.parse_args()

    # --- 2. Prepare paths and load metadata ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    
    print(f"Loading metadata file: {args.metadata_csv} ...")
    try:
        # Load all required label columns
        metadata_df = pd.read_csv(args.metadata_csv, usecols=['cell_label', 'class', 'subclass', 'supertype', 'cluster'])
        # Ensure the ID is a string for subsequent merging
        metadata_df['cell_label'] = metadata_df['cell_label'].astype(str)
    except FileNotFoundError:
        print(f"Error: Metadata file not found: {args.metadata_csv}")
        return

    # --- 3. Process the training and test sets separately ---
    process_file_to_matrix_and_labels(train_path, metadata_df, output_dir)
    process_file_to_matrix_and_labels(test_path, metadata_df, output_dir)

    print("\n--- All tasks completed! ---")

if __name__ == '__main__':
    main()