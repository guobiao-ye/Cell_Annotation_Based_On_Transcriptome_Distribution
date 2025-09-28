import os
import glob
import pandas as pd
import argparse
import random

def main():
    """
    Main function: Integrates and samples MERFISH gene data.
    1. Reads cell metadata (containing cell classes).
    2. Iterates through all normalized gene data files in a directory.
    3. For each file:
       a. Merges gene data with cell metadata to add class information to cells.
       b. Groups by cell class and randomly samples up to 5000 cells for each class.
       c. Filters for all gene records belonging to these sampled cells.
    4. Integrates the filtered gene records from all files into a single CSV file and saves it.
    """
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Samples and integrates MERFISH gene data based on cell class."
    )
    parser.add_argument(
        '--metadata', 
        required=True, 
        help="Path to the cell metadata CSV file (e.g., 'Zhuang-ABCA-2_cell_metadata_full.csv')."
    )
    parser.add_argument(
        '--input_dir', 
        required=True, 
        help="Directory containing the normalized gene data CSV files (e.g., './normalized_results/mouse1_coronal_parallel')."
    )
    parser.add_argument(
        '--output_file', 
        required=True, 
        help="Path for the final integrated output CSV file (e.g., 'sampled_genes_by_class.csv')."
    )
    args = parser.parse_args()

    # --- 2. Load cell metadata ---
    print(f"Step 1/4: Loading cell metadata file: {args.metadata}")
    try:
        # Loading only the necessary columns to save memory
        metadata_df = pd.read_csv(args.metadata, usecols=['cell_label', 'class'])
        # Ensuring cell labels are of string type for correct merging
        metadata_df['cell_label'] = metadata_df['cell_label'].astype(str)
        # Removing cells without class information
        metadata_df.dropna(subset=['class'], inplace=True)
        print(f"Successfully loaded {len(metadata_df)} valid cell metadata records.")
    except FileNotFoundError:
        print(f"Error: Metadata file not found: {args.metadata}")
        return
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return

    # --- 3. Find and process all input gene files ---
    # Finding all files starting with 'normalized_' and ending with '.csv'
    search_path = os.path.join(args.input_dir, 'normalized_*.csv')
    gene_files = glob.glob(search_path)

    if not gene_files:
        print(f"Error: No 'normalized_*.csv' files found in the directory '{args.input_dir}'.")
        return
        
    print(f"\nStep 2/4: Found {len(gene_files)} gene data files in '{args.input_dir}' to process.")

    all_sampled_genes_list = []
    
    # --- 4. Iterate through each gene file for annotation, sampling, and filtering ---
    for i, file_path in enumerate(gene_files):
        filename = os.path.basename(file_path)
        print(f"  - ({i+1}/{len(gene_files)}) Processing file: {filename}")
        
        # Reading gene data
        gene_df = pd.read_csv(file_path)
        # Similarly, ensuring cell IDs are of string type
        gene_df['cell_id'] = gene_df['cell_id'].astype(str)

        # a. Merging gene data with metadata (inner join keeps only the intersection)
        annotated_df = pd.merge(
            gene_df,
            metadata_df,
            left_on='cell_id',
            right_on='cell_label',
            how='inner'
        )

        if annotated_df.empty:
            print(f"    - Warning: No intersection between cells in file {filename} and metadata, skipping.")
            continue

        # b. Grouping by cell class and randomly sampling up to 5000 cell IDs for each class
        # First, get all unique cell IDs for each class
        cells_by_class = annotated_df.groupby('class')['cell_id'].unique()
        
        sampled_cell_ids = set()
        for class_name, cell_ids in cells_by_class.items():
            # Determine the number of samples k (up to 5000)
            k = min(len(cell_ids), 5000)
            # Randomly sample from the cells of the current class
            sampled_ids_for_class = random.sample(list(cell_ids), k)
            # Add the sampled IDs to the total set
            sampled_cell_ids.update(sampled_ids_for_class)

        print(f"    - Sampled {len(sampled_cell_ids)} cells from {len(cells_by_class)} classes.")

        # c. Filtering for all gene records belonging to these sampled cells
        final_genes_for_file = annotated_df[annotated_df['cell_id'].isin(sampled_cell_ids)].copy()
        
        # Adding a column to record the data source
        final_genes_for_file['source_file'] = filename
        
        all_sampled_genes_list.append(final_genes_for_file)
        print(f"    - Kept {len(final_genes_for_file)} gene records.")

    # --- 5. Integrate and save the final results ---
    if not all_sampled_genes_list:
        print("\nStep 3/4: No data was sampled from any file, no output file will be generated.")
        return

    print(f"\nStep 3/4: Integrating the sampling results from all {len(all_sampled_genes_list)} files...")
    final_df = pd.concat(all_sampled_genes_list, ignore_index=True)

    # Cleaning up the columns, removing the redundant 'cell_label'
    final_df = final_df.drop(columns=['cell_label'])

    print(f"\nStep 4/4: Saving a total of {len(final_df)} gene records to: {args.output_file}")
    final_df.to_csv(args.output_file, index=False)
    
    print("\n--- Processing complete ---")

if __name__ == '__main__':
    main()