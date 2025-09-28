import pandas as pd
import argparse

def main():
    """
    Main function: Reads a cell metadata file and counts the number of categories 
    at different hierarchical levels (class, subclass, supertype, cluster).
    """
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Counts the number of 'ground truth' categories at different hierarchical levels in a cell metadata file."
    )
    parser.add_argument(
        '--metadata_csv', 
        required=True, 
        help="Metadata CSV file containing 'ground truth' cell labels (e.g., 'Zhuang-ABCA-2_cell_metadata_full.csv')."
    )
    args = parser.parse_args()

    # --- 2. Load data ---
    print(f"Loading metadata file: {args.metadata_csv} ...")
    try:
        # Load only the columns we are interested in to save memory
        columns_to_load = ['class', 'subclass', 'supertype', 'cluster']
        df = pd.read_csv(args.metadata_csv, usecols=columns_to_load)
        print("File loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found: {args.metadata_csv}")
        return
    except ValueError as e:
        print(f"Error: The file might be missing necessary columns. Please ensure the file contains {columns_to_load}.")
        print(f"Detailed error: {e}")
        return

    # --- 3. Count and print results ---
    print("\n--- Ground Truth Category Statistics ---")
    
    for column_name in columns_to_load:
        # Use the .nunique() method to directly count the number of unique values; 
        # it automatically handles NaNs (does not count them)
        num_unique_categories = df[column_name].nunique()
        
        print(f"Number of unique categories in column '{column_name}': {num_unique_categories}")
        
    print("\n------------------------------------")
    print("Counting complete.")


if __name__ == '__main__':
    main()