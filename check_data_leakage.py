import pandas as pd
import argparse
from pathlib import Path

def main():
    """
    Main function: Checks for data leakage based on 'cell_id' between the training and test sets.
    """
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Checks for any intersection of cell IDs (data leakage) between training and test set CSV files."
    )
    parser.add_argument(
        '--train_csv', 
        required=True, 
        help="Path to the training set CSV file."
    )
    parser.add_argument(
        '--test_csv', 
        required=True, 
        help="Path to the test set CSV file."
    )
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)

    # --- 2. Check if files exist ---
    if not train_path.exists():
        print(f"Error: Training set file not found: {train_path}")
        return
    if not test_path.exists():
        print(f"Error: Test set file not found: {test_path}")
        return
        
    # --- 3. Load data and extract cell IDs ---
    print("Starting data leakage check...")
    
    try:
        print(f"  - Loading training set: {train_path.name}")
        # Load only the 'cell_id' column for efficiency
        train_df = pd.read_csv(train_path, usecols=['cell_id'])
        
        print(f"  - Loading test set: {test_path.name}")
        test_df = pd.read_csv(test_path, usecols=['cell_id'])
    except ValueError:
        print("Error: The 'cell_id' column is missing in one or both files. Please check the file contents.")
        return
        
    # --- 4. Extract unique cell IDs and find the intersection ---
    # Using the set data structure because its intersection operation is very fast
    train_cell_ids = set(train_df['cell_id'].unique())
    test_cell_ids = set(test_df['cell_id'].unique())
    
    print(f"\n  - Number of unique cells in the training set: {len(train_cell_ids)}")
    print(f"  - Number of unique cells in the test set: {len(test_cell_ids)}")
    
    # Calculate the intersection
    intersection = train_cell_ids.intersection(test_cell_ids)
    num_intersection = len(intersection)
    
    # --- 5. Report the results ---
    print("\n--- Check Results ---")
    
    if num_intersection == 0:
        print("✅ Check Passed!")
        print("No common cell IDs were found between the training and test sets.")
        print("The data split is clean, with no data leakage.")
    else:
        print("❌ Check Failed! Data leakage detected!")
        print(f"Found {num_intersection} common cell IDs between the training and test sets.")
        print("This means the model has seen test data during training, and the evaluation results will be unreliable.")
        print("\nHere are some of the leaked cell IDs (showing up to 10):")
        for i, cell_id in enumerate(list(intersection)):
            if i >= 10:
                break
            print(f"  - {cell_id}")
            
    print("------------------")

if __name__ == '__main__':
    main()