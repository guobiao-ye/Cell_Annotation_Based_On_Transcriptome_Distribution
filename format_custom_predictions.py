import pandas as pd
import argparse
from pathlib import Path

def main():
    """
    Main function: Converts the prediction result file of a custom model to the standard format for benchmarking.
    """
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Formats the prediction results of a custom model to interface with the standard benchmarking workflow."
    )
    parser.add_argument(
        '--input_csv', 
        required=True, 
        help="CSV file containing the prediction results and coordinates of the custom model (e.g., 'prediction_results_with_coords.csv')."
    )
    parser.add_argument(
        '--output_file', 
        required=True, 
        help="Path for the output standardized prediction CSV file."
    )
    parser.add_argument(
        '--model_name', 
        default='SPHAEC', 
        help="Give your model a name, which will be used for column names and reports. (e.g., SPHAEC, PointNet, etc.)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_file)

    # --- 2. Check and load the input file ---
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Step 1/3: Loading custom model's prediction results: {input_path.name}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- 3. Check, select, and rename columns ---
    print("Step 2/3: Formatting data...")
    
    # Define the required input columns and the new standard column names
    required_cols = ['cell_id', 'true_class', 'predicted_class']
    standard_cols = {
        'cell_id': 'cell_id',
        'true_class': 'true_label',
        'predicted_class': f'predicted_label_{args.model_name}'
    }

    # Check if all required columns exist
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: The input file is missing a required column: '{col}'")
            print(f"The available columns in the file are: {list(df.columns)}")
            return
            
    # Select the columns we need
    formatted_df = df[required_cols].copy()
    
    # Rename to the standard format
    formatted_df.rename(columns=standard_cols, inplace=True)
    
    print("Data formatting complete.")

    # --- 4. Save as a standardized file ---
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    formatted_df.to_csv(output_path, index=False)
    
    print(f"\nStep 3/3: Standardized prediction file has been successfully saved to:")
    print(output_path)
    print("\n--- Task complete! ---")
    print(f"You can now use '{output_path.parent}' as part of the input directory for analyze_benchmarks.R.")


if __name__ == '__main__':
    main()