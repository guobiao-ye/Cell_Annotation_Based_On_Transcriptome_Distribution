import pandas as pd
import argparse
from pathlib import Path

def main():
    """
    Main function: Reads a gene spot list file, randomly removes a certain percentage 
    of transcripts for each cell to simulate the dropout effect, and saves it as a new test file.
    """
    # --- 1. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Simulates the transcript dropout effect for the test set."
    )
    parser.add_argument(
        '--input_csv', 
        required=True, 
        help="Path to the original test set gene spot CSV file."
    )
    parser.add_argument(
        '--output_csv', 
        required=True, 
        help="Path for the new CSV file to save the downsampled test set."
    )
    parser.add_argument(
        '--keep_fraction', 
        type=float,
        default=0.7,
        help="The fraction of transcripts to keep (e.g., 0.7 means keep 70%, remove 30%)."
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help="Seed for random sampling to ensure reproducibility."
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    # --- 2. Check and load the input file ---
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
        
    print(f"Step 1/3: Loading original test set file: {input_path.name}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
        
    print(f"The original file contains {len(df)} transcripts from {df['cell_id'].nunique()} cells.")

    # --- 3. Group by cell and perform random sampling ---
    print(f"\nStep 2/3: Randomly keeping {args.keep_fraction * 100:.0f}% of transcripts for each cell...")

    # Use groupby().apply() to operate on the transcript subset for each cell
    # df.sample(frac=...) is an efficient method in pandas for random sampling
    # random_state ensures that the results are the same every time it's run
    downsampled_df = df.groupby('cell_id', group_keys=False).apply(
        lambda x: x.sample(frac=args.keep_fraction, random_state=args.random_seed)
    )
    
    # Reset the index to make it look like a regular DataFrame
    downsampled_df = downsampled_df.reset_index(drop=True)

    print("Random downsampling complete.")
    print(f"The new downsampled file contains {len(downsampled_df)} transcripts.")

    # --- 4. Save to a new CSV file ---
    print(f"\nStep 3/3: Saving the downsampled test set to: {output_path.name}")
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    downsampled_df.to_csv(output_path, index=False)
    
    print("\n--- Task complete! ---")
    print(f"You can now use '{output_path}' as a new test set to evaluate the model's robustness.")


if __name__ == '__main__':
    main()