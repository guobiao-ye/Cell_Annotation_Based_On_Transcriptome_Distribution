import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from pathlib import Path
import colorsys

def generate_color_map(genes, output_file):
    """Generates a [guaranteed unique] color map for a set of genes and saves it as a CSV file."""
    unique_genes = sorted(list(set(genes)))
    num_genes = len(unique_genes)
    colors = []
    for i in range(num_genes):
        hue = i / num_genes
        lightness = 0.5
        saturation = 0.9
        rgb_float = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb_float)
    hex_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]
    if len(set(hex_colors)) != num_genes:
        print("Warning: Duplicate colors were generated even with the new method!")
    color_map_dict = dict(zip(unique_genes, hex_colors))
    color_map_df = pd.DataFrame(list(color_map_dict.items()), columns=['gene', 'color'])
    color_map_df.to_csv(output_file, index=False)
    print(f"Successfully generated a [unique] color map for {num_genes} genes and saved it to: {output_file}")
    return color_map_dict

def plot_and_save_cell_images(df, color_map, output_dir):
    """Plots the gene distribution for each cell in the DataFrame and saves it by class."""
    grouped_by_cell = df.groupby('cell_id')
    print(f"\nStarting to generate images for {len(grouped_by_cell)} cells...")
    for cell_id, cell_df in tqdm(grouped_by_cell, desc="Processing cells"):
        cell_class = cell_df['class'].iloc[0]
        class_folder = Path(output_dir) / str(cell_class).replace(' ', '_').replace('/', '_')
        class_folder.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.scatter(
            cell_df['normalized_x'], cell_df['normalized_y'], c=cell_df['target_gene'].map(color_map),
            s=5, alpha=0.8
        )
        plt.title(f"Cell: {cell_id}\nClass: {cell_class}", fontsize=10)
        plt.xlabel("Normalized X"); plt.ylabel("Normalized Y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks([]); plt.yticks([])
        output_image_path = class_folder / f"{cell_id}.png"
        plt.savefig(output_image_path, dpi=100, bbox_inches='tight')
        plt.close()

def main():
    """Main function: Loads or generates a color map and plots cell images for the specified dataset."""
    parser = argparse.ArgumentParser(description="Generate a gene color map for MERFISH cell data and plot distribution maps.")
    parser.add_argument('--input_file', required=True, help="Input gene data CSV file.")
    parser.add_argument('--output_dir', required=True, help="Root directory to save images.")
    parser.add_argument('--color_map_file', required=True, help="[Full or relative path] to the color map file. If the file does not exist, it will be created at this path.")
    args = parser.parse_args()

    # --- 1. Prepare paths and directories ---
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Key: Directly parse the --color_map_file argument as a Path object
    color_map_path = Path(args.color_map_file)

    # --- 2. Load data ---
    print(f"Step 1/3: Loading gene data: {args.input_file}")
    try:
        gene_df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}"); return

    # --- 3. Load or generate color map ---
    print("\nStep 2/3: Processing gene color map...")
    if color_map_path.exists():
        print(f"Loading existing color map file: {color_map_path}")
        color_map_df = pd.read_csv(color_map_path)
        gene_color_map = dict(zip(color_map_df['gene'], color_map_df['color']))
    else:
        print(f"Color map file not found, creating a new one at '{color_map_path}'...")
        color_map_path.parent.mkdir(parents=True, exist_ok=True)
        gene_color_map = generate_color_map(gene_df['target_gene'], color_map_path)
    
    # --- 4. Batch plotting and saving ---
    print("\nStep 3/3: Starting to batch-generate cell images...")
    image_output_dir = output_path / 'cell_images'
    plot_and_save_cell_images(gene_df, gene_color_map, image_output_dir)

    print("\n--- All tasks completed ---")
    print(f"Cell images have been saved to the respective class folders under the '{image_output_dir}' directory.")

if __name__ == '__main__':
    main()