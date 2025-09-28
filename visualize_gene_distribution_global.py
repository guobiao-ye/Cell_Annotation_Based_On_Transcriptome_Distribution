import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.colors as colors

def plot_global_density(df, output_dir):
    """
    Plots the global density of all genes in the normalized coordinate system.
    """
    print("Generating global gene density plot...")
    plt.figure(figsize=(10, 10))
    plt.hexbin(df['normalized_x'], df['normalized_y'], gridsize=100, cmap='inferno', bins='log')
    plt.colorbar(label='Log(Number of Gene Spots)')
    plt.xlabel('Normalized X Coordinate')
    plt.ylabel('Normalized Y Coordinate')
    plt.title('Global Normalized Density Distribution of All Genes in All Cells')
    plt.axhline(0, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axvline(0, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_dir, '1_global_gene_density.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Global gene density plot saved.")

def plot_specific_gene_distribution_fast(df, output_dir, n_genes=5):
    """
    Plots the normalized distribution for the N most common genes using a fast 2D histogram.
    This version is robust against NaN values.
    """
    print(f"Generating FAST distribution plots for the top {n_genes} most common genes...")
    
    top_genes = df['target_gene'].value_counts().nlargest(n_genes).index.tolist()

    # First filter for top genes, then remove all possible NaN values at once
    gene_subset_df = df[df['target_gene'].isin(top_genes)].dropna(subset=['normalized_x', 'normalized_y'])
    
    # Check if there is still data after cleaning
    if gene_subset_df.empty:
        print(f"Warning: No valid (non-NaN) data found for the top {n_genes} genes. Skipping this plot.")
        return

    # Determine the common coordinate range for all top genes to ensure all subplots have consistent axes
    x_min, x_max = gene_subset_df['normalized_x'].min(), gene_subset_df['normalized_x'].max()
    y_min, y_max = gene_subset_df['normalized_y'].min(), gene_subset_df['normalized_y'].max()

    fig, axes = plt.subplots(1, n_genes, figsize=(n_genes * 5, 5.5))
    if n_genes == 1:
        axes = [axes]
        
    fig.suptitle(f'Top {n_genes} Most Common Genes: Normalized Spatial Density (Fast Plot)', fontsize=16)

    for i, gene in enumerate(top_genes):
        # Get gene data from the already cleaned subset
        gene_df = gene_subset_df[gene_subset_df['target_gene'] == gene]
        
        # Double check in case all data for a specific gene was NaN
        if gene_df.empty:
            ax = axes[i]
            ax.set_title(f'Gene: {gene}\n(No valid data)')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax = axes[i]
        
        # Use a fast 2D histogram (hist2d)
        h = ax.hist2d(
            gene_df['normalized_x'], gene_df['normalized_y'],
            bins=100,
            cmap="viridis",
            norm=colors.LogNorm()
        )
        
        ax.set_title(f'Gene: {gene}\n(n={len(gene_df):,})')
        ax.set_xlabel('Normalized X')
        if i == 0:
            ax.set_ylabel('Normalized Y')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')

    fig.colorbar(h[3], ax=axes.ravel().tolist(), shrink=0.8, label="Density (log scale)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, '2_specific_gene_distributions_fast.png'), dpi=300)
    plt.close()
    print("Specific gene distribution (fast) plots saved.")

def plot_single_cell_view(df, output_dir, n_cells=4):
    """
    Visualizes the gene distribution within N randomly selected cells.
    """
    print(f"Generating {n_cells} random single-cell views...")
    cell_counts = df['cell_id'].value_counts()
    high_count_cells = cell_counts[cell_counts > 50].index
    if len(high_count_cells) < n_cells:
        print("Warning: Not enough cells with high gene counts. Sampling from all cells.")
        high_count_cells = cell_counts.index
        
    random_cells = np.random.choice(high_count_cells, size=min(n_cells, len(high_count_cells)), replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    fig.suptitle('Gene Distribution within Random Single Cells (Normalized Coordinates)', fontsize=16)

    for i, cell_id in enumerate(random_cells):
        cell_df = df[df['cell_id'] == cell_id]
        ax = axes[i]
        
        sns.scatterplot(
            data=cell_df, x='normalized_x', y='normalized_y', 
            hue='z', palette='bright', s=10, ax=ax, legend='full'
        )
        
        ax.set_title(f'Cell ID: {cell_id}\n(Total spots: {len(cell_df):,})')
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(title='Z-Plane')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, '3_single_cell_views.png'), dpi=300)
    plt.close()
    print("Single-cell view plots saved.")

def plot_radial_distribution(df, output_dir):
    """
    Plots the radial distribution of genes from the cell center.
    """
    print("Generating radial distribution analysis plot...")
    if 'radial_dist' not in df.columns:
        df['radial_dist'] = np.sqrt(df['normalized_x']**2 + df['normalized_y']**2)
    
    top_genes = df['target_gene'].value_counts().nlargest(5).index.tolist()
    
    plt.figure(figsize=(12, 7))
    
    # Use histplot, which is more efficient for large data
    sns.histplot(df['radial_dist'], bins=100, color='grey', stat='density', label='All Genes')
    
    # kdeplot can still be used to compare trends, but only on subsets
    for gene in top_genes:
        # Downsample each gene for kdeplot to speed up
        gene_df = df[df['target_gene'] == gene]
        sample_size = min(50000, len(gene_df)) # Use at most 50,000 points to draw the KDE
        sns.kdeplot(gene_df['radial_dist'].sample(sample_size), label=f'Gene: {gene}', linewidth=2)
        
    plt.title('Radial Distribution Density from Cell Center (0,0)')
    plt.xlabel('Radial Distance from Center')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, '4_radial_distribution.png'), dpi=300)
    plt.close()
    print("Radial distribution plot saved.")

def plot_z_distribution(df, output_dir):
    """
    Analyzes and visualizes the gene distribution across the Z-planes.
    """
    print("Generating Z-plane distribution plot...")
    plt.figure(figsize=(8, 6))
    
    ax = sns.countplot(data=df, x='z', palette='viridis', order=[0, 1, 2])
    
    plt.title('Gene Spot Counts per Z-Plane')
    plt.xlabel('Z-Plane')
    plt.ylabel('Number of Gene Spots')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')

    plt.savefig(os.path.join(output_dir, '5_z_axis_distribution.png'), dpi=300)
    plt.close()
    print("Z-plane distribution plot saved.")

def main():
    """
    Main function to load data and run all plotting functions.
    """
    input_file = 'normalized_genes_in_cells_cpu.csv'
    output_dir = 'visualization_results_en'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Loading data from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Please run the data processing script first.")
        return
    print(f"Data loaded successfully, with {len(df):,} records.")

    # --- Call plotting functions ---
    plot_global_density(df, output_dir)
    # Call the optimized fast version
    plot_specific_gene_distribution_fast(df, output_dir, n_genes=5)
    plot_single_cell_view(df, output_dir, n_cells=4)
    plot_radial_distribution(df, output_dir)
    plot_z_distribution(df, output_dir)

    print(f"\nAll visualizations are complete! Results are saved in the '{output_dir}' folder.")

if __name__ == '__main__':
    main()