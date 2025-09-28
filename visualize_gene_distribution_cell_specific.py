import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.colors as colors

# --- Helper Functions ---

def load_and_merge_data(genes_file, metadata_file):
    """Loads gene and metadata, then merges them."""
    print(f"Loading normalized gene data from '{genes_file}'...")
    try:
        df_genes = pd.read_csv(genes_file)
    except FileNotFoundError:
        print(f"Error: Gene file '{genes_file}' not found.")
        return None
    
    print(f"Loading cell metadata from '{metadata_file}'...")
    try:
        df_meta = pd.read_csv(metadata_file)
    except FileNotFoundError:
        print(f"Error: Metadata file '{metadata_file}' not found.")
        return None

    # Ensure ID columns are the same data type for merging
    df_genes['cell_id'] = df_genes['cell_id'].astype(str)
    df_meta['cell_label'] = df_meta['cell_label'].astype(str)
    
    print("Merging gene data with cell metadata...")
    df_meta.rename(columns={'cell_label': 'cell_id'}, inplace=True)
    
    df_merged = pd.merge(
        df_genes, 
        df_meta[['cell_id', 'subclass', 'supertype', 'class']], 
        on='cell_id', 
        how='left'
    )
    
    total_genes = len(df_genes)
    pre_merge_len = len(df_merged)
    df_merged.dropna(subset=['subclass'], inplace=True)
    post_merge_len = len(df_merged)
    unmatched_count = pre_merge_len - post_merge_len
    
    if unmatched_count > 0:
        print(f"\nWARNING: {unmatched_count:,} out of {total_genes:,} gene spots ({unmatched_count/total_genes:.2%}) could not be matched to cell metadata.")
        print("This may indicate a mismatch between the gene data and the metadata file.\n")
    
    print(f"Data merged successfully. Final dataset has {len(df_merged):,} records.")
    return df_merged

# --- Visualization Functions ---

def plot_subclass_global_density(df, output_dir, n_subclasses=4):
    """Plots global gene density for the most abundant cell subclasses."""
    print(f"Generating global density plots for top {n_subclasses} cell subclasses...")
    top_subclasses = df['subclass'].value_counts().nlargest(n_subclasses).index.tolist()

    fig, axes = plt.subplots(1, n_subclasses, figsize=(n_subclasses * 5, 5.5))
    if n_subclasses == 1: axes = [axes]
    fig.suptitle('Global Normalized Gene Density by Cell Subclass', fontsize=16)

    collection = None
    for i, subclass in enumerate(top_subclasses):
        subclass_df = df[df['subclass'] == subclass]
        ax = axes[i]
        
        # Use hexbin and capture the returned collection for the colorbar
        collection = ax.hexbin(
            subclass_df['normalized_x'], subclass_df['normalized_y'],
            gridsize=80, cmap='inferno', bins='log'
        )
        
        ax.set_title(f'Subclass: {subclass}\n(n={len(subclass_df):,} spots)')
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y' if i == 0 else '')
        ax.axhline(0, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(0, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_aspect('equal', adjustable='box')
    
    if collection:
        fig.colorbar(collection, ax=axes.ravel().tolist(), shrink=0.8, label="Gene Spot Count (log scale)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, '1_subclass_global_density.png'), dpi=300)
    plt.close()
    print("Subclass global density plots saved.")

def plot_gene_in_different_subclasses_normalized(df, output_dir, gene_name):
    """
    Compares the PROBABILITY DENSITY of a single gene's distribution 
    across different cell subclasses, with improved layout and background.
    """
    print(f"Comparing NORMALIZED DENSITY of gene '{gene_name}' across cell subclasses...")
    
    gene_df = df[df['target_gene'] == gene_name].dropna(subset=['normalized_x', 'normalized_y'])
    if gene_df.empty:
        print(f"Warning: Gene '{gene_name}' not found. Skipping plot.")
        return
        
    top_subclasses = gene_df['subclass'].value_counts().nlargest(4).index.tolist()
    if len(top_subclasses) < 1:
        print(f"Warning: Gene '{gene_name}' not in any subclasses for comparison. Skipping.")
        return

    # 确定所有子图的共同坐标范围
    subset_df = gene_df[gene_df['subclass'].isin(top_subclasses)]
    x_min, x_max = subset_df['normalized_x'].min(), subset_df['normalized_x'].max()
    y_min, y_max = subset_df['normalized_y'].min(), subset_df['normalized_y'].max()
    
    max_density = 0
    histograms = []
    
    # 第一遍循环：计算所有直方图并找到最大密度值
    for subclass in top_subclasses:
        subclass_gene_df = subset_df[subset_df['subclass'] == subclass]
        if subclass_gene_df.empty:
            histograms.append(None)
            continue
        
        counts, xedges, yedges = np.histogram2d(
            subclass_gene_df['normalized_x'], subclass_gene_df['normalized_y'],
            bins=50, range=[[x_min, x_max], [y_min, y_max]]
        )
        
        bin_area = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
        # 避免除以零
        if counts.sum() > 0:
            density = counts / (counts.sum() * bin_area)
            if density.max() > max_density:
                max_density = density.max()
        else:
            density = counts # 保持为0
            
        histograms.append((density, xedges, yedges))

    # 第二遍循环：绘图
    fig, axes = plt.subplots(1, len(top_subclasses), figsize=(len(top_subclasses) * 5, 5))
    if len(top_subclasses) == 1: axes = [axes]
    fig.suptitle(f"Normalized Probability Density of Gene '{gene_name}' by Subclass", fontsize=16)
    
    mappable = None
    for i, subclass in enumerate(top_subclasses):
        ax = axes[i]
        
        # --- FIX 1: Set background color to white ---
        ax.set_facecolor('white')
        
        hist_data = histograms[i]
        
        if hist_data is None or np.all(hist_data[0]==0):
            ax.set_title(f'Subclass: {subclass}\n(No data)')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max) # 保持坐标轴一致
            ax.set_aspect('equal', adjustable='box')
            continue

        density, xedges, yedges = hist_data
        
        # 使用 pcolormesh 绘制归一化后的密度
        # 使用 vmin=1e-9 (一个非常小的数) 避免0值被渲染
        pcm = ax.pcolormesh(
            xedges, yedges, density.T, 
            cmap='viridis', 
            vmax=max_density if max_density > 0 else 1, # 避免vmax为0
            shading='auto'
        )
        mappable = pcm # 保存最后一个有效的mappable对象用于颜色条

        ax.set_title(f'Subclass: {subclass}\n(n={df[(df["subclass"]==subclass) & (df["target_gene"]==gene_name)].shape[0]:,})')
        ax.set_xlabel('Normalized X')
        if i == 0:
            ax.set_ylabel('Normalized Y')
        ax.set_aspect('equal', adjustable='box')
        # 仍然设置统一的坐标范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # --- FIX 2: Manually adjust layout and add colorbar to the right ---
    # 调整子图布局，为右侧的颜色条留出空间
    fig.subplots_adjust(right=0.85) 
    
    if mappable:
        # 在图的右侧创建一个新的轴用于颜色条
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) # [left, bottom, width, height]
        fig.colorbar(mappable, cax=cbar_ax, label="Probability Density")
    
    # tight_layout 不再需要，因为我们手动调整了
    # plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    plt.savefig(os.path.join(output_dir, f'2_gene_{gene_name}_by_subclass_normalized.png'), dpi=300)
    plt.close()
    print(f"Normalized plot for gene '{gene_name}' by subclass saved.")

def plot_gene_by_z_plane_in_subclass(df, output_dir, gene_name, subclass_name):
    """Visualizes a gene's distribution across Z-planes within a specific cell subclass."""
    print(f"Visualizing gene '{gene_name}' in subclass '{subclass_name}' by Z-plane...")
    
    target_df = df[(df['target_gene'] == gene_name) & (df['subclass'] == subclass_name)]
    if target_df.empty:
        print(f"Warning: No data for gene '{gene_name}' in subclass '{subclass_name}'. Skipping.")
        return

    g = sns.FacetGrid(target_df, col="z", col_wrap=3, height=5, aspect=1)
    g.map_dataframe(sns.kdeplot, x="normalized_x", y="normalized_y", fill=True, cmap="magma")
    g.fig.suptitle(f"Distribution of '{gene_name}' in '{subclass_name}' across Z-Planes", y=1.03)
    g.set_axis_labels("Normalized X", "Normalized Y")
    g.set_titles("Z-Plane: {col_name}")
    
    plt.savefig(os.path.join(output_dir, f'3_gene_{gene_name}_in_{subclass_name.replace(" ", "_")}_by_z.png'), dpi=300)
    plt.close()
    print(f"Z-plane plot for '{gene_name}' in '{subclass_name}' saved.")

def plot_radial_distribution_by_subclass(df, output_dir, n_subclasses=4):
    """Compares the radial distribution of genes for different cell subclasses."""
    print(f"Comparing radial distributions for top {n_subclasses} subclasses...")
    if 'radial_dist' not in df.columns:
        df['radial_dist'] = np.sqrt(df['normalized_x']**2 + df['normalized_y']**2)
    
    top_subclasses = df['subclass'].value_counts().nlargest(n_subclasses).index.tolist()

    plt.figure(figsize=(12, 7))
    for subclass in top_subclasses:
        subclass_df = df[df['subclass'] == subclass]
        sample_size = min(100000, len(subclass_df))
        sns.kdeplot(subclass_df['radial_dist'].sample(sample_size, random_state=1), 
                    label=f'{subclass} (n={len(subclass_df):,})', 
                    linewidth=2)

    plt.title('Radial Distribution Density by Cell Subclass')
    plt.xlabel('Radial Distance from Cell Center')
    plt.ylabel('Density')
    plt.legend(title='Cell Subclass')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, '4_radial_distribution_by_subclass.png'), dpi=300)
    plt.close()
    print("Radial distribution comparison plot saved.")

def main():
    """Main function to orchestrate the cell type-specific analysis."""
    genes_file = 'normalized_genes_in_cells_cpu.csv'
    metadata_file = 'Zhuang-ABCA-2_cell_metadata_full.csv'
    output_dir = 'visualization_celltype_specific_ABCA2'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    df_merged = load_and_merge_data(genes_file, metadata_file)
    if df_merged is None or df_merged.empty:
        print("No data after merging. Exiting.")
        return

    # --- Dynamically select genes and subclasses for analysis ---
    print("\n--- Data Overview after Merging ---")
    top_genes = df_merged['target_gene'].value_counts()
    print("Top 10 most abundant genes:")
    print(top_genes.head(10))
    
    top_subclasses = df_merged['subclass'].value_counts()
    print("\nTop 10 most abundant cell subclasses:")
    print(top_subclasses.head(10))

    # Automatically select the most abundant items for analysis
    # We will pick the top 2 genes and top 1 subclass for detailed plots
    # We skip the very first gene as it is often a non-coding RNA with a very different pattern
    if len(top_genes) > 1:
        GENE_OF_INTEREST_1 = top_genes.index[1]
    elif not top_genes.empty:
        GENE_OF_INTEREST_1 = top_genes.index[0]
    else:
        GENE_OF_INTEREST_1 = None

    if len(top_genes) > 2:
        GENE_OF_INTEREST_2 = top_genes.index[2]
    else:
        GENE_OF_INTEREST_2 = None

    SUBCLASS_OF_INTEREST = top_subclasses.index[0] if not top_subclasses.empty else None
    
    print(f"\n--- Automatically selected for detailed analysis ---")
    print(f"Gene 1 for analysis: {GENE_OF_INTEREST_1}")
    print(f"Gene 2 for analysis: {GENE_OF_INTEREST_2}")
    print(f"Subclass for Z-plane analysis: {SUBCLASS_OF_INTEREST}")
    print("--------------------------------------------------\n")

    # --- Run Visualizations ---
    plot_subclass_global_density(df_merged, output_dir, n_subclasses=4)
    if GENE_OF_INTEREST_1:
        plot_gene_in_different_subclasses_normalized(df_merged, output_dir, gene_name=GENE_OF_INTEREST_1)
    if GENE_OF_INTEREST_2:
        plot_gene_in_different_subclasses_normalized(df_merged, output_dir, gene_name=GENE_OF_INTEREST_2)
    if SUBCLASS_OF_INTEREST and GENE_OF_INTEREST_1:
        plot_gene_by_z_plane_in_subclass(df_merged, output_dir, gene_name=GENE_OF_INTEREST_1, subclass_name=SUBCLASS_OF_INTEREST)
    plot_radial_distribution_by_subclass(df_merged, output_dir, n_subclasses=4)
    
    print(f"\nAll cell type-specific visualizations for {metadata_file} are complete!")
    print(f"Results are saved in the '{output_dir}' folder.")

if __name__ == '__main__':
    main()