import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. Visualization Functions ---

# 1.1 Visualize correct/incorrect predictions (unchanged)
def plot_3d_accuracy(df, output_html_path):
    df['status'] = np.where(df['true_class'] == df['predicted_class'], 'Correct', 'Incorrect')
    fig = go.Figure()
    correct_df = df[df['status'] == 'Correct']
    fig.add_trace(go.Scatter3d(
        x=correct_df['x'], y=correct_df['y'], z=correct_df['z'],
        mode='markers', marker=dict(size=3, color='green', symbol='circle', opacity=0.7),
        name='Correctly Predicted', customdata=correct_df[['cell_id', 'true_class']],
        hovertemplate="<b>Cell ID:</b> %{customdata[0]}<br><b>Class (True):</b> %{customdata[1]}<extra></extra>"
    ))
    incorrect_df = df[df['status'] == 'Incorrect']
    fig.add_trace(go.Scatter3d(
        x=incorrect_df['x'], y=incorrect_df['y'], z=incorrect_df['z'],
        mode='markers', marker=dict(size=5, color='red', symbol='x'),
        name='Incorrectly Predicted', customdata=incorrect_df[['cell_id', 'true_class', 'predicted_class']],
        hovertemplate="<b>Cell ID:</b> %{customdata[0]}<br><b>True:</b> %{customdata[1]}<br><b>Predicted:</b> %{customdata[2]}<extra></extra>"
    ))
    fig.update_layout(title='3D Visualization of Prediction Accuracy', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectratio=dict(x=1, y=1, z=1)), legend_title="Prediction Status", margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html(output_html_path)
    print(f"[Output 1/3] 3D accuracy visualization saved to: {output_html_path}")

def plot_3d_by_class(df, output_html_path):
    """Creates an interactive 3D scatter plot using Plotly, colored by true class, with a Z-axis range selector slider."""
    unique_classes = sorted(df['true_class'].unique())
    num_classes = len(unique_classes)
    
    cmap = plt.get_cmap('viridis', num_classes)
    colors = cmap(np.linspace(0, 1, num_classes))
    
    class_color_map = {
        cls: f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' 
        for cls, c in zip(unique_classes, colors)
    }

    fig = go.Figure()
    
    # Set a unique uid for each trace so the slider can reference them
    for i, cls in enumerate(tqdm(unique_classes, desc="Plotting classes")):
        class_df = df[df['true_class'] == cls]
        fig.add_trace(go.Scatter3d(
            x=class_df['x'], y=class_df['y'], z=class_df['z'],
            mode='markers',
            marker=dict(size=3, color=class_color_map[cls], symbol='circle', opacity=0.8),
            name=cls, customdata=class_df[['cell_id']],
            hovertemplate="<b>Cell ID:</b> %{customdata[0]}<br><b>Class:</b> " + cls + "<extra></extra>",
            visible=True,
            uid=f"trace-{i}" # Assign a unique ID
        ))
        
    # Create two sliders to control the Z-axis range
    z_min, z_max = df['z'].min(), df['z'].max()
    
    # Create an empty figure to update the layout
    layout_fig = go.Figure()
    
    # Update the layout, adding two sliders
    layout_fig.update_layout(
        title='3D Visualization of True Cell Types Distribution (with Z-axis Range Selector)',
        sliders=[
            {
                "active": 0, "currentvalue": {"prefix": "Z-Min: "},
                "pad": {"t": 20, "b": 10}, "y": 0, "len": 0.9, "x": 0.1,
                "steps": [
                    {"label": f"{z_val:.2f}",
                     "method": "relayout",
                     "args": [{"scene.zaxis.range[0]": z_val}]}
                    for z_val in np.linspace(z_min, z_max, 20)
                ]
            },
            {
                "active": 19, "currentvalue": {"prefix": "Z-Max: "},
                "pad": {"t": 20, "b": 10}, "y": -0.1, "len": 0.9, "x": 0.1,
                "steps": [
                    {"label": f"{z_val:.2f}",
                     "method": "relayout",
                     "args": [{"scene.zaxis.range[1]": z_val}]}
                    for z_val in np.linspace(z_min, z_max, 20)
                ]
            }
        ]
    )

    # Apply the sliders from the new layout to the main figure
    fig.layout.sliders = layout_fig.layout.sliders
    
    # Update other layout properties of the main figure
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis=dict(title='Z', range=[z_min, z_max]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        legend_title="True Cell Class",
        margin=dict(l=0, r=0, b=0, t=40),
        legend={'itemsizing': 'constant'}
    )
    
    fig.write_html(output_html_path)
    print(f"[Output 2/3] 3D class distribution plot with Z-axis range slider saved to: {output_html_path}")


# 1.3 Visualize confusion matrix (unchanged)
def plot_confusion_matrix(df, output_image_path):
    class_labels = sorted(df['true_class'].unique())
    cm = confusion_matrix(df['true_class'], df['predicted_class'], labels=class_labels)
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = np.nan_to_num(cm.astype('float') / cm_sum)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_normalized, annot=True, fmt=".1%", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Normalized Confusion Matrix', fontsize=20)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout(pad=3.0)
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print(f"[Output 3/3] Confusion matrix plot saved to: {output_image_path}")

# --- 2. Main function (unchanged) ---
def main():
    parser = argparse.ArgumentParser(description="Generate various visual analysis plots from a CSV file containing prediction results.")
    parser.add_argument('--input_csv', required=True, help="CSV file containing prediction results and coordinates.")
    parser.add_argument('--output_dir', required=True, help="Directory to save all visualization results.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Step 1/2: Loading prediction results file: {args.input_csv}")
    try:
        results_df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_csv}"); return

    print(f"\nStep 2/2: Starting to generate visualizations for {len(results_df)} cells...")
    
    accuracy_html_path = output_dir / '1_3d_prediction_accuracy.html'
    by_class_html_path = output_dir / '2_3d_distribution_by_class_with_range_slider.html' # Update filename
    cm_image_path = output_dir / '3_confusion_matrix.png'
    
    plot_3d_accuracy(results_df, accuracy_html_path)
    plot_3d_by_class(results_df, by_class_html_path)
    plot_confusion_matrix(results_df, cm_image_path)
    
    print("\n--- All visualization tasks complete! ---")

if __name__ == '__main__':
    main()