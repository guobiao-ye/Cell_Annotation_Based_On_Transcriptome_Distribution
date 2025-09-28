# Download the cell boundary file (for example, mouse 1 / ABAC-2)
mkdir boundaries
cd ./boundaries
wget -r -np -nH --cut-dirs=5 -R "index.html*" https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/processed_data/cell_boundaries_updated/mouse1_coronal/
python check_csv.py
cd ..

# Download the decoded spots file (for example, mouse 1 / ABAC-2)
mkdir decoded_spots
cd ./decoded_spots
wget -r -np -nH --cut-dirs=5 -R "index.html*" https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/processed_data/decoded_spots/mouse1_coronal/
python check_spot_column.py
python check_csv.py
cd ..



# Use a multi-core CPU to simultaneously align cell boundaries and genes, and normalize (output to MERFISH/normalized_results/mouse1_coronal_parallel). Run the batch processing script (note to activate the conda environment for merfish_env)
chmod +x run_batch_registration.sh
./run_batch_registration.sh

# Global visualization of alignment results
python visualize_gene_distribution_global.py

# Cellular and gene-specific visualization of alignment results
python visualize_gene_distribution_cell_specific.py



# Get ground_truth
# First, follow the tutorial at https://alleninstitute.github.io/abc_atlas_access/notebooks/zhuang_merfish_tutorial.html to load the data
python gound_truth_saving.py

# Count the number of subcategories
python count_ground_truth_subclasses.py \
  --metadata_csv ./Zhuang-ABCA-2_cell_metadata_full.csv

# Merge the metadata (annotation information) with the gene files, and sample 5000 cells for each category
python sample_genes_by_class.py \
  --metadata ./Zhuang-ABCA-2_cell_metadata_full.csv \
  --input_dir ./normalized_results/mouse1_coronal_parallel/ \
  --output_file ./sample/final_sampled_genes_with_class.csv

python sample_genes_by_subclass.py \
  --metadata ./Zhuang-ABCA-2_cell_metadata_full.csv \
  --input_dir ./normalized_results/mouse1_coronal_parallel/ \
  --output_file ./sample/final_sampled_genes_with_subclass.csv

# Divide 600,000 cells into the training set, and the rest are used as the test set.
python create_train_test_split.py \
  --input_file ./sample/final_sampled_genes_with_class.csv \
  --output_dir ./train_test_split_data/ \
  --train_size 600000

python create_train_test_split_sub.py \
  --input_file ./sample/final_sampled_genes_with_subclass.csv \
  --output_dir ./train_test_split_data_sub/ \
  --train_size 600000

python create_train_test_split_sub.py \
  --input_file ./sample_50cells/final_sampled_genes_with_subclass.csv \
  --output_dir ./train_test_split_data_sub_60000train/ \
  --train_size 60000

# From the small sample, 600 cells were selected as the training set, while the rest were used as the test set.
python create_train_test_split_v2.py \
  --input_file ./sample_50cells/final_sampled_genes_with_class.csv \
  --output_dir ./train_test_split_data_600train/ \
  --train_size 600

# Visualize the distribution of each cell's transcript (optional, with high computational requirements)
python cell_genes_image.py \
  --input_file ./train_test_split_data/train_set_genes.csv \
  --output_dir ./cell_image_for_train/

python cell_genes_image.py \
  --input_file ./train_test_split_data/test_set_genes.csv \
  --output_dir ./cell_image_for_test/ \
  --color_map_file ./cell_image_for_train/gene_color_map.csv



# Check for any data breaches
python check_data_leakage.py \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv



# Start training
# Environment: pytorch (cuda=11.8) and tqdm
# Python version: 3.10
# Environment name: dl
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Only consider x and y, use 2D point cloud modeling
# sbatch run_merfish_training.sh
python train_classifier_2D.py \
  --train_csv ./train_test_split_data/train_set_genes.csv \
  --test_csv ./train_test_split_data/test_set_genes.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --epochs 25 \
  --batch_size 32 \
  --output_dir ./merfish_model_results/

# Consider x, y, z. Use 3D point cloud modeling
# sbatch run_merfish_training_3D.sh
python train_classifier_3D.py \
  --train_csv ./train_test_split_data/train_set_genes.csv \
  --test_csv ./train_test_split_data/test_set_genes.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_dir ./merfish_model_results_3d/ \
  --epochs 25 \
  --batch_size 32 \
  --lr 0.001

python train_classifier_3D.py \
  --train_csv ./train_test_split_data_600train/train_set_genes.csv \
  --test_csv ./train_test_split_data_600train/test_set_genes.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_dir ./merfish_model_results_3d_600train/ \
  --epochs 25 \
  --batch_size 32 \
  --lr 0.001

python train_classifier_3D.py \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_dir ./merfish_model_results_3d_50cells_2000tests/ \
  --epochs 25 \
  --batch_size 32 \
  --lr 0.001

# Abolition Experiment (Using Only Genetic Labels)
python train_classifier_NO_SPATIAL.py \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_dir ./merfish_model_results_3d_50cells_2000tests_NO_SPATIAL/ \
  --epochs 25 \
  --batch_size 32 \
  --lr 0.001

# Ablation Experiment (Using Only Genetic Coordinates)
python train_classifier_NO_GENES.py \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
  --output_dir ./merfish_model_results_3d_50cells_2000tests_NO_GENES/ \
  --epochs 25 \
  --batch_size 32 \
  --lr 0.001

# New architecture + No use of attention, rapid convergence
python train_classifier_3D_hybrid.py \
    --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
    --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
    --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
    --output_dir ./v3/ \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001

# New architecture + Enable attention enhancement to achieve better performance
python train_classifier_3D_hybrid.py \
    --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
    --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
    --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
    --output_dir ./v3_attention/ \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --use_attention  # Enable attention

# The new framework's ablation experiments
python ablation_study.py \
    --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
    --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
    --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
    --output_dir ./merfish_ablation_results_3d_50cells_2000tests/ \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.001 \
    --hidden_dim 128 \
    --gene_embed_dim 32 \
    --max_points 400 \
    --dropout_rate 0.2 \
    --spatial_normalize

python train_classifier_3D_sub.py \
  --train_csv ./train_test_split_data_sub/train_set_genes.csv \
  --test_csv ./train_test_split_data_sub/test_set_genes.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_dir ./merfish_model_results_3d_sub/ \
  --epochs 25 \
  --batch_size 32 \
  --lr 0.001

python train_classifier_3D_sub.py \
  --train_csv ./train_test_split_data_sub_60000train/train_set_genes.csv \
  --test_csv ./train_test_split_data_sub_60000train/test_set_genes.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_dir ./merfish_model_results_3d_sub_60000train/ \
  --epochs 25 \
  --batch_size 32 \
  --lr 0.001


# Check the various indicators of the annotation model before training (small sample training, small sample prediction)
python train_classifier_3D_with_0epoch.py \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_dir ./test_1epoch/ \
  --epochs 1 \
  --batch_size 32 \
  --lr 0.001

# Load the training weights (3D version), and output the prediction results (large sample training, large sample prediction)
python visualize_predictions.py \
  --model_path ./merfish_model_results_3d/model_epoch_25.pth \
  --test_csv ./train_test_split_data/test_set_genes.csv \
  --train_csv ./train_test_split_data/train_set_genes.csv \
  --metadata_csv ./Zhuang-ABCA-2_cell_metadata_full.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_csv ./prediction_analysis_results/prediction_results_with_coords.csv

# Load the training weights (3D version), and output the prediction results (large sample training, small sample prediction)
python visualize_predictions.py \
  --model_path ./merfish_model_results_3d/model_epoch_25.pth \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --metadata_csv ./Zhuang-ABCA-2_cell_metadata_full.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_csv ./prediction_analysis_results_50cells_2000tests/prediction_results_with_coords.csv

# Load the training weights (3D version), and output the prediction results (small sample training, small sample prediction)
python visualize_predictions.py \
  --model_path ./merfish_model_results_3d_50cells_2000tests/model_epoch_15.pth \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --metadata_csv ./Zhuang-ABCA-2_cell_metadata_full.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_csv ./prediction_analysis_results_small_train_50cells_2000tests/prediction_results_with_coords.csv

# Visualized prediction results (large training and large testing)
python generate_visuals_from_results.py \
  --input_csv ./prediction_analysis_results/prediction_results_with_coords.csv \
  --output_dir ./final_visualizations/

# Process the prediction result files to interface with the benchmark script (small training and small testing, 15 epochs)
python format_custom_predictions.py \
  --input_csv ./prediction_analysis_results_small_train_50cells_2000tests/prediction_results_with_coords.csv \
  --model_name SPHAEC_3D \
  --output_file ./benchmark_results/SPHAEC_3D/SPHAEC_3D_predictions.csv




# ----------benchmark-------------
conda activate benchmark

# Generate the count matrix
python create_count_matrix_and_labels.py \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
  --metadata_csv ./Zhuang-ABCA-2_cell_metadata_full.csv \
  --output_dir ./data_for_benchmark_2000tests/

python create_count_matrix_and_labels.py \
  --train_csv ./train_test_split_data_600train/train_set_genes.csv \
  --test_csv ./train_test_split_data_600train/test_set_genes.csv \
  --metadata_csv ./Zhuang-ABCA-2_cell_metadata_full.csv \
  --output_dir ./data_for_benchmark_600train/

# Benchmarking - Scanvi
python benchmark_scanvi.py \
  --train_matrix ./benchmark_data/train_set_count_matrix.csv \
  --train_labels ./benchmark_data/train_set_labels.csv \
  --test_matrix ./benchmark_data/test_set_count_matrix.csv \
  --test_labels ./benchmark_data/test_set_labels.csv \
  --output_dir ./benchmark_results/scanvi/

python benchmark_scanvi.py \
  --train_matrix ./data_for_benchmark_600train/train_set_count_matrix.csv \
  --train_labels ./data_for_benchmark_600train/train_set_labels.csv \
  --test_matrix ./data_for_benchmark_600train/test_set_count_matrix.csv \
  --test_labels ./data_for_benchmark_600train/test_set_labels.csv \
  --output_dir ./benchmark_results_600train/scanvi/

# Benchmark - KNN
python benchmark_knn.py \
  --train_matrix ./benchmark_data/train_set_count_matrix.csv \
  --train_labels ./benchmark_data/train_set_labels.csv \
  --test_matrix ./benchmark_data/test_set_count_matrix.csv \
  --test_labels ./benchmark_data/test_set_labels.csv \
  --output_dir ./benchmark_results/knn/ \
  --n_neighbors 15

python benchmark_knn.py \
  --train_matrix ./data_for_benchmark_600train/train_set_count_matrix.csv \
  --train_labels ./data_for_benchmark_600train/train_set_labels.csv \
  --test_matrix ./data_for_benchmark_600train/test_set_count_matrix.csv \
  --test_labels ./data_for_benchmark_600train/test_set_labels.csv \
  --output_dir ./benchmark_results_600train/knn/ \
  --n_neighbors 15



# For each cell in the test_set_genes.csv file, randomly delete 30% of the transcripts, generating a corrupted test file to verify the performance of the cell type prediction algorithm in dealing with dropout.
python create_dropout_test_set.py \
  --input_csv ./train_test_split_data_50cells_2000tests/test_set_genes.csv \
  --output_csv ./train_test_split_data_50cells_2000tests/test_set_genes_dropout_30pct.csv \
  --keep_fraction 0.7

# Load the training weights (3D version), and output the prediction results under dropout (for small sample training and small sample prediction)
python visualize_predictions.py \
  --model_path ./merfish_model_results_3d_50cells_2000tests/model_epoch_15.pth \
  --test_csv ./train_test_split_data_50cells_2000tests/test_set_genes_dropout_30pct.csv \
  --train_csv ./train_test_split_data_50cells_2000tests/train_set_genes.csv \
  --metadata_csv ./Zhuang-ABCA-2_cell_metadata_full.csv \
  --color_map ./cell_image_for_train_50cells_2000tests/gene_color_map.csv \
  --output_csv ./prediction_analysis_results_small_train_50cells_2000tests_dropout_30pct/prediction_results_with_coords.csv
