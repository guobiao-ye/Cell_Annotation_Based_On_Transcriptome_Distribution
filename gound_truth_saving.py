# 1. First, get the DataFrame you want to save from the dictionary
# The value of datasets[0] is 'Zhuang-ABCA-1'
df_to_save = cell_extended[datasets[0]]

# 2. Define the desired filename
output_filename = 'Zhuang-ABCA-1_cell_metadata_full.csv'

# 3. Call the .to_csv() method to write the DataFrame to a file
# encoding='utf-8' is used to prevent garbled text issues with Chinese characters or special symbols
df_to_save.to_csv(output_filename, encoding='utf-8')

print(f"Data has been successfully saved to the file: {output_filename}")