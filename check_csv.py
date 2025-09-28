import csv

# Define the file name to be read
file_name = '/gpfs/home/guobiaoye/MERFISH/CCF/ccf_coordinates.csv'

try:
    # Safely open the file using the 'with' statement
    with open(file_name, mode='r', newline='', encoding='utf-8') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)
        
        print(f"--- First few rows of the file '{file_name}' (using the built-in csv module) ---")

        # Iterate over each row of the reader and use enumerate to count
        for i, row in enumerate(csv_reader):
            if i < 10:
                # Each row is read as a list of strings
                print(row)
            else:
                # Stop after reading 10 lines to avoid reading the entire large file
                break

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    print("Please make sure the CSV file is in the same directory as your Python script.")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")