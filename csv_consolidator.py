import pandas as pd
import os

def consolidate_csv(folder_path):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize an empty DataFrame for the consolidated data
    consolidated_data = pd.DataFrame()

    for file_name in csv_files:
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)

        # Read the CSV file, skipping the first row and using the second row as the header
        data = pd.read_csv(file_path, header=1)

        # Extract the filename without extension for column naming
        new_column_name = os.path.splitext(file_name)[0]

        # Rename the 'A' column to the new column name
        data.rename(columns={'A': new_column_name}, inplace=True)

        # Merge with the consolidated data
        if consolidated_data.empty:
            consolidated_data = data
        else:
            consolidated_data = pd.merge(consolidated_data, data, on='cm-1', how='outer')

    return consolidated_data

# Example usage
folder_path = r'CSVs\231208_4xCB-loading_KBrTransmission_ambient-cure' 
consolidated_data = consolidate_csv(folder_path)

# Save the consolidated data to a new CSV file
folder_name = os.path.basename(folder_path)
export_file_name = f"{folder_name}_consolidated.csv"
consolidated_data.to_csv(export_file_name, index=False)