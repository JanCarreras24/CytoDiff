import os
import pandas as pd

# Define the root directory
root_directory = "/ictstr01/home/aih/rao.umer/codes/chipai_project/extracted_feats/dinobloom-g/"

# Initialize an empty list to store data
data_list = []

# Iterate over the directories and files in the root directory
for dirpath, dirnames, files in sorted(os.walk(root_directory)):
    for file in files:
        # Extract full file path
        filepath = os.path.join(dirpath, file)
        
        # Extract slide_id and label from the file name
        # Assuming the format: "slide_id_label.h5"
        if file.endswith(".h5"):
            try:
                # Split the filename based on the last underscore
                slide_id, label_with_ext = file.rsplit("_", 1)
                label = label_with_ext.rsplit(".", 1)[0]  # Remove ".h5" from the label
            except ValueError:
                print(f"Unexpected file naming format: {file}")
                continue

            print('slide_id:', slide_id)
            print('label:', label)
            print('tensor_path:', filepath)
            
            # Append to data list
            data_list.append({
                'patient_id': slide_id,
                'label': label,
                'tensor_paths': filepath
            })
        else:
            print(f"Skipping non-HDF5 file: {file}")

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(data_list)

# Save the DataFrame to a CSV file
csv_path = "/ictstr01/home/aih/rao.umer/codes/chipai_project/extracted_feats/datasets/dinobloom-g.csv"
df.to_csv(csv_path, index=False)

print(f"Data has been written to {csv_path}")
