# k-fold cross validation (k=5) pattern:
# Fold 0: [-- | -- | -- | ** | ##]
# Fold 1: [-- | -- | ** | ## | --]
# Fold 2: [-- | ** | ## | -- | --]
# Fold 3: [** | ## | -- | -- | --]
# Fold 4: [## | -- | -- | -- | **]

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

def generate_kfold_splits(
    data_path='',
    num_splits=5,
    random_state=42,
    filename='filename.csv',
    output_filename='kfold_splits.csv',
    counts_filename='kfold_counts.csv'
):
    # 1. Load your dataset
    data = pd.read_csv(os.path.join(data_path, filename))

    # 2. Ensure the label column is present
    if 'label' not in data.columns:
        raise ValueError("Label column ('label') missing from the dataset.")

    # 3. Create directory for saving (if needed)
    save_path = data_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 4. Assign a temporary fold_id to each sample using StratifiedKFold (as "base" folds)
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    
    # Initialize a column in 'data' to store fold labels (0 to 4).
    data['fold_id'] = -1

    for fold_number, (train_index, test_index) in enumerate(skf.split(data, data['label'])):
        data.loc[test_index, 'fold_id'] = fold_number

    # 5. For each of the k folds, create a new column marking each row as train/val/test
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        data[fold_col] = 'train'  # default

        # According to the specified pattern:
        # test_fold = (i + 4) % 5  # one "step" behind i
        # val_fold  = (i + 3) % 5  # two "steps" behind i
        test_fold = (i + num_splits - 1) % num_splits
        val_fold  = (i + num_splits - 2) % num_splits

        data.loc[data['fold_id'] == val_fold,  fold_col] = 'val'
        data.loc[data['fold_id'] == test_fold, fold_col] = 'test'

    # 6. Create and save the counts for each fold
    fold_results = []
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        train_count = (data[fold_col] == 'train').sum()
        val_count   = (data[fold_col] == 'val').sum()
        test_count  = (data[fold_col] == 'test').sum()

        fold_results.append({
            'fold': i,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count
        })

    # Convert the list of dicts to a DataFrame
    counts_df = pd.DataFrame(fold_results)
    counts_df.to_csv(os.path.join(save_path, counts_filename), index=False)

    # 7. Remove the 'fold_id' column if not needed
    data.drop(columns=['fold_id'], inplace=True)

    # 8. Save the combined data with new split columns
    data.to_csv(os.path.join(save_path, output_filename), index=False)

    print(f"\n{num_splits}-fold pattern splits (Train/Val/Test) saved to '{output_filename}'")
    print(f"K-Fold counts saved to '{counts_filename}'")

# Example usage:

data_path = '/home/aih/jan.boada/project/codes/classification/csv_files/mix/5000'
generate_kfold_splits(data_path=data_path, 
                      num_splits=5, 
                      random_state=42,
                      filename='matek_metadata_base.csv',
                      output_filename='matek_metadata.csv',
                      counts_filename='matek_metadata_kfold_counts.csv')
