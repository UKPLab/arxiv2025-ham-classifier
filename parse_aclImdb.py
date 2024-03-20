# Script for parsing the aclImdb dataset
# Dataset is stored in two folders test and train
# Each folder contains two folders pos and neg
# Each of these folders contains text files with reviews
# Each review is stored in a separate file
# Script takes all reviews from all files and stores them in a train, test and dev set
# Test and dev are obtained by splitting the test set 50/50

import os
import random
import pandas as pd

def process_folder(path):
    # Given a path either train or test, returns a pandas dataframe of all reviews
    # Within these directories, reviews are stored in text files named following the
    # convention [[id]_[rating].txt] where [id] is a unique id and [rating] is
    # the star rating for that review on a 1-10 scale.
    # (label, rating, id, review) 
    # Review is stored as a string
    data = []
    subfolders = ['pos', 'neg']
    for subfolder in subfolders:
        for root, dirs, files in os.walk(os.path.join(path, subfolder)):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), 'r', encoding="utf8") as f:
                        review = f.read()
                        rating = int(file.split('_')[1].split('.')[0])
                        label = 1 if subfolder == 'pos' else 0
                        id = file.split('_')[0]
                        data.append((label, rating, id, review) )

    df = pd.DataFrame(data, columns=['label', 'rating', 'id', 'review'])
    return df

# Set the path to the dataset folder
aclimdb_root = 'data/aclImdb'

# df_train = process_folder(os.path.join(aclimdb_root, 'train'))
# df_testdev = process_folder(os.path.join(aclimdb_root, 'test'))

# # Split the testdev set into test and dev sets
# df_dev = df_testdev.sample(frac=0.5, random_state=42)
# df_test = df_testdev.drop(df_dev.index)

# # Save the dataframes as csv
# df_train.to_csv(os.path.join(aclimdb_root, 'train.csv'), index=False)
# df_test.to_csv(os.path.join(aclimdb_root, 'test.csv'), index=False)
# df_dev.to_csv(os.path.join(aclimdb_root, 'dev.csv'), index=False)

# # Create mini version of dataset in another folder
# aclimdb_mini_root = 'data/aclImdb_mini'
# os.makedirs(aclimdb_mini_root, exist_ok=True)
# df_train_mini = df_train.sample(frac=0.1, random_state=42)
# df_test_mini = df_test.sample(frac=0.1, random_state=42)
# df_dev_mini = df_dev.sample(frac=0.1, random_state=42)
# df_train_mini.to_csv(os.path.join(aclimdb_mini_root, 'train.csv'), index=False)
# df_test_mini.to_csv(os.path.join(aclimdb_mini_root, 'test.csv'), index=False)
# df_dev_mini.to_csv(os.path.join(aclimdb_mini_root, 'dev.csv'), index=False)

# TODO: REMOVE

df_train = pd.read_csv(os.path.join(aclimdb_root, 'train.csv'))
df_test = pd.read_csv(os.path.join(aclimdb_root, 'test.csv'))
df_dev = pd.read_csv(os.path.join(aclimdb_root, 'dev.csv'))

##############


# Create a short version of dataset containing
# only reviews of 10 words or less
aclimdb_short_root = 'data/aclImdb_short'
max_len = 15
os.makedirs(aclimdb_short_root, exist_ok=True)
df_train_short = df_train[df_train['review'].str.split().str.len() <= max_len]
df_test_short = df_test[df_test['review'].str.split().str.len() <= max_len]
df_dev_short = df_dev[df_dev['review'].str.split().str.len() <= max_len]
df_train_short.to_csv(os.path.join(aclimdb_short_root, 'train.csv'), index=False)
df_test_short.to_csv(os.path.join(aclimdb_short_root, 'test.csv'), index=False)
df_dev_short.to_csv(os.path.join(aclimdb_short_root, 'dev.csv'), index=False)

print('Done')

