import glob
import pandas as pd

training_set_dir = './raw_dataset/training_set'
training_set_file_names = [file for file in glob.glob(f'{training_set_dir}/*.txt')]

raw_data = []

for filename in training_set_file_names:
    row = []
    with open(filename) as f:
        row.append(f.read())
    party_indicator = filename.split('_')[-1][0]
    row.append(party_indicator)
    # 0 -> dem
    # 1 -> repub
    party_num_label = 0 if party_indicator == 'D' else 1
    row.append(party_num_label)

    raw_data.append(row)

df = pd.DataFrame(raw_data)
df.rename(columns={0: 'text', 1: 'party', 2: 'party_num_label'}, inplace=True)

print(df.head())

df.to_csv('./raw_dataset/training_set.csv', index=False)
