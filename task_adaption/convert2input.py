import pandas as pd
df = pd.read_csv('small_dataset.tsv', sep='\t')
with open('input.txt', 'w+') as file:
    for line in df['Sentence2'].values:
        file.write(f'{line}\n')