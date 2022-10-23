from eda import eda
import re
import pandas as pd

augment_sents = []
original_sents = []

with open('data_vi.txt', 'r') as file:
    for line in file:
        line = re.sub("\n", "", line)
        augment_sent = eda(line, alpha_rs=0.3, p_rd=0.3)
        augment_sents.append(augment_sent)
        original_sents.append(line)

df = pd.DataFrame({'input': augment_sents, 'label': original_sents})
df.to_csv('data.csv', index=False)