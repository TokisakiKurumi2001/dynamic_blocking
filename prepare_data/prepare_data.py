from datasets import load_dataset
import re
dataset = load_dataset('mt_eng_vietnamese', 'iwslt2015-vi-en')
vi = []
for type in ['train', 'test', 'validation']:
    for el in dataset[type]['translation']:
        sent_vi = el['vi']
        sent_vi = re.sub('\n', '', sent_vi)
        words = [word for word in sent_vi.split(' ') if word != '']
        if len(words) < 5:
            continue
        vi.append(sent_vi)

with open('data_vi.txt', 'w+') as file:
    for line in vi:
        file.write(f'{line}\n')