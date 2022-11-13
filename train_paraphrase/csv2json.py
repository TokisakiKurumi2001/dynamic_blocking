import pandas as pd
from sklearn.model_selection import train_test_split
import json


def write_to_file(file_name, input, label):
    with open(file_name, "w+") as file:
        for inp, lab in zip(input, label):
            output = {"translation": {"corrupted": inp, "full": lab}}
            file.write(json.dumps(output, ensure_ascii=False))
            file.write("\n")


if __name__ == "__main__":
    df = pd.DataFrame()
    for i in range(1, 14+1):
        _df = pd.read_csv(f'raw_data/segment_{i}.csv')
        df = pd.concat([df, _df], ignore_index=True)
        # print(len(df))

    df = df.dropna()
    inputs = df["Input"].values
    labels = df["Output"].values
    inp_train, inp_test, label_train, label_test = train_test_split(
        inputs, labels, test_size=0.3, random_state=41)
    inp_test, inp_valid, label_test, label_valid = train_test_split(
        inp_test, label_test, test_size=0.5, random_state=41)
    write_to_file("data/train.json", inp_train, label_train)
    write_to_file("data/test.json", inp_test, label_test)
    write_to_file("data/valid.json", inp_valid, label_valid)
