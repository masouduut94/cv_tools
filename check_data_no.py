import pandas as pd
from pathlib import Path


train_csvs = list(Path('data/annotated/labels/train').glob('*.csv'))
test_csvs = list(Path('data/annotated/labels/test').glob('*.csv'))

count_service = 0

for item in train_csvs:
    item = item.as_posix()
    df = pd.read_csv(item)
    df = df[df.toss]
    count_service += len(df)

print("train service number: ", count_service)


count_service = 0

for item in test_csvs:
    item = item.as_posix()
    df = pd.read_csv(item)
    df = df[df.toss]
    count_service += len(df)

print("test service number: ", count_service)
