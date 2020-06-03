import csv
import random
import utils
with open('./train.csv', 'r') as f:
    train_csv = list(csv.reader(f))
    del(train_csv[0])
    random.shuffle(train_csv)
    total_len = len(train_csv)
    validation = []
    train = []

    for i, line in enumerate(train_csv):
        if i < total_len * 0.8:
            train.append(line)
        else:
            validation.append(line)

with open('./validation.csv','w') as f:
    validation_csv = csv.writer(f)
    for i in validation:
        validation_csv.writerow(i)


with open('./train_leftover.csv','w') as f:
    train_1by1_csv = csv.writer(f)
    for i in train:
        train_1by1_csv.writerow(i)



