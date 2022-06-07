import pandas as pd
import numpy as np
from tqdm import tqdm
import json

data = pd.read_csv("train.csv")
data = data.sort_values(by=['image_id', "rad_id"], ascending=True)

all_names = pd.read_csv('images_names.txt', sep=" ").sample(frac=1, random_state=666).reset_index(drop=True)
train_names = all_names.iloc[:int(len(all_names) * 0.8), :]['names']
test_names = all_names.iloc[int(len(all_names) * 0.8):, :]['names']
breakpoint()

#train_names.to_csv("train_names.txt", sep=' ')
#test_names.to_csv("test_names.txt", sep=' ')


"""
samples_tr = []
labels = []
dicto_tr = {"samples": samples_tr, "labels": labels}

samples_te = []
dicto_te = {"samples": samples_te, "labels": labels}

hist = {}

for i in tqdm(range(len(data.values))):
    if data.values[i][0] not in hist:
        hist[data.values[i][0]] = 0
        c = 1
        image_labels = [data.values[i][1]]
        while data.values[i + c][3] == data.values[i][3]:
            image_labels.append(data.values[i + c][1])
            c += 1
        if data.values[i][0] in train_names:
            samples_tr.append({"image_name": data.values[i][0] + ".png", "image_labels": image_labels})
        else:
            samples_te.append({"image_name": data.values[i][0] + ".png", "image_labels": image_labels})

    else:
        hist[data.values[i][0]] += 1

    if data.values[i][1] not in labels:
        labels.append(data.values[i][1])

with open('train.json', 'w') as outfile:
    json.dump(json.dumps(dicto_tr), outfile)

with open('test.json', 'w') as outfile:
    json.dump(json.dumps(dicto_te), outfile)

# print(np.mean(list(hist.values()))) # valor promedio de cuantos radiologos anotaron las imagenes
"""