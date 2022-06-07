#!/usr/local/bin/python3
# Erick Sebastian Lozano Roa
# Pamela Ramírez González

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("train.csv").iloc[:,[0,2]].sort_values(by="image_id")
hist = {}
c=0
for i in data.values:
  if i[0] not in hist:
    hist[i[0]] = [i[1]]
  else:
    hist[i[0]].append(i[1])
    
for k,v in hist.items():
  hist[k] = np.unique(v)
  
f_hist = {}

for i in list(hist.values()):
  for j in i:
    if j not in f_hist:
      f_hist[j] = 1
    else:
      f_hist[j] += 1

print(sum(list(f_hist.values())))
breakpoint()
