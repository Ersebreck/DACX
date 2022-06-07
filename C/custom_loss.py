import json
import pandas as pd
import numpy as np
import torch

anno_path = '/home/dajimenez166/DACX/train.json'

with open(anno_path) as fp:
    json_data = json.loads(json.load(fp))

samples = json_data['samples']
classes = json_data['labels']

def data_processing(obj):
    dic={}
    dic['image']=obj['image_name']
    image_labels = obj['image_labels']
    for label in classes:
        dic[label]= 1 if label in image_labels else 0
    return dic

data = list(map(data_processing,samples))

df = pd.DataFrame.from_dict(data)

N = len(df)
labels = df.keys()[1:]

class_weights = {}
positive_weights = {}
negative_weights = {}

for label in sorted(labels):
    positive_weights[label] = N /(2 * sum(df[label] == 1))
    negative_weights[label] = N /(2 * sum(df[label] == 0))
    
class_weights['positive_weights'] = positive_weights
class_weights['negative_weights'] = negative_weights

Wp = class_weights['positive_weights']
Wn = class_weights['negative_weights']

y_tp=[[0.6806, 0.7255, 0.5066, 0.4294, 0.6175, 0.4111, 0.5674, 0.4463, 0.6277,
         0.4600, 0.4759, 0.4562, 0.5041, 0.5796, 0.4787],
        [0.5995, 0.5478, 0.5441, 0.4409, 0.5713, 0.4673, 0.5614, 0.4888, 0.6141,
         0.4784, 0.4905, 0.4643, 0.5033, 0.5589, 0.4916],
        [0.6110, 0.6222, 0.5489, 0.4422, 0.5263, 0.4334, 0.5316, 0.4214, 0.6288,
         0.4808, 0.4935, 0.5183, 0.5200, 0.5909, 0.5856],
        [0.5181, 0.6649, 0.5025, 0.4561, 0.5652, 0.4272, 0.4699, 0.3266, 0.5552,
         0.4600, 0.4164, 0.6116, 0.4746, 0.5709, 0.5491]]

y_pred=[[0, 0, 0, 0, 0, 0, 0, 0, 1,0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1,1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

def custom_loss(y_logit=y_pred, y_true=y_tp):
    '''
    Multi-label cross-entropy
    * Required "Wp", "Wn" as positive & negative class-weights
    y_true: true value
    y_logit: predicted value
    '''
    # y_logit= y_logit.cpu().detach().numpy()
    # y_true= y_true.cpu().detach().numpy()

    individual_losses=[]
    loss = float(0)

    for i in range(len(y_logit)):
        individual_loss=float(0)        
        for j, key in enumerate(Wp.keys()):
            first_term = Wp[key] * y_true[i][j] * np.log(y_logit[i][j] + np.finfo(np.float32).eps)
            second_term = Wn[key] * (1 - y_true[i][j]) * np.log(1 - y_logit[i][j] + np.finfo(np.float32).eps)
            individual_loss -= (first_term + second_term)
        individual_losses.append(individual_loss)
   
    loss = np.mean(individual_losses)
    return loss

# y_true=[[0.6806, 0.7255, 0.5066, 0.4294, 0.6175, 0.4111, 0.5674, 0.4463, 0.6277,
#          0.4600, 0.4759, 0.4562, 0.5041, 0.5796, 0.4787],
#         [0.5995, 0.5478, 0.5441, 0.4409, 0.5713, 0.4673, 0.5614, 0.4888, 0.6141,
#          0.4784, 0.4905, 0.4643, 0.5033, 0.5589, 0.4916],
#         [0.6110, 0.6222, 0.5489, 0.4422, 0.5263, 0.4334, 0.5316, 0.4214, 0.6288,
#          0.4808, 0.4935, 0.5183, 0.5200, 0.5909, 0.5856],
#         [0.5181, 0.6649, 0.5025, 0.4561, 0.5652, 0.4272, 0.4699, 0.3266, 0.5552,
#          0.4600, 0.4164, 0.6116, 0.4746, 0.5709, 0.5491]]

# y_true = torch.tensor(y_true)

# y_pred=[[0, 0, 0, 0, 0, 0, 0, 0, 1,0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 1, 1,1, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

# y_pred = torch.tensor(y_pred)

# print(custom_loss(y_true,y_pred))