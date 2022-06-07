import numpy as np
import torch
import torchvision.models as models
import os
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(pred, target, threshold=0.5):

    pred = np.array(pred > threshold, dtype=float)

    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),

            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),

            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),

            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),

            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),

            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),

            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),

            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),

            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }
            
def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path, 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)