import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_recall_curve


def create_ris_dict():
    a = ord('а')
    chars = [chr(i) for i in range(a, a + 32)]
    nums = [i + 1 for i in range(0, 33)]
    rus_dict = dict(zip(chars, nums))
    return rus_dict


def encode(word, max_len):
    cur_word = []
    rus_dict = create_ris_dict()
    for i in word:
        cur_word.append(rus_dict[i])
    while len(cur_word) < max_len:
        cur_word.append(1)
    return cur_word


def f_score(preds, labels):
    sigmoid = nn.Sigmoid()
    assert all(map(lambda preds: preds.size(1) == 2, preds))
    def to_probs(preds): sigmoid(preds)[:, 0].cpu().detach().numpy()
    def to_labels(labels): (1 - labels).cpu().detach().numpy()

    preds = list(map(to_probs, preds))
    labels = list(map(to_labels, labels))

    preds = [item for subl in preds for item in subl]
    labels = [item for subl in labels for item in subl]
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    return fscore[ix]
