import numpy as np
import config as CFG
from datasets import load_dataset
from utils import get_labels

dataset = 'SetFit/sst2'
concept_set = CFG.concept_set[dataset]
train_dataset = load_dataset(dataset, split='train')
val_dataset = load_dataset(dataset, split='validation')

train_similarity = np.load('./mpnet_acs/SetFit_sst2/concept_labels_train.npy')
val_similarity = np.load('./mpnet_acs/SetFit_sst2/concept_labels_val.npy')

for i in range(train_similarity.shape[0]):
    for j in range(len(concept_set)):
        if get_labels(j, dataset) != train_dataset['label'][i]:
            train_similarity[i][j] = 0.0
        else:
            if train_similarity[i][j] < 0.0:
                train_similarity[i][j] = 0.0

for i in range(val_similarity.shape[0]):
    for j in range(len(concept_set)):
        if get_labels(j, dataset) != val_dataset['label'][i]:
            val_similarity[i][j] = 0.0
        else:
            if val_similarity[i][j] < 0.0:
                val_similarity[i][j] = 0.0

np.save('./mpnet_acs/SetFit_sst2/concept_labels_train_acc.npy', train_similarity)
np.save('./mpnet_acs/SetFit_sst2/concept_labels_val_acc.npy', val_similarity)
print("Saved ACC-corrected concept label files.")
