import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
import config as CFG

parser = argparse.ArgumentParser()
parser.add_argument("--train_npy", type=str, required=True)
parser.add_argument("--val_npy", type=str, default=None)
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--saga_epoch", type=int, default=500)
parser.add_argument("--saga_batch_size", type=int, default=256)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load ACC concept labels ---
train_c = torch.tensor(np.load(args.train_npy), dtype=torch.float32)
if args.val_npy:
    val_c = torch.tensor(np.load(args.val_npy), dtype=torch.float32)
else:
    val_c = None

# --- Normalize like train_FL.py ---
train_c, train_mean, train_std = (train_c - train_c.mean(0)) / (train_c.std(0) + 1e-8), train_c.mean(0), train_c.std(0)
train_c = F.relu(train_c)
if val_c is not None:
    val_c = (val_c - train_mean) / (train_std + 1e-8)
    val_c = F.relu(val_c)

# --- Load labels ---
dataset = args.dataset
train_dataset = load_dataset(dataset, split="train")
train_y = torch.LongTensor(train_dataset["label"])
if val_c is not None:
    val_dataset = load_dataset(dataset, split="validation")
    val_y = torch.LongTensor(val_dataset["label"])

test_dataset = load_dataset(dataset, split="test")
test_y = torch.LongTensor(test_dataset["label"])

# --- Placeholder test features (for glm_saga) ---
test_c = torch.zeros((len(test_dataset), train_c.shape[1]))

# --- Dataset setup ---
indexed_train_ds = IndexedTensorDataset(train_c, train_y)
indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
if val_c is not None:
    val_loader = DataLoader(TensorDataset(val_c, val_y), batch_size=args.saga_batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(test_c, test_y), batch_size=args.saga_batch_size, shuffle=False)

# --- Final classifier ---
print("dim of concept features: ", train_c.shape[1])
linear = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset])
linear.weight.data.zero_()
linear.bias.data.zero_()

STEP_SIZE = 0.05
ALPHA = 0.99
metadata = {"max_reg": {"nongrouped": 0.0007}}

print("training final layer directly from ACC features...")
if val_c is not None:
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
                           val_loader=val_loader, test_loader=test_loader, do_zero=True,
                           n_classes=CFG.class_num[dataset])
else:
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
                           test_loader=test_loader, do_zero=True,
                           n_classes=CFG.class_num[dataset])

print("training done.")

if 'path' in output_proj and len(output_proj['path']) > 0:
    if 'metrics' in output_proj['path'][-1] and 'acc_test' in output_proj['path'][-1]['metrics']:
        print("Test accuracy:", output_proj['path'][-1]['metrics']['acc_test'])
    else:
        print("No test accuracy found in glm_saga output.")
else:
    print("glm_saga output missing expected structure.")
