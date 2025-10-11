import argparse
import numpy as np
import config as CFG
from datasets import load_dataset
from utils import get_labels

# --- aguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SetFit/sst2", help="Tên dataset (vd: SetFit/sst2)")
parser.add_argument("--train_path", type=str, required=True, help="Đường dẫn đến file concept_labels_train.npy")
parser.add_argument("--val_path", type=str, required=True, help="Đường dẫn đến file concept_labels_val.npy")
parser.add_argument("--output_prefix", type=str, default=None, help="Thư mục output, mặc định cùng thư mục input")
args = parser.parse_args()

# --- Load data and config ---
dataset = args.dataset
concept_set = CFG.concept_set[dataset]
train_dataset = load_dataset(dataset, split='train')
val_dataset = load_dataset(dataset, split='validation')

train_similarity = np.load(args.train_path)
val_similarity = np.load(args.val_path)

print(f"Loaded train: {train_similarity.shape}, val: {val_similarity.shape}")

# --- Concept Correction ---
for i in range(train_similarity.shape[0]):
    for j in range(len(concept_set)):
        if get_labels(j, dataset) != train_dataset['label'][i]:
            train_similarity[i][j] = 0.0
        elif train_similarity[i][j] < 0.0:
            train_similarity[i][j] = 0.0

for i in range(val_similarity.shape[0]):
    for j in range(len(concept_set)):
        if get_labels(j, dataset) != val_dataset['label'][i]:
            val_similarity[i][j] = 0.0
        elif val_similarity[i][j] < 0.0:
            val_similarity[i][j] = 0.0

if args.output_prefix is None:
    args.output_prefix = "/".join(args.train_path.split("/")[:-1])

np.save(f"{args.output_prefix}/concept_labels_train_acc.npy", train_similarity)
np.save(f"{args.output_prefix}/concept_labels_val_acc.npy", val_similarity)

print(f"Saved ACC-corrected files to {args.output_prefix}")
