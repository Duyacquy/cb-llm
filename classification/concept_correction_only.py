import argparse
import os
import numpy as np
import config as CFG
from datasets import load_dataset
from utils import get_labels
from tqdm import tqdm

# --- arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SetFit/sst2", help="Tên dataset (vd: SetFit/sst2)")
parser.add_argument("--train_path", type=str, required=True, help="Đường dẫn đến file concept_labels_train.npy")
parser.add_argument("--val_path", type=str, default=None, help="(Tuỳ chọn) Đường dẫn đến file concept_labels_val.npy (chỉ dùng cho SetFit/sst2)")
parser.add_argument("--output_prefix", type=str, default=None, help="Thư mục output, mặc định cùng thư mục input train")
args = parser.parse_args()

dataset = args.dataset
concept_set = CFG.concept_set[dataset]
print(concept_set[:5])

# --- load HF labels ---
train_ds = load_dataset(dataset, split="train")
train_y = np.array(train_ds["label"])

val_y = None
if args.dataset == "SetFit/sst2":
    # chỉ SetFit/sst2 mới load validation
    if args.val_path is None:
        raise ValueError("SetFit/sst2 yêu cầu --val_path trỏ tới concept_labels_val.npy")
    val_ds = load_dataset(dataset, split="validation")
    val_y = np.array(val_ds["label"])

# --- load ACS matrices ---
train_sim = np.load(args.train_path)
if train_sim.shape[0] != len(train_y):
    raise ValueError(f"Rows train_sim ({train_sim.shape[0]}) != số mẫu train ({len(train_y)}). Kiểm tra lại đường dẫn và dataset.")

val_sim = None
if args.dataset == "SetFit/sst2":
    val_sim = np.load(args.val_path)
    if val_sim.shape[0] != len(val_y):
        raise ValueError(f"Rows val_sim ({val_sim.shape[0]}) != số mẫu val ({len(val_y)}). Kiểm tra lại val npy/dataset.")

print(f"Loaded train: {train_sim.shape}" + (f", val: {val_sim.shape}" if val_sim is not None else ""))

# --- concept->parent label mapping ---
num_concepts = len(concept_set)
concept_parent = np.array([get_labels(j, dataset) for j in range(num_concepts)])

# --- ACC for TRAIN (vectorized) ---
mask_train = (concept_parent[None, :] == train_y[:, None])
train_sim = np.where(mask_train, train_sim, 0.0)
train_sim = np.maximum(train_sim, 0.0)

# --- ACC for VAL (only for SetFit/sst2) ---
if args.dataset == "SetFit/sst2":
    mask_val = (concept_parent[None, :] == val_y[:, None])
    val_sim = np.where(mask_val, val_sim, 0.0)
    val_sim = np.maximum(val_sim, 0.0)

# --- save ---
if args.output_prefix is None:
    args.output_prefix = "/".join(args.train_path.split("/")[:-1])
os.makedirs(args.output_prefix, exist_ok=True)

np.save(os.path.join(args.output_prefix, "concept_labels_train_acc.npy"), train_sim)
if args.dataset == "SetFit/sst2":
    np.save(os.path.join(args.output_prefix, "concept_labels_val_acc.npy"), val_sim)

print(f"Saved ACC-corrected file(s) to {args.output_prefix}")