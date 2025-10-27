import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
import config as CFG

def load_npy(path):
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D array [num_samples, num_concepts], got shape {arr.shape}")
    return torch.tensor(arr, dtype=torch.float32)

def normalize_relu(x, mean, std):
    x = (x - mean) / (std + 1e-8)
    return F.relu(x)

def pick_last_path_item(output_proj):
    if isinstance(output_proj, dict) and "path" in output_proj and len(output_proj["path"]) > 0:
        return output_proj["path"][-1]
    return None

def extract_weights_from_path_item(item, fallback_linear):
    W = item.get("weight", None) if isinstance(item, dict) else None
    b = item.get("bias", None) if isinstance(item, dict) else None
    if W is None or b is None:
        W = fallback_linear.weight.detach().cpu()
        b = fallback_linear.bias.detach().cpu()
    return W, b

def test_accuracy(W, b, feats, y_true):
    logits = feats @ W.T + b
    pred = torch.argmax(logits, dim=-1)
    return (pred == y_true).float().mean().item()

def stratified_split_indices(y_np, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    y_np = np.asarray(y_np)
    idx = np.arange(len(y_np))
    train_idx, val_idx = [], []
    for cls in np.unique(y_np):
        cls_idx = idx[y_np == cls]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        val_idx.append(cls_idx[:n_val])
        train_idx.append(cls_idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

def load_and_normalize_labels(dataset_name, split):
    ds = load_dataset(dataset_name, split=split)
    label_key = CFG.dataset_config[dataset_name]["label_column"]
    raw_y = ds[label_key]

    # ép sang int nếu cần (ví dụ string -> id)
    # nếu là string label thì map về chỉ số ổn định
    if isinstance(raw_y[0], str):
        uniq = sorted(list(set(raw_y)))
        table = {lab: i for i, lab in enumerate(uniq)}
        y_np = np.array([table[v] for v in raw_y], dtype=np.int64)
    else:
        # giả sử là số rồi (có thể 1..K thay vì 0..K-1), ta shift về 0-based
        y_np = np.array(raw_y, dtype=np.int64)
        min_val = y_np.min()
        if min_val != 0:
            y_np = y_np - min_val

    return y_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npy", type=str, required=True,
                        help="ACS features for FULL train (rows align with HF train split order)")
    parser.add_argument("--test_npy", type=str, required=True,
                        help="ACS features for test (rows align with HF test split order)")
    parser.add_argument("--val_npy", type=str, default=None,
                        help="(Optional) ACS features for validation split. "
                             "If provided AND dataset has native validation, we use that; "
                             "otherwise we'll ignore this for datasets without val.")
    parser.add_argument("--dataset", type=str, default="SetFit/sst2")
    parser.add_argument("--saga_epoch", type=int, default=500)
    parser.add_argument("--saga_batch_size", type=int, default=256)
    parser.add_argument("--save_prefix", type=str, default=None,
                        help="If set, save W_g/b_g and normalization prefix_path_*")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Used only when --val_npy is absent (stratify split from train_full)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratified split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset

    print("--------------------TRAIN FL-------------------------")

    has_val = CFG.dataset_config[dataset_name].get("has_val", False) if "has_val" in CFG.dataset_config[dataset_name] else False

    # nhãn train/test (đã chuẩn hoá về int 0..C-1)
    y_train_full = load_and_normalize_labels(dataset_name, split="train")
    y_test_full  = load_and_normalize_labels(dataset_name, split="test")

    test_y = torch.LongTensor(y_test_full)

    # nạp đặc trưng ACS
    train_full_c = load_npy(args.train_npy)  # FULL train ACS
    test_c = load_npy(args.test_npy)         # test ACS

    # chia val
    if args.val_npy is not None and has_val:
        val_c = load_npy(args.val_npy)

        y_val_full = load_and_normalize_labels(dataset_name, split="validation")
        val_y = torch.LongTensor(y_val_full)

        train_idx = np.arange(len(train_full_c))
        train_c = train_full_c
        train_y = torch.LongTensor(y_train_full)
    else:
        train_idx, val_idx = stratified_split_indices(
            y_train_full,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        train_c = train_full_c[train_idx]
        val_c   = train_full_c[val_idx]
        train_y = torch.LongTensor(y_train_full[train_idx])
        val_y   = torch.LongTensor(y_train_full[val_idx])

    train_mean = train_c.mean(0)
    train_std  = train_c.std(0)
    train_c = normalize_relu(train_c, train_mean, train_std)
    val_c   = normalize_relu(val_c,   train_mean, train_std)
    test_c  = normalize_relu(test_c,  train_mean, train_std)

    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    indexed_train_loader = DataLoader(
        indexed_train_ds,
        batch_size=args.saga_batch_size,
        shuffle=True
    )
    val_loader  = DataLoader(
        TensorDataset(val_c, val_y),
        batch_size=args.saga_batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(test_c, test_y),
        batch_size=args.saga_batch_size,
        shuffle=False
    )

    print("dim of concept features: ", train_c.shape[1])
    linear = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset_name])
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.05
    ALPHA = 0.99
    print("training final layer from ACS(train[/val]) and evaluating on ACS(test)...")
    output_proj = glm_saga(
        linear,
        indexed_train_loader,
        STEP_SIZE,
        args.saga_epoch,
        ALPHA,
        k=10,
        val_loader=val_loader,
        test_loader=test_loader,
        do_zero=True,
        n_classes=CFG.class_num[dataset_name],
    )
    print("training done.")

    last_item = pick_last_path_item(output_proj)
    W_g, b_g = extract_weights_from_path_item(last_item, linear)

    acc_test = test_accuracy(W_g, b_g, test_c, test_y)
    print(f"Test accuracy (ACS test via trained FL): {acc_test:.4f}")
    if last_item and "metrics" in last_item and "acc_test" in last_item["metrics"]:
        print(f"Test accuracy (glm_saga reported): {last_item['metrics']['acc_test']:.4f}")

    if args.save_prefix:
        os.makedirs(os.path.dirname(args.save_prefix), exist_ok=True)
        torch.save(W_g,        args.save_prefix + "_W_g.pt")
        torch.save(b_g,        args.save_prefix + "_b_g.pt")
        torch.save(train_mean, args.save_prefix + "_train_mean.pt")
        torch.save(train_std,  args.save_prefix + "_train_std.pt")
        print(f"Saved weights & normalization to prefix: {args.save_prefix}")