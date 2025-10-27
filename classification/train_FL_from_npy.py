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

def compute_metrics(pred, y_true, num_classes):
    # pred, y_true: torch.LongTensor [N]
    pred_np = pred.cpu().numpy()
    y_np = y_true.cpu().numpy()

    # confusion matrix counts per class
    # tp[c], fp[c], fn[c]
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        tp[c] = np.sum((pred_np == c) & (y_np == c))
        fp[c] = np.sum((pred_np == c) & (y_np != c))
        fn[c] = np.sum((pred_np != c) & (y_np == c))

    # precision_c = tp / (tp+fp), recall_c = tp / (tp+fn)
    # handle divide-by-zero -> 0
    precision_c = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp) != 0)
    recall_c    = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn) != 0)

    # f1_c = 2 * p * r / (p+r)
    f1_c = np.divide(
        2 * precision_c * recall_c,
        precision_c + recall_c,
        out=np.zeros_like(precision_c, dtype=float),
        where=(precision_c + recall_c) != 0
    )

    precision_macro = precision_c.mean()
    recall_macro    = recall_c.mean()
    f1_macro        = f1_c.mean()

    return precision_macro, recall_macro, f1_macro

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

    if isinstance(raw_y[0], str):
        uniq = sorted(list(set(raw_y)))
        table = {lab: i for i, lab in enumerate(uniq)}
        y_np = np.array([table[v] for v in raw_y], dtype=np.int64)
    else:
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

    y_train_full = load_and_normalize_labels(dataset_name, split="train")
    y_test_full  = load_and_normalize_labels(dataset_name, split="test")

    test_y = torch.LongTensor(y_test_full)

    train_full_c = load_npy(args.train_npy)
    test_c = load_npy(args.test_npy)

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

    # tính logits test và các metric
    logits_test = test_c @ W_g.T + b_g
    pred_test = torch.argmax(logits_test, dim=-1)

    acc_test = (pred_test == test_y).float().mean().item()
    precision_macro, recall_macro, f1_macro = compute_metrics(
        pred_test,
        test_y,
        num_classes=CFG.class_num[dataset_name]
    )

    print(f"Test accuracy:  {acc_test:.4f}")
    print(f"Test precision (macro): {precision_macro:.44f}")
    print(f"Test recall    (macro): {recall_macro:.4f}")
    print(f"Test f1        (macro): {f1_macro:.4f}")

    if last_item and "metrics" in last_item and "acc_test" in last_item["metrics"]:
        print(f"Test accuracy (glm_saga reported): {last_item['metrics']['acc_test']:.4f}")

    if args.save_prefix:
        os.makedirs(os.path.dirname(args.save_prefix), exist_ok=True)
        torch.save(W_g,        args.save_prefix + "_W_g.pt")
        torch.save(b_g,        args.save_prefix + "_b_g.pt")
        torch.save(train_mean, args.save_prefix + "_train_mean.pt")
        torch.save(train_std,  args.save_prefix + "_train_std.pt")
        print(f"Saved weights & normalization to prefix: {args.save_prefix}")