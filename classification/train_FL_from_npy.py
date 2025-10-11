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
        # fall back to the model's current params
        W = fallback_linear.weight.detach().cpu()
        b = fallback_linear.bias.detach().cpu()
    return W, b

def test_accuracy(W, b, feats, y_true):
    # feats: [N, C], W: [num_labels, C], b: [num_labels]
    logits = feats @ W.T + b
    pred = torch.argmax(logits, dim=-1)
    acc = (pred == y_true).float().mean().item()
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npy", type=str, required=True, help="ACC features for train")
    parser.add_argument("--val_npy", type=str, default=None, help="ACC features for val (optional)")
    parser.add_argument("--test_npy", type=str, required=True, help="ACS features for test (NO ACC)")
    parser.add_argument("--dataset", type=str, default="SetFit/sst2")
    parser.add_argument("--saga_epoch", type=int, default=500)
    parser.add_argument("--saga_batch_size", type=int, default=256)
    parser.add_argument("--save_prefix", type=str, default=None, help="If set, save W_g and b_g to this prefix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset

    # --- Load concept features ---
    # Train/Val: ACC; Test: ACS (no ACC)
    train_c = load_npy(args.train_npy)
    val_c = load_npy(args.val_npy) if args.val_npy else None
    test_c = load_npy(args.test_npy)

    # --- Normalize like train_FL.py (fit on TRAIN only), then ReLU ---
    train_mean = train_c.mean(0)
    train_std  = train_c.std(0)
    train_c = normalize_relu(train_c, train_mean, train_std)
    if val_c is not None:
        val_c = normalize_relu(val_c, train_mean, train_std)
    test_c = normalize_relu(test_c, train_mean, train_std)

    # --- Load labels (HF datasets) ---
    train_ds = load_dataset(dataset_name, split="train")
    train_y = torch.LongTensor(train_ds["label"])
    if val_c is not None:
        val_ds = load_dataset(dataset_name, split="validation")
        val_y = torch.LongTensor(val_ds["label"])
    test_ds = load_dataset(dataset_name, split="test")
    test_y = torch.LongTensor(test_ds["label"])

    # --- Dataloaders for GLM-SAGA ---
    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    if val_c is not None:
        val_loader = DataLoader(TensorDataset(val_c, val_y), batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_c, test_y), batch_size=args.saga_batch_size, shuffle=False)

    # --- Final linear classifier ---
    print("dim of concept features: ", train_c.shape[1])
    linear = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset_name])
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {"max_reg": {"nongrouped": 0.0007}}

    print("training final layer from ACC(train[/val]) and evaluating on ACS(test)...")
    if val_c is not None:
        output_proj = glm_saga(
            linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
            val_loader=val_loader, test_loader=test_loader, do_zero=True,
            n_classes=CFG.class_num[dataset_name]
        )
    else:
        output_proj = glm_saga(
            linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
            test_loader=test_loader, do_zero=True,
            n_classes=CFG.class_num[dataset_name]
        )

    print("training done.")

    # --- Extract weights (prefer from solver's path; else fall back to model) ---
    last_item = pick_last_path_item(output_proj)
    W_g, b_g = extract_weights_from_path_item(last_item, linear)

    # --- Compute test accuracy ourselves (robust to solver's optional metrics) ---
    acc = test_accuracy(W_g, b_g, test_c, test_y)
    print(f"Test accuracy (ACS test via trained FL): {acc:.4f}")

    # Also print solver-reported acc if available
    if last_item and "metrics" in last_item and "acc_test" in last_item["metrics"]:
        print(f"Test accuracy (glm_saga reported): {last_item['metrics']['acc_test']:.4f}")

    # --- Optional save ---
    if args.save_prefix:
        os.makedirs(os.path.dirname(args.save_prefix), exist_ok=True)
        torch.save(W_g, args.save_prefix + "_W_g.pt")
        torch.save(b_g, args.save_prefix + "_b_g.pt")
        torch.save(train_mean, args.save_prefix + "_train_mean.pt")
        torch.save(train_std, args.save_prefix + "_train_std.pt")
        print(f"Saved weights & normalization to prefix: {args.save_prefix}")