import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
from datasets import load_dataset
import config as CFG
from dataset_utils import preprocess

# ================= CONFIGURATION =================
# Đã đổi tên nhãn đại diện thành số 1, 2, 3... theo yêu cầu của bạn
LABEL_ORDER_CONFIG = {
    "SetFit/sst2": ["negative", "positive"],
    "Duyacquy/UCI_drug": ["1", "5", "10"],
    "Duyacquy/Pubmed_20k": ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"],
    
    # Đã sửa thành 1, 2, 3, 4, 5 tương ứng với thứ tự concept neoplasms -> general
    "Duyacquy/Single_label_medical_abstract": ["1", "2", "3", "4", "5"],
    
    "fancyzhx/ag_news": ["world", "sports", "business", "sci_tech"],
    "fancyzhx/dbpedia_14": [
        "company", "educational institution", "artist", "athlete", "office holder",
        "transportation", "building", "natural place", "village", "animal",
        "plant", "album", "film", "written work"
    ],
    "Duyacquy/Legal_text": ["applied", "cited", "considered", "followed", "referred to"],
    "Duyacquy/Ecommerce_text": ["household", "books", "electronics", "clothing_accessories"],
    "Duyacquy/Stack_overflow_question": ["HQ", "LQ_EDIT", "LQ_CLOSE"]
}

CONCEPTS_PER_CLASS = 30 
# =================================================

def load_npy(path):
    arr = np.load(path)
    return torch.tensor(arr, dtype=torch.float32)

def normalize_relu(x, mean, std):
    x = (x - mean) / (std + 1e-8)
    return F.relu(x)

def get_representative_label(concept_idx, label_list):
    label_idx = concept_idx // CONCEPTS_PER_CLASS
    if label_idx < len(label_list):
        return label_list[label_idx]
    return "Unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_npy", type=str, required=True)
    # SỬA: Dùng save_prefix thay vì model_dir để khớp với cách lưu file
    parser.add_argument("--save_prefix", type=str, required=True, 
                        help="Prefix đường dẫn file weight, ví dụ: .../FL (script sẽ tự thêm _W_g.pt)")
    parser.add_argument("--output_json", type=str, default="concept_analysis.json")
    parser.add_argument("--top_k", type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data for {args.dataset}...")

    # 1. Load Weights (SỬA: dùng logic cộng chuỗi thay vì os.path.join)
    try:
        # save_prefix ví dụ là ".../FL" -> file thực là ".../FL_W_g.pt"
        W_g = torch.load(args.save_prefix + "_W_g.pt", map_location=device)
        b_g = torch.load(args.save_prefix + "_b_g.pt", map_location=device)
        train_mean = torch.load(args.save_prefix + "_train_mean.pt", map_location=device)
        train_std = torch.load(args.save_prefix + "_train_std.pt", map_location=device)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file weights tại prefix: {args.save_prefix}")
        print("Vui lòng kiểm tra lại đường dẫn '--save_prefix'.")
        return

    # 2. Load Concept Features
    test_c = load_npy(args.test_npy).to(device)
    test_c_norm = normalize_relu(test_c, train_mean, train_std)
    
    # 3. Predict
    logits = test_c_norm @ W_g.T + b_g
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()

    # 4. Load Raw Text & Label
    print("Loading HF dataset...")
    ds = load_dataset(args.dataset, split="test")
    if args.dataset in CFG.dataset_config:
        ds = preprocess(ds, args.dataset, CFG.dataset_config[args.dataset]["text_column"], CFG.dataset_config[args.dataset]["label_column"])
    
    text_col = CFG.dataset_config[args.dataset]["text_column"]
    label_col = CFG.dataset_config[args.dataset]["label_column"]
    
    concept_list = CFG.concept_set[args.dataset]
    representative_labels = LABEL_ORDER_CONFIG.get(args.dataset, [])
    
    if not representative_labels:
        representative_labels = CFG.concepts_from_labels.get(args.dataset, [])

    final_results = []
    print("Analyzing concepts...")
    
    test_c_np = test_c.cpu().numpy()
    raw_texts = ds[text_col]
    raw_labels = ds[label_col]

    min_len = min(len(raw_texts), len(test_c_np))

    for i in range(min_len):
        text_content = raw_texts[i]
        
        # --- XỬ LÝ ĐỒNG BỘ LABEL VÀ PREDICT (0-4 vs 1-5) ---
        true_label_val = raw_labels[i] # Giá trị gốc từ dataset (đã là 1-5)
        pred_label_val = predictions[i] # Giá trị từ model (đang là 0-4)
        
        # Riêng cho Medical Abstract, cộng 1 vào predict để khớp với label gốc 1-5
        if args.dataset == "Duyacquy/Single_label_medical_abstract":
            pred_label_str = str(pred_label_val + 1)
            true_label_str = str(true_label_val) # Giữ nguyên gốc
        else:
            # Các dataset khác giữ nguyên logic cũ hoặc map theo tên
            # Nếu cần map tên class (vd neoplasms) thì dùng dict mapping ở đây
            # Nhưng bạn yêu cầu output là số 1-5 nên ta để string số
            pred_label_str = str(pred_label_val)
            true_label_str = str(true_label_val)

        # ---------------------------------------------------

        scores = test_c_np[i]
        sorted_indices = np.argsort(scores)
        
        top_indices = sorted_indices[-args.top_k:][::-1]
        bottom_indices = sorted_indices[:args.top_k]
        
        activated_concepts = []
        
        def add_concept_info(idx, type_desc):
            c_text = concept_list[idx] if idx < len(concept_list) else "ERR"
            c_rep_label = get_representative_label(idx, representative_labels)
            
            activated_concepts.append({
                "concept": c_text,
                "label_representative": c_rep_label,
                "score": float(scores[idx]),
                "type": type_desc
            })

        for idx in top_indices:
            add_concept_info(idx, "top_high_score")
            
        for idx in bottom_indices:
            add_concept_info(idx, "bottom_low_score")

        entry = {
            "id": i,
            "text": text_content,
            "label": true_label_str,
            "predict": pred_label_str,
            "concept_activate": activated_concepts
        }
        final_results.append(entry)

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    print(f"Done! Saved to {args.output_json}")

if __name__ == "__main__":
    main()