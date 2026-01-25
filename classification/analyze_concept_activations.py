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
LABEL_ORDER_CONFIG = {
    # 1. SST2
    "SetFit/sst2": [
        "negative",
        "positive"
    ],

    # 2. UCI Drug Review
    "Duyacquy/UCI_drug": [
        "1", 
        "5", 
        "10"
    ],

    # 3. Pubmed 20k
    "Duyacquy/Pubmed_20k": [
        "BACKGROUND",
        "OBJECTIVE",
        "METHODS",
        "RESULTS",
        "CONCLUSIONS"
    ],

    # 4. Medical Abstract
    "Duyacquy/Single_label_medical_abstract": [
        "neoplasms",
        "digestive_system_diseases",
        "nervous_system_diseases",
        "cardiovascular_diseases",
        "general_pathological_diseases"
    ],

    # 5. Ag News
    "fancyzhx/ag_news": [
        "world",
        "sports",
        "business",
        "sci_tech"
    ],

    # 6. Dbpedia 14
    "fancyzhx/dbpedia_14": [
        "company",
        "educational institution",
        "artist",
        "athlete",
        "office holder",
        "transportation",
        "building",
        "natural place",
        "village",
        "animal",
        "plant",
        "album",
        "film",
        "written work"
    ],

    # 7. Legal Text
    "Duyacquy/Legal_text": [
        "applied",
        "cited",
        "considered",
        "followed",
        "referred to"
    ],

    # 8. Ecommerce Text
    "Duyacquy/Ecommerce_text": [
        "household",
        "books",
        "electronics",
        "clothing_accessories"
    ],

    # 9. Stack Overflow
    "Duyacquy/Stack_overflow_question": [
        "HQ",
        "LQ_EDIT",
        "LQ_CLOSE"
    ]
}

# Số lượng concept được sinh ra cho mỗi class
CONCEPTS_PER_CLASS = 30
# =================================================

def load_npy(path):
    arr = np.load(path)
    return torch.tensor(arr, dtype=torch.float32)

def normalize_relu(x, mean, std):
    # Tái tạo lại logic normalize trong train_FL
    x = (x - mean) / (std + 1e-8)
    return F.relu(x)

def get_representative_label(concept_idx, label_list):
    """
    Map từ index của concept sang tên nhãn đại diện.
    Logic: index // 30 -> thứ tự nhãn
    """
    label_idx = concept_idx // CONCEPTS_PER_CLASS
    if label_idx < len(label_list):
        return label_list[label_idx]
    return "Unknown_or_General"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Tên dataset, vd: Duyacquy/Single_label_medical_abstract")
    parser.add_argument("--test_npy", type=str, required=True, help="Đường dẫn file concept_labels_test.npy")
    parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa file W_g.pt, b_g.pt, mean/std")
    parser.add_argument("--output_json", type=str, default="concept_analysis.json", help="Đường dẫn file output json")
    parser.add_argument("--top_k", type=int, default=10, help="Lấy bao nhiêu concept cao/thấp nhất")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data for {args.dataset}...")

    # 1. Load Weights & Stats để dự đoán
    try:
        W_g = torch.load(os.path.join(args.model_dir, "_W_g.pt"), map_location=device)
        b_g = torch.load(os.path.join(args.model_dir, "_b_g.pt"), map_location=device)
        train_mean = torch.load(os.path.join(args.model_dir, "_train_mean.pt"), map_location=device)
        train_std = torch.load(os.path.join(args.model_dir, "_train_std.pt"), map_location=device)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file weights (W_g, b_g, mean, std) trong thư mục model_dir.")
        return

    # 2. Load Concept Features (Test set)
    test_c = load_npy(args.test_npy).to(device)
    
    # 3. Chuẩn hóa và Dự đoán (Predict)
    # Normalize
    test_c_norm = normalize_relu(test_c, train_mean, train_std)
    
    # Tính Logits và Predict
    logits = test_c_norm @ W_g.T + b_g
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()

    # 4. Load Raw Text & True Labels từ HuggingFace
    # Lưu ý: Phải load đúng split 'test' để khớp với file npy
    print("Loading HF dataset to get raw text...")
    ds = load_dataset(args.dataset, split="test")
    
    # Preprocess để đảm bảo data khớp (clean dataset)
    # Cần import logic preprocess từ file dataset_utils của bạn
    # Giả sử file npy được tạo từ tập test CÓ qua preprocess
    if args.dataset in CFG.dataset_config:
        ds = preprocess(ds, args.dataset, CFG.dataset_config[args.dataset]["text_column"], CFG.dataset_config[args.dataset]["label_column"])
    
    text_col = CFG.dataset_config[args.dataset]["text_column"]
    label_col = CFG.dataset_config[args.dataset]["label_column"]
    
    # Lấy danh sách concept text gốc
    concept_list = CFG.concept_set[args.dataset]
    
    # Lấy danh sách label đại diện (định nghĩa ở đầu file)
    representative_labels = LABEL_ORDER_CONFIG.get(args.dataset, [])
    if not representative_labels:
        print(f"Warning: Chưa định nghĩa LABEL_ORDER_CONFIG cho {args.dataset}, dùng label mặc định từ config.")
        # Fallback lấy từ config nếu không định nghĩa tay
        representative_labels = CFG.concepts_from_labels.get(args.dataset, [])

    # Lấy map label ID sang tên thật (cho field 'label' và 'predict')
    # Ví dụ: 0 -> "neoplasms", 1 -> "digestive..."
    # Nếu trong config.py concepts_from_labels chính là tên class thì dùng luôn
    idx_to_class_name = {i: name for i, name in enumerate(CFG.concepts_from_labels[args.dataset])}

    final_results = []
    
    # 5. Loop và Extract
    print("Analyzing concepts...")
    # Convert feature về CPU numpy để xử lý vòng lặp cho nhanh
    # Lưu ý: concept_activate nên lấy từ giá trị RAW (test_c) hay sau ReLU (test_c_norm)?
    # Thường user muốn xem concept nào có điểm tương đồng cao nhất -> Dùng RAW (Cosine Sim).
    test_c_np = test_c.cpu().numpy() 
    
    raw_texts = ds[text_col]
    raw_labels = ds[label_col] # Đây có thể là Int hoặc String tùy dataset

    # Đảm bảo độ dài khớp nhau
    min_len = min(len(raw_texts), len(test_c_np))
    if len(raw_texts) != len(test_c_np):
        print(f"Warning: Mismatch length (HF: {len(raw_texts)} vs NPY: {len(test_c_np)}). Truncating to {min_len}.")

    for i in range(min_len):
        # Thông tin cơ bản
        text_content = raw_texts[i]
        
        # Xử lý Label thật
        true_label_val = raw_labels[i]
        if isinstance(true_label_val, int):
            true_label_str = idx_to_class_name.get(true_label_val, str(true_label_val))
        else:
            true_label_str = str(true_label_val)

        # Xử lý Label dự đoán
        pred_label_val = predictions[i]
        pred_label_str = idx_to_class_name.get(pred_label_val, str(pred_label_val))

        # Lấy row score
        scores = test_c_np[i] # Shape: (num_concepts,)
        
        # Sắp xếp index theo score tăng dần
        sorted_indices = np.argsort(scores)
        
        # Lấy Top K (cuối mảng) và Bottom K (đầu mảng)
        # Lưu ý: Top K cần đảo ngược để cái cao nhất lên đầu
        top_indices = sorted_indices[-args.top_k:][::-1]
        bottom_indices = sorted_indices[:args.top_k]
        
        # Gộp lại danh sách concept quan tâm
        activated_concepts = []
        
        # Helper để add vào list
        def add_concept_info(idx, type_desc):
            c_text = concept_list[idx] if idx < len(concept_list) else "ERR_IDX"
            c_rep_label = get_representative_label(idx, representative_labels)
            
            activated_concepts.append({
                "concept": c_text,
                "label_representative": c_rep_label,
                "score": float(scores[idx]),
                "type": type_desc # "top" or "bottom"
            })

        for idx in top_indices:
            add_concept_info(idx, "top_high_score")
            
        for idx in bottom_indices:
            add_concept_info(idx, "bottom_low_score")

        # Tạo entry
        entry = {
            "id": i,
            "text": text_content,
            "label": true_label_str,
            "predict": pred_label_str,
            "concept_activate": activated_concepts
        }
        final_results.append(entry)

    # 6. Save JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    print(f"Done! Results saved to {args.output_json}")

if __name__ == "__main__":
    main()