import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
import config as CFG
from dataset_utils import preprocess

# --- CẤU HÌNH ---
LABEL_MAP = {
    0: "BACKGROUND",
    1: "OBJECTIVE",
    2: "METHODS",
    3: "RESULTS",
    4: "CONCLUSIONS"
}

# Định nghĩa phạm vi index cho từng nhãn như bạn mô tả
# Mỗi nhãn 30 concept
CONCEPT_RANGES = {
    0: (0, 30),    # Background
    1: (30, 60),   # Objective
    2: (60, 90),   # Methods
    3: (90, 120),  # Results
    4: (120, 150)  # Conclusions
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Tên dataset (vd: khalid/pubmed_20k_rct)")
    parser.add_argument("--sim_model_name", type=str, default="roberta-base", help="Model embedding")
    parser.add_argument("--num_samples", type=int, default=3000, help="Số lượng mẫu để vẽ biểu đồ (đừng quá lớn kẻo chậm)")
    parser.add_argument("--output_dir", type=str, default="./distribution_analysis", help="Thư mục lưu ảnh")
    return parser.parse_args()

def get_concept_scores(model, tokenizer, texts, concepts, device):
    """Tính Cosine Similarity giữa texts và concepts"""
    # 1. Encode Concepts
    c_inputs = tokenizer(concepts, return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        c_out = model(**c_inputs)
        # Sử dụng Mean Pooling cho nhất quán
        c_emb = c_out.last_hidden_state.mean(dim=1)
        c_emb = F.normalize(c_emb, p=2, dim=1)

    # 2. Encode Texts (Batching)
    batch_size = 32
    text_embs = []
    print("Computing text embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i+batch_size]
        t_inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            t_out = model(**t_inputs)
            t_emb_batch = t_out.last_hidden_state.mean(dim=1)
            t_emb_batch = F.normalize(t_emb_batch, p=2, dim=1)
            text_embs.append(t_emb_batch.cpu())
    
    text_embs = torch.cat(text_embs, dim=0).to(device)
    
    # 3. Compute Cosine Similarity Matrix [N_texts, N_concepts]
    # text_embs: [N, D], c_emb: [M, D] -> [N, M]
    scores = torch.mm(text_embs, c_emb.T).cpu().numpy()
    return scores

def get_top_tfidf_keywords(texts, labels, top_n=5):
    """Tìm top keywords đặc trưng cho từng class dựa trên TF-IDF trung bình"""
    print("Computing TF-IDF...")
    # Giới hạn max_features để chạy nhanh và lọc bớt nhiễu
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(texts)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    top_keywords_per_class = {}
    
    for label_idx in LABEL_MAP.keys():
        # Lấy các dòng thuộc label này
        mask = (np.array(labels) == label_idx)
        if not np.any(mask): continue
        
        # Tính trung bình TF-IDF của từng từ trong class này
        mean_tfidf = np.array(X[mask].mean(axis=0)).flatten()
        
        # Lấy top N index cao nhất
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        top_words = feature_names[top_indices]
        
        top_keywords_per_class[label_idx] = list(top_words)
        
    return tfidf, top_keywords_per_class

def plot_kde(data_df, value_col, label_col, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Vẽ KDE plot phân theo label
    # palette định màu sắc cho các label
    sns.kdeplot(
        data=data_df, 
        x=value_col, 
        hue=label_col, 
        fill=True, 
        common_norm=False, 
        palette="tab10",
        alpha=0.3,
        linewidth=2
    )
    
    plt.title(title, fontsize=12)
    plt.xlabel("Score")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading dataset: {args.dataset}")
    # Load dataset
    ds = load_dataset(args.dataset, split='train') # Hoặc 'validation' nếu muốn
    
    # Preprocess
    text_col = CFG.dataset_config[args.dataset]["text_column"]
    label_col = CFG.dataset_config[args.dataset]["label_column"]
    ds = preprocess(ds, args.dataset, text_col, label_col)
    
    # Sampling
    if len(ds) > args.num_samples:
        indices = np.random.choice(len(ds), args.num_samples, replace=False)
        texts = [ds[int(i)][text_col] for i in indices]
        labels = [ds[int(i)][label_col] for i in indices]
    else:
        texts = ds[text_col]
        labels = ds[label_col]
        
    # Chuyển label sang tên gọi (BACKGROUND,...) để vẽ cho đẹp
    label_names = [LABEL_MAP.get(l, str(l)) for l in labels]
    df_base = pd.DataFrame({"label": label_names})

    # --- PHẦN 1: COSINE SIMILARITY VỚI CONCEPTS ---
    try:
        concept_list = CFG.concept_set[args.dataset]
        print(f"Total concepts found: {len(concept_list)}")
    except:
        print("Error: Could not find concepts in config.CFG")
        return

    # Load Model Embedding
    print(f"Loading model: {args.sim_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.sim_model_name)
    model = AutoModel.from_pretrained(args.sim_model_name).to(device)
    
    # Chọn concept đại diện để vẽ
    # Mỗi nhãn lấy 3 concept: Đầu, Giữa, Cuối của khoảng range
    selected_concept_indices = []
    for lbl_idx, (start, end) in CONCEPT_RANGES.items():
        if end > len(concept_list): break
        # Lấy 3 concept đại diện: đầu, giữa, gần cuối range
        indices = [start, start + 15, end - 1] 
        selected_concept_indices.extend([(idx, lbl_idx) for idx in indices])

    # Chỉ tính score cho các concept được chọn để tiết kiệm thời gian
    # Hoặc tính hết rồi trích xuất
    # Ở đây ta tính score cho TẤT CẢ text với CÁC CONCEPT ĐƯỢC CHỌN
    target_concepts = [concept_list[i] for i, _ in selected_concept_indices]
    
    print(f"Calculating similarity for {len(target_concepts)} selected concepts...")
    scores = get_concept_scores(model, tokenizer, texts, target_concepts, device)
    
    print("Plotting Concept Distributions...")
    concept_dir = os.path.join(args.output_dir, "concepts")
    os.makedirs(concept_dir, exist_ok=True)
    
    for i, (concept_idx, target_label_idx) in enumerate(selected_concept_indices):
        concept_name = concept_list[concept_idx]
        target_label_name = LABEL_MAP[target_label_idx]
        
        # Chuẩn bị data vẽ
        df_plot = df_base.copy()
        df_plot["score"] = scores[:, i]
        
        # Tên file: Label_ConceptIndex_FirstFewWords
        safe_name = "".join([c if c.isalnum() else "_" for c in concept_name[:30]])
        fname = os.path.join(concept_dir, f"{target_label_name}_idx{concept_idx}_{safe_name}.png")
        
        title = f"Concept: '{concept_name}'\n(Belongs to Class: {target_label_name})"
        plot_kde(df_plot, "score", "label", title, fname)

    # --- PHẦN 2: TF-IDF ---
    print("\n--- TF-IDF Analysis ---")
    tfidf_model, top_keywords = get_top_tfidf_keywords(texts, labels, top_n=5)
    
    # Tính score TF-IDF cho toàn bộ text với các từ khóa này
    # Ta cần lấy index của các từ khóa này trong bộ từ điển
    vocab = tfidf_model.vocabulary_
    
    tfidf_matrix = tfidf_model.transform(texts)
    
    tfidf_dir = os.path.join(args.output_dir, "tfidf")
    os.makedirs(tfidf_dir, exist_ok=True)
    
    print("Plotting TF-IDF Distributions...")
    for label_idx, keywords in top_keywords.items():
        label_name = LABEL_MAP[label_idx]
        for keyword in keywords:
            if keyword not in vocab: continue
            
            kw_idx = vocab[keyword]
            # Lấy cột điểm của từ này (sparse matrix -> dense array)
            kw_scores = tfidf_matrix[:, kw_idx].toarray().flatten()
            
            df_plot = df_base.copy()
            df_plot["score"] = kw_scores
            
            fname = os.path.join(tfidf_dir, f"{label_name}_{keyword}.png")
            title = f"Keyword: '{keyword}' (Top feature for {label_name})"
            
            plot_kde(df_plot, "score", "label", title, fname)

    print(f"\nDONE! Check results in: {args.output_dir}")

if __name__ == "__main__":
    main()