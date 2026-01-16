import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import config as CFG
from dataset_utils import preprocess

# --- CẤU HÌNH ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Tên dataset (vd: Duyacquy/Single_label_medical_abstract)")
    parser.add_argument("--train_npy", type=str, required=True, help="Path file npy train")
    parser.add_argument("--test_npy", type=str, required=True, help="Path file npy test")
    parser.add_argument("--sim_model_name", type=str, default="roberta-base", help="Model dùng để test Negation (phải khớp với model tạo npy)")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Nơi lưu biểu đồ")
    return parser.parse_args()

def load_data(args):
    print(f"Loading dataset: {args.dataset}...")
    # Load dataset gốc
    dataset = load_dataset(args.dataset)
    text_col = CFG.dataset_config[args.dataset]["text_column"]
    label_col = CFG.dataset_config[args.dataset]["label_column"]
    
    # Preprocess
    train_ds = preprocess(dataset['train'], args.dataset, text_col, label_col)
    test_ds = preprocess(dataset['test'], args.dataset, text_col, label_col)
    
    # Load Concept Features (NPY)
    print("Loading concept features (NPY)...")
    X_concept_train = np.load(args.train_npy)
    X_concept_test = np.load(args.test_npy)
    
    # Lấy label và text
    y_train = np.array(train_ds[label_col])
    texts_train = train_ds[text_col]
    
    return texts_train, y_train, X_concept_train, X_concept_test, text_col, label_col

def analyze_distributions(texts_train, y_train, X_concept_train, concept_names, output_dir):
    print("\n--- 1. DISTRIBUTION ANALYSIS: TF-IDF vs COSINE ---")
    
    # 1.1 Tìm Concept quan trọng nhất (Discriminative Concept) bằng Logistic Regression
    lr_concept = LogisticRegression(max_iter=1000)
    lr_concept.fit(X_concept_train, y_train)
    
    # Lấy class đầu tiên làm ví dụ (Binary hoặc Multi-class đều lấy class 0 vs Rest)
    target_class = 0
    # Tìm feature có trọng số cao nhất cho target_class
    if lr_concept.coef_.ndim == 1: # Binary
        top_concept_idx = np.argmax(np.abs(lr_concept.coef_))
        coef_val = lr_concept.coef_[top_concept_idx]
    else: # Multi-class
        top_concept_idx = np.argmax(lr_concept.coef_[target_class])
        coef_val = lr_concept.coef_[target_class][top_concept_idx]
        
    top_concept_name = concept_names[top_concept_idx] if top_concept_idx < len(concept_names) else f"Concept_{top_concept_idx}"
    print(f"Top Discriminative Concept for Class {target_class}: '{top_concept_name}' (Index: {top_concept_idx})")

    # 1.2 Tìm Keyword quan trọng nhất bằng TF-IDF
    print("Computing TF-IDF...")
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X_tfidf = tfidf.fit_transform(texts_train)
    
    lr_tfidf = LogisticRegression(max_iter=1000)
    lr_tfidf.fit(X_tfidf, y_train)
    
    if lr_tfidf.coef_.ndim == 1:
        top_word_idx = np.argmax(np.abs(lr_tfidf.coef_))
    else:
        top_word_idx = np.argmax(lr_tfidf.coef_[target_class])
        
    top_word = tfidf.get_feature_names_out()[top_word_idx]
    print(f"Top Discriminative Word for Class {target_class}: '{top_word}'")

    # 1.3 Vẽ biểu đồ so sánh
    # Tách dữ liệu thành 2 nhóm: Thuộc Class 0 (Positive) và Không thuộc Class 0 (Negative)
    is_target = (y_train == target_class)
    
    # Lấy điểm số
    concept_scores_pos = X_concept_train[is_target, top_concept_idx]
    concept_scores_neg = X_concept_train[~is_target, top_concept_idx]
    
    tfidf_scores_pos = X_tfidf[is_target, top_word_idx].toarray().flatten()
    tfidf_scores_neg = X_tfidf[~is_target, top_word_idx].toarray().flatten()
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot TF-IDF
    sns.histplot(tfidf_scores_pos, color='blue', label=f'Class {target_class}', ax=axes[0], kde=True, stat="density", element="step", alpha=0.3)
    sns.histplot(tfidf_scores_neg, color='red', label=f'Other Classes', ax=axes[0], kde=True, stat="density", element="step", alpha=0.3)
    axes[0].set_title(f"TF-IDF Score Distribution\nWord: '{top_word}'")
    axes[0].set_xlabel("TF-IDF Score")
    axes[0].legend()
    
    # Plot Cosine
    sns.histplot(concept_scores_pos, color='blue', label=f'Class {target_class}', ax=axes[1], kde=True, stat="density", element="step", alpha=0.3)
    sns.histplot(concept_scores_neg, color='red', label=f'Other Classes', ax=axes[1], kde=True, stat="density", element="step", alpha=0.3)
    axes[1].set_title(f"Cosine Similarity Distribution\nConcept: '{top_concept_name}'")
    axes[1].set_xlabel("Cosine Similarity Score")
    axes[1].legend()
    
    save_path = os.path.join(output_dir, "distribution_overlap.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved distribution plot to {save_path}")

def run_negation_test(sim_model_name, concept_names, output_dir):
    print("\n--- 2. NEGATION BLINDNESS TEST ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model embedding để test trực tiếp
    print(f"Loading embedding model: {sim_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(sim_model_name)
    model = AutoModel.from_pretrained(sim_model_name).to(device)
    
    # Hàm lấy embedding
    def get_emb(text_list):
        inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**inputs)
            # Tùy model mà pooling khác nhau, ở đây demo mean pooling (phổ biến)
            # Nếu dùng roberta-base gốc thì thường dùng pooler_output hoặc mean
            if "roberta" in sim_model_name or "bert" in sim_model_name:
                 emb = out.pooler_output
            else:
                 emb = out.last_hidden_state.mean(dim=1)
        return F.normalize(emb, p=2, dim=1)

    # Chọn 1 concept ngẫu nhiên để test
    test_concept = concept_names[0] # Lấy concept đầu tiên
    print(f"Testing on Concept: '{test_concept}'")
    
    # Tạo các biến thể câu
    scenarios = [
        f"The patient has {test_concept}.",          # Khẳng định (Positive)
        f"Patient denies {test_concept}.",           # Phủ định (Negative)
        f"No evidence of {test_concept}.",           # Phủ định mạnh
        f"History of {test_concept}.",               # Trung tính
    ]
    labels = ["Positive", "Denies", "No Evidence", "History"]
    
    concept_emb = get_emb([test_concept])
    scenario_embs = get_emb(scenarios)
    
    # Tính Cosine Similarity
    scores = (scenario_embs @ concept_emb.T).cpu().numpy().flatten()
    
    # In kết quả
    print("-" * 40)
    for lbl, txt, sc in zip(labels, scenarios, scores):
        print(f"[{lbl}] Text: '{txt}' \n   -> Similarity: {sc:.4f}")
    print("-" * 40)
    
    # Vẽ biểu đồ Bar Chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, scores, color=['green', 'red', 'red', 'orange'])
    plt.title(f"Negation Blindness of Embedding Model\nConcept: {test_concept}")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1.1)
    
    # Thêm giá trị lên cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha='center', va='bottom')

    save_path = os.path.join(output_dir, "negation_test.png")
    plt.savefig(save_path)
    print(f"Saved negation test chart to {save_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Concept Names từ config
    try:
        concept_names = CFG.concept_set[args.dataset]
    except KeyError:
        print(f"Error: Dataset {args.dataset} not found in config.CFG.concept_set")
        return

    # 1. Load Data
    texts_train, y_train, X_concept_train, _, _, _ = load_data(args)
    
    # 2. Phân tích phân bố (Distribution Overlap)
    analyze_distributions(texts_train, y_train, X_concept_train, concept_names, args.output_dir)
    
    # 3. Test Phủ định (Negation)
    # Lưu ý: sim_model_name cần khớp với huggingface model ID (vd: roberta-base, sentence-transformers/all-mpnet-base-v2)
    run_negation_test(args.sim_model_name, concept_names, args.output_dir)
    
    print("\n[DONE] All proofs generated.")

if __name__ == "__main__":
    main()