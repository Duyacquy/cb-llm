import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import config as CFG
import os

# --- THÊM ARGPARSE ĐỂ NHẬN LỆNH TỪ BÊN NGOÀI ---
parser = argparse.ArgumentParser(description="Verify Noise Concepts")
parser.add_argument("--dataset", type=str, default="Duyacquy/Single_label_medical_abstract", help="HuggingFace dataset name")
parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name (e.g., bert-base-uncased)")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to verify")

args = parser.parse_args()

# Gán biến từ tham số dòng lệnh
DATASET_NAME = args.dataset
MODEL_NAME = args.model_name
NUM_SAMPLES = args.num_samples

# Danh sách Concept Noise (Giữ nguyên danh sách của bạn)
NOISE_CONCEPTS = [
    "@13132", "abcxyz", "qwe098", "##$%12", "zt!9x",
    "77aa!!", "_xoxo_", "mnbvc", "404zz", "ppp@@@",
    "l9l9l", "^&*abc", "kkk000", "1x2y3z", "??!!@@",
    "r4nd0m", "uvw___", "9z9z9", "hehe$$", "___aaa",
    "%%%xyz", "00qaz", "asd###", "!ping!", "zx12zx",
    "@@@nope", "lol123", "8*8*8", "brrr__", "xXx000",
    "aaaabbbccc31", "aaaabbbccc32", "aaaabbbccc33", "aaaabbbccc34", "aaaabbbccc35",
    "aaaabbbccc36", "aaaabbbccc37", "aaaabbbccc38", "aaaabbbccc39", "aaaabbbccc40"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bert_embeddings_and_labels(n_samples):
    print(f"Loading data: {DATASET_NAME} & Model: {MODEL_NAME}...")
    
    # Load dataset
    try:
        ds = load_dataset(DATASET_NAME, split="test")
    except:
        print("Không tìm thấy tập test, thử load tập train...")
        ds = load_dataset(DATASET_NAME, split="train")

    # Lấy subset ngẫu nhiên nếu dataset lớn hơn n_samples
    if len(ds) > n_samples:
        indices = np.random.choice(len(ds), n_samples, replace=False)
        subset = ds.select(indices)
    else:
        subset = ds
    
    # Lấy config cột text/label từ file config.py hoặc tự động đoán
    if DATASET_NAME in CFG.dataset_config:
        text_col = CFG.dataset_config[DATASET_NAME]["text_column"]
        label_col = CFG.dataset_config[DATASET_NAME]["label_column"]
    else:
        # Fallback nếu dataset mới chưa có trong config
        text_col = "text" 
        label_col = "label"
        print(f"Warning: Dataset {DATASET_NAME} chưa có trong config.py, dùng mặc định: text_col='{text_col}', label_col='{label_col}'")
    
    texts = subset[text_col]
    labels = subset[label_col]
    
    # Load BERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # Tokenize concept noise
    print("Encoding Noise Concepts...")
    c_inputs = tokenizer(NOISE_CONCEPTS, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        # Xử lý logic pooler tùy model (đơn giản hóa cho BERT)
        outputs = model(**c_inputs)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
             c_emb = outputs.pooler_output
        else:
             # Fallback cho các model không có pooler_output (như gpt2) -> dùng mean pooling
             c_emb = outputs.last_hidden_state.mean(dim=1)
             
        c_emb = F.normalize(c_emb, p=2, dim=1)
        
    # Get Text Embeddings
    print("Encoding Texts...")
    all_text_embs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        # In tiến độ để không bị treo
        print(f"\rProcessing batch {i//batch_size + 1}/{(len(texts)//batch_size)+1}", end="")
        
        batch_texts = texts[i : i+batch_size]
        t_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**t_inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                t_emb = outputs.pooler_output
            else:
                t_emb = outputs.last_hidden_state.mean(dim=1)

            t_emb = F.normalize(t_emb, p=2, dim=1)
            all_text_embs.append(t_emb.cpu())
            
    print("\nDone encoding.")
    text_embs = torch.cat(all_text_embs, dim=0) 
    return text_embs, c_emb.cpu(), np.array(labels)

def normalize_relu(x):
    # Mô phỏng hàm trong train_FL_from_npy.py
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    x = (x - mean) / (std + 1e-8)
    return F.relu(x)

# --- THỰC THI ---
if __name__ == "__main__":
    # 1. Lấy dữ liệu
    text_embs, concept_embs, labels = get_bert_embeddings_and_labels(NUM_SAMPLES)

    # 2. Tính toán các trạng thái
    # State 1: Before (BERT Embeddings) -> text_embs

    # State 2: Projection (Dot Product)
    scores_raw = text_embs @ concept_embs.T 

    # State 3: Final (Norm + ReLU)
    scores_final = normalize_relu(scores_raw)

    # 3. Tính chỉ số Silhouette
    print("\n--- KẾT QUẢ ĐỊNH LƯỢNG (Silhouette Score) ---")
    try:
        score_bert = silhouette_score(text_embs, labels)
        score_proj = silhouette_score(scores_raw, labels)
        score_final = silhouette_score(scores_final, labels)

        print(f"1. BERT Original:       {score_bert:.4f}")
        print(f"2. Noise Projected:     {score_proj:.4f}")
        print(f"3. Final (Norm+ReLU):   {score_final:.4f}")
    except ValueError as e:
        print(f"Lỗi tính Silhouette (có thể do chỉ có 1 class): {e}")

    # 4. Trực quan hóa
    print("\n--- Đang chạy t-SNE... ---")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')

    vis_bert = tsne.fit_transform(text_embs)
    vis_final = tsne.fit_transform(scores_final)

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    scatter1 = axs[0].scatter(vis_bert[:, 0], vis_bert[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    axs[0].set_title(f"Original BERT Space")
    
    scatter2 = axs[1].scatter(vis_final[:, 0], vis_final[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    axs[1].set_title(f"Noise Concepts Space (After Norm+ReLU)")

    save_path = "verification_plot.png"
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ vào {save_path}")