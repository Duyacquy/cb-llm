import argparse
import os
import time
import sys

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

import config as CFG
from dataset_utils import train_val_test_split
from utils import mean_pooling, decorate_dataset, decorate_concepts

parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.add_argument(
    "--dataset",
    type=str,
    default="SetFit/sst2",
    help="Tên dataset HuggingFace, ví dụ: SetFit/sst2, Duyacquy/Single_label_medical_abstract"
)
parser.add_argument(
    "--concept_text_sim_model",
    type=str,
    default="mpnet",
    help="mpnet, simcse hoặc angle"
)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)

args = parser.parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SimDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc  # dict of tokenized inputs

    def __getitem__(self, idx):
        # return dict[str -> tensor] for a single row
        return {k: torch.tensor(v[idx]) for k in self.enc.items()}

    def __len__(self):
        return len(self.enc["input_ids"])


def build_sim_loader(enc, batch_size, num_workers=0, shuffle=False):
    ds = SimDataset(enc)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

print("--------------------GET CONCEPT LABEL-------------------------")
print(f"[INFO] loading dataset {args.dataset}")
text_col = CFG.dataset_config[args.dataset]["text_column"]
label_col = CFG.dataset_config[args.dataset]["label_column"]

train_dataset, test_dataset = train_val_test_split(
    args.dataset,
    label_column=label_col,
    ratio=0.2,
    has_val=CFG.dataset_config[args.dataset].get("has_val", False),
)

print(f"[INFO] train size = {len(train_dataset)}, test size = {len(test_dataset)}")

concept_set = CFG.concept_set[args.dataset]
print(f"[INFO] num concepts = {len(concept_set)}")

if args.concept_text_sim_model == "mpnet":
    print("[INFO] using sentence-transformers/all-mpnet-base-v2")
    tokenizer_sim = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    sim_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
    sim_model.eval()
    sim_batch_size = 256

elif args.concept_text_sim_model == "simcse":
    print("[INFO] using princeton-nlp/sup-simcse-bert-base-uncased")
    tokenizer_sim = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    sim_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
    sim_model.eval()
    sim_batch_size = 256

elif args.concept_text_sim_model == "angle":
    print("[INFO] using SeanLee97/angle-llama-7b-nli-v2 (PEFT)")
    cfg = PeftConfig.from_pretrained("SeanLee97/angle-llama-7b-nli-v2")
    tokenizer_sim = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    base_lm = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path).bfloat16()
    sim_model = PeftModel.from_pretrained(base_lm, "SeanLee97/angle-llama-7b-nli-v2")
    sim_model = sim_model.to(device)
    sim_model.eval()
    sim_batch_size = 8 

    train_dataset = train_dataset.map(
        decorate_dataset,
        fn_kwargs={"d": args.dataset}
    )
    test_dataset = test_dataset.map(
        decorate_dataset,
        fn_kwargs={"d": args.dataset}
    )
    concept_set = decorate_concepts(concept_set)

else:
    raise Exception("concept-text sim model should be mpnet, simcse or angle")

print("[INFO] tokenizing train/test for similarity model...")

encoded_sim_train = train_dataset.map(
    lambda e: tokenizer_sim(
        e[text_col],
        padding=True,
        truncation=True,
        max_length=args.max_length
    ),
    batched=True,
    batch_size=len(train_dataset)
)

encoded_sim_test = test_dataset.map(
    lambda e: tokenizer_sim(
        e[text_col],
        padding=True,
        truncation=True,
        max_length=args.max_length
    ),
    batched=True,
    batch_size=len(test_dataset)
)

encoded_sim_train = encoded_sim_train.remove_columns([text_col])
encoded_sim_test  = encoded_sim_test.remove_columns([text_col])

encoded_sim_train = {k: np.array(v) for k, v in encoded_sim_train.to_dict().items()}
encoded_sim_test  = {k: np.array(v) for k, v in encoded_sim_test.to_dict().items()}

train_loader = build_sim_loader(encoded_sim_train, batch_size=sim_batch_size, num_workers=args.num_workers)
test_loader  = build_sim_loader(encoded_sim_test,  batch_size=sim_batch_size, num_workers=args.num_workers)

print("[INFO] encoding concept set...")
concept_tokens = tokenizer_sim(
    concept_set,
    padding=True,
    truncation=True,
    max_length=args.max_length,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    if args.concept_text_sim_model in ["mpnet", "simcse"]:
        concept_emb = sim_model(
            input_ids=concept_tokens["input_ids"],
            attention_mask=concept_tokens["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )
        if args.concept_text_sim_model == "mpnet":
            concept_emb = mean_pooling(concept_emb, concept_tokens["attention_mask"])
        else:
            concept_emb = concept_emb.pooler_output
    elif args.concept_text_sim_model == "angle":
        out = sim_model(
            output_hidden_states=True,
            input_ids=concept_tokens["input_ids"],
            attention_mask=concept_tokens["attention_mask"]
        )
        concept_emb = out.hidden_states[-1][:, -1].float()
    else:
        raise RuntimeError("unreachable")

    concept_emb = F.normalize(concept_emb, p=2, dim=1)  

def compute_similarity_matrix(loader):
    sims = []
    start_t = time.time()
    for i, batch_sim in enumerate(loader):
        print(f"[INFO] batch {i}", end="\r")

        batch_sim = {k: v.to(device) for k, v in batch_sim.items()}

        with torch.no_grad():
            if args.concept_text_sim_model in ["mpnet", "simcse"]:
                text_emb_out = sim_model(
                    input_ids=batch_sim["input_ids"],
                    attention_mask=batch_sim["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True
                )
                if args.concept_text_sim_model == "mpnet":
                    text_emb = mean_pooling(text_emb_out, batch_sim["attention_mask"])
                else:
                    text_emb = text_emb_out.pooler_output
            elif args.concept_text_sim_model == "angle":
                out = sim_model(
                    output_hidden_states=True,
                    input_ids=batch_sim["input_ids"],
                    attention_mask=batch_sim["attention_mask"]
                )
                text_emb = out.hidden_states[-1][:, -1].float()
            else:
                raise RuntimeError("unreachable")

            text_emb = F.normalize(text_emb, p=2, dim=1)

        # cosine sim = dot product of normalized vectors
        sims.append(text_emb @ concept_emb.T)

    sims = torch.cat(sims, dim=0).cpu().detach().numpy()
    end_t = time.time()
    print(f"\n[INFO] similarity computed in {(end_t - start_t)/3600:.4f} hours")
    return sims

print("[INFO] computing train similarity...")
train_similarity = compute_similarity_matrix(train_loader)

print("[INFO] computing test similarity...")
test_similarity = compute_similarity_matrix(test_loader)

out_dir = f"./{args.concept_text_sim_model}_acs/{args.dataset.replace('/', '_')}"
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "concept_labels_train.npy"), train_similarity)
np.save(os.path.join(out_dir, "concept_labels_test.npy"),  test_similarity)

print(f"[INFO] saved:")
print(f" - {os.path.join(out_dir, 'concept_labels_train.npy')} shape={train_similarity.shape}")
print(f" - {os.path.join(out_dir, 'concept_labels_test.npy')}  shape={test_similarity.shape}")
print("[INFO] done.")