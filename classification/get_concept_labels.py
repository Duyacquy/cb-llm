import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import config as CFG
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dataset_utils import train_val_test_split
from peft import PeftModel, PeftConfig
from utils import mean_pooling, decorate_dataset, decorate_concepts
import sys
import time
from dataset_utils import train_val_test_split, preprocess

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--concept_text_sim_model", type=str, default="mpnet", help="mpnet, simcse or angle")

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=4)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parser.parse_args()

class SimDataset(torch.utils.data.Dataset):
    def __init__(self, encode_sim):
        self.encode_sim = encode_sim

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encode_sim.items()}
        return t

    def __len__(self):
        return len(self.encode_sim['input_ids'])

def build_sim_loaders(encode_sim):
    dataset = SimDataset(encode_sim)
    if args.concept_text_sim_model == 'angle':
        batch_size = 8
    else:
        batch_size = 256
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)
    return dataloader

print("--------------------------------GET_CONCEPT_LABEL--------------------------------")
print("loading data...")
train_dataset, test_dataset = train_val_test_split(args.dataset, CFG.dataset_config[args.dataset]["label_column"], ratio=0.2, has_val=False)

print(f"[INFO] train size = {len(train_dataset)}, test size = {len(test_dataset)}")

concept_set = CFG.concept_set[args.dataset]
print(f"[INFO] num concepts = {len(concept_set)}")

if args.concept_text_sim_model == 'mpnet':
    print("tokenizing and preparing mpnet")
    tokenizer_sim = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    sim_model.eval()
elif args.concept_text_sim_model == 'simcse':
    tokenizer_sim = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    sim_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
    sim_model.eval()
elif args.concept_text_sim_model == 'angle':
    print("tokenizing and preparing angle")
    config = PeftConfig.from_pretrained('SeanLee97/angle-llama-7b-nli-v2')
    tokenizer_sim = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    sim_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).bfloat16()
    sim_model = PeftModel.from_pretrained(sim_model, 'SeanLee97/angle-llama-7b-nli-v2')
    sim_model = sim_model.to(device)
    sim_model.eval()
    train_dataset = train_dataset.map(decorate_dataset, fn_kwargs={"d": args.dataset})
    test_dataset = test_dataset.map(decorate_dataset, fn_kwargs={"d": args.dataset})
    concept_set = decorate_concepts(concept_set)
else:
    raise Exception("concept-text sim model should be mpnet, simcse or angle")

train_dataset = preprocess(train_dataset, args.dataset, CFG.dataset_config[args.dataset]["text_column"], CFG.dataset_config[args.dataset]["label_column"])
test_dataset = preprocess(test_dataset, args.dataset, CFG.dataset_config[args.dataset]["text_column"], CFG.dataset_config[args.dataset]["label_column"])

encoded_sim_train_dataset = train_dataset.map(
    lambda e: tokenizer_sim(e[CFG.dataset_config[args.dataset]["text_column"]], padding=True, truncation=True, max_length=args.max_length),
    batched=True,
    batch_size=len(train_dataset)
)

encoded_sim_test_dataset = test_dataset.map(
    lambda e: tokenizer_sim(e[CFG.dataset_config[args.dataset]["text_column"]], padding=True, truncation=True, max_length=args.max_length),
    batched=True,
    batch_size=len(test_dataset)
)

encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns([CFG.dataset_config[args.dataset]["text_column"]])
encoded_sim_test_dataset = encoded_sim_test_dataset.remove_columns([CFG.dataset_config[args.dataset]["text_column"]])

encoded_sim_train_dataset = encoded_sim_train_dataset[:len(encoded_sim_train_dataset)]
encoded_sim_test_dataset = encoded_sim_test_dataset[:len(encoded_sim_test_dataset)]

encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=args.max_length)

train_sim_loader = build_sim_loaders(encoded_sim_train_dataset)
test_sim_loader = build_sim_loaders(encoded_sim_test_dataset)

print("getting concept labels...")
encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}
with torch.no_grad():
    if args.concept_text_sim_model == 'mpnet':
        concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"])
        concept_features = mean_pooling(concept_features, encoded_c["attention_mask"])
    elif args.concept_text_sim_model == 'simcse':
        concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
    elif args.concept_text_sim_model == 'angle':
        concept_features = sim_model(output_hidden_states=True, input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"]).hidden_states[-1][:, -1].float()
    else:
        raise Exception("concept-text sim model should be mpnet, simcse or angle")
    concept_features = F.normalize(concept_features, p=2, dim=1)

start = time.time()
print("getting concept scores for train set...")
train_sim = []
for i, batch_sim in enumerate(train_sim_loader):
    print("batch ", str(i), end="\r")
    batch_sim = {k: v.to(device) for k, v in batch_sim.items()}
    with torch.no_grad():
        if args.concept_text_sim_model == 'mpnet':
            text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"])
            text_features = mean_pooling(text_features, batch_sim["attention_mask"])
        elif args.concept_text_sim_model == 'simcse':
            text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
        elif args.concept_text_sim_model == 'angle':
            text_features = sim_model(output_hidden_states=True, input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"]).hidden_states[-1][:, -1].float()
        else:
            raise Exception("concept-text sim model should be mpnet, simcse or angle")
        text_features = F.normalize(text_features, p=2, dim=1)
    train_sim.append(text_features @ concept_features.T)
train_similarity = torch.cat(train_sim, dim=0).cpu().detach().numpy()
end = time.time()
print("time of concept scoring:", (end-start)/3600, "hours")

print("getting concept scores for test set...")
test_sim = []
start = time.time()
for i, batch_sim in enumerate(test_sim_loader):
    print("batch ", str(i), end="\r")
    batch_sim = {k: v.to(device) for k, v in batch_sim.items()}
    with torch.no_grad():
        if args.concept_text_sim_model == 'mpnet':
            text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"])
            text_features = mean_pooling(text_features, batch_sim["attention_mask"])
        elif args.concept_text_sim_model == 'simcse':
            text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
        elif args.concept_text_sim_model == 'angle':
            text_features = sim_model(output_hidden_states=True, input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"]).hidden_states[-1][:, -1].float()
        else:
            raise Exception("concept-text sim model should be mpnet, simcse or angle")
        text_features = F.normalize(text_features, p=2, dim=1)
    test_sim.append(text_features @ concept_features.T)
test_similarity = torch.cat(test_sim, dim=0).cpu().detach().numpy()
end = time.time()
print("time of concept scoring:", (end-start)/3600, "hours")

d_name = args.dataset.replace('/', '_')
prefix = "./"
if args.concept_text_sim_model == 'mpnet':
    prefix += "mpnet_acs"
elif args.concept_text_sim_model == 'simcse':
    prefix += "simcse_acs"
elif args.concept_text_sim_model == 'angle':
    prefix += "angle_acs"
prefix += "/"
prefix += d_name
prefix += "/"
if not os.path.exists(prefix):
    os.makedirs(prefix)

np.save(prefix + "concept_labels_train.npy", train_similarity)
np.save(prefix + "concept_labels_test.npy", test_similarity)