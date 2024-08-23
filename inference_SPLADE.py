import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
from tqdm import tqdm
import gzip 
import json
import os

class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg
    
    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        if self.agg == "max":
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive
        else:
            return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)


agg = "max"
model_type_or_dir = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer

model = Splade(model_type_or_dir, agg=agg)
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = 100
file_per = 100000
i = 0
starting_i = 7400000 #500000

fo = None
with open("../msmarco/collection.tsv") as f:
    for line in tqdm(f):
        if i < starting_i:
            i += 1
            continue
        if i % file_per == 0:
            if fo is not None:
                fo.close()
                #break
            fo = gzip.open(os.path.join(out_dir, f"file_{i // file_per}.jsonl.gz"), "w")

        did, doc = line.strip().split("\t")     
        with torch.no_grad():
            doc_rep = model(**tokenizer(doc, return_tensors="pt", truncation=True).to('cuda')).squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        #print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
        outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
        fo.write(outline.encode('utf-8'))
        fo.flush()
        i += 1
