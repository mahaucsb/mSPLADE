import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
from tqdm import tqdm
import gzip 
import json
import os

print(f"\nFile:inference_SPLADE.py\n") #maha
print(f"CUDA available: {torch.cuda.is_available()}, Current device: {torch.cuda.current_device()}")

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

print(f"Model: {model_type_or_dir}")
print(f"Output directory: {out_dir}")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer

model = Splade(model_type_or_dir, agg=agg)
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = 100
file_per = 1000000  #Yingrui 100000  --> process 1M documents per file
batch_size = 32 # maha
i = 0
#starting_i = 7400000 #500000 #was uncomment maha
print(f"Starging processing from document {i}")
fo = None
docs = [] #maha
ids = [] #maha
with open("/content/msmarco/collection.tsv") as f:#was maha: with open("../msmarco/collection.tsv") as f:  
    for line in tqdm(f):
        # if i < starting_i:  #comment maha
        #     i += 1
        #     continue
        if i % file_per == 0:
           if fo is not None:
                fo.close()
                #break
            fo = gzip.open(os.path.join(out_dir, f"file_{i // file_per}.jsonl.gz"), "w")
            print(f"Created new output file: file_{i // file_per}.jsonl.gz")

        did, doc = line.strip().split("\t")    
        docs.append(doc) #maha

        # with torch.no_grad(): #why remove these two lines???
        #     doc_rep = model(**tokenizer(doc, return_tensors="pt", truncation=True).to('cuda')).squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        #print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:  #maha:comment
        # weights = doc_rep[col].cpu().tolist()
        # d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
        # outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
        # fo.write(outline.encode('utf-8'))
        # fo.flush()
        # i += 1
        #maha what follows all new!
        if len(docs) == batch_size:
            results = process_batch(docs, ids)
            fo.write(''.join(results).encode('utf-8'))
            docs = []
            ids = []
        
        if i % 100000 == 0:
            print(f"Processed {i} documents. Current document ID: {did}")
        
        i += 1

    # Process any remaining documents
    if docs:
        results = process_batch(docs, ids)
        fo.write(''.join(results).encode('utf-8'))

if fo is not None:
    fo.close()

print(f"Processing completed. Total documents processed: {i}")
print(f"Final output directory contents:")
os.system(f"ls -l {out_dir}")
