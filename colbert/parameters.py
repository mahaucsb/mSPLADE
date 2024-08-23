import torch

DEVICE = torch.device("cuda")

SAVED_CHECKPOINTS = [0,200, 400, 600, 800, 1000, 1200, 1400, 1500]
SAVED_CHECKPOINTS += [32*1000, 100*1000, 150*1000, 200*1000, 300*1000, 400*1000]
SAVED_CHECKPOINTS += [10*1000, 20*1000, 30*1000, 40*1000, 50*1000, 60*1000, 70*1000, 80*1000, 90*1000]
SAVED_CHECKPOINTS += [25*1000, 50*1000, 75*1000]
SAVED_CHECKPOINTS += [1000 * x for x in range(1,100)]
SAVED_CHECKPOINTS = set(SAVED_CHECKPOINTS)

LSH_DIM = 256

model_type = "static_product_dual" # static_product, static_additive, product, additive, static_product_dual, or no
before_norm = True
nlayer = 1