from llvm_ml.torch import BasicBlockDataset
import csv
import matplotlib.pyplot as plt
import numpy as np

with open("./data/tokens_x86.csv") as f:
    reader = csv.reader(f)
    vocab = [row[1] for row in reader]

banned_ids = []
dataset = BasicBlockDataset("./data/ryzen3600_v16.cbuf", vocab, masked=False, banned_ids=banned_ids, prefilter=True)

hist = [0 for _ in range(dataset.num_opcodes)]

for i in range(dataset.len()):
    bb, _, _, _ = dataset.get(i)

    for node in bb.x:
        hist[node] += 1

hist = np.array(hist, dtype=float)
hist_sum = np.sum(hist)
hist = hist / hist_sum
hist.tofile("data/class_weights.txt", sep="\n")