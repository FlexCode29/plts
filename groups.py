import transformer_lens
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
palette = ['#FFC533', '#f48c06', '#DD5703', '#d00000', '#6A040F']
cmap = LinearSegmentedColormap.from_list("paper", palette)
from tqdm import tqdm
import os

os.environ["HF_TOKEN"] = "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"

import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = transformer_lens.HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
dataloader = DataLoader(dataset, batch_size=4)

def cache_hook(x, hook, cache):
    cache[hook.name].append(x[:, -1])
    return x

from functools import partial
from transformer_lens.utils import get_act_name


def compute_angular_distance(W1, W2):
    cosine_similarity = F.cosine_similarity(W1, W2, dim=-1)
    angular_distance = torch.arccos(cosine_similarity)
    return angular_distance / torch.pi


all_distances = []
max_tokens = 10_000 * 256
processed_tokens = 0
n_layers = model.cfg.n_layers
alpha = 0.05
avg_batch = None

with torch.no_grad():
    with tqdm(total=max_tokens) as pbar:
        for ex in dataloader:
            tokens = model.to_tokens(ex["text"])[:, :256]
            batch_size = tokens.size(0)
            cache = {get_act_name("resid_post", i): [] for i in range(n_layers)}
            hooks = [(get_act_name("resid_post", i), partial(cache_hook, cache=cache)) for i in range(n_layers)]

            _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

            cache = {k: torch.cat(v, dim=0) for k, v in cache.items()}
            activations_batch = torch.stack([cache[k] for k in cache], dim=0)

            # Center activations
            if avg_batch is None:
                avg_batch = activations_batch.mean(dim=-1, keepdim=True)
            else:
                avg_batch = avg_batch * (1 - alpha) + activations_batch.mean(dim=-1, keepdim=True) * alpha

            activations_batch = activations_batch - avg_batch

            distances_tensor = torch.zeros(batch_size, n_layers, n_layers)

            for i in range(n_layers):
                for j in range(i):
                    W1 = activations_batch[i]
                    W2 = activations_batch[j]
                    angular_distances = compute_angular_distance(W1, W2)
                    distances_tensor[:, i, j] = angular_distances

            mean_distances = distances_tensor.mean(dim=0)

            all_distances.append(mean_distances)
            processed_tokens += tokens.numel()
            pbar.update(tokens.numel())

            if processed_tokens >= max_tokens:
                break

np.save("distances/gemma_2_2b_256_40k.npy", torch.stack(all_distances).cpu().numpy())

dist = np.array(all_distances).mean(0) * 2

#dist = np.load('distances/pythia_160m_256_5M_dists.npy')

sns.set_context("paper")

fig, ax = plt.subplots(1, 1, figsize=(12, 11), dpi=150, layout="tight") #7, 6.5

cmap.set_bad("white")
dist_bad = np.copy(dist)
dist_bad[np.triu_indices_from(dist_bad, 1)] = np.nan
mask = np.zeros_like(dist, dtype=bool)
mask[np.tril_indices_from(mask)] = True
# for text, show_annot in zip(ax.texts, mask.ravel()):
#     text.set_visible(show_annot)mask = np.zeros_like(dist, dtype=bool)
mask[np.tril_indices_from(mask)] = True
# for text, show_annot in zip(ax.texts, mask.ravel()):
#     text.set_visible(show_annot)

sns.heatmap(
    dist_bad,
    cmap=cmap,
    vmin=0,
    annot=True,
    fmt=".2f",
    ax=ax,
    mask=np.triu(dist),
    square=False,
    linewidths=0.3,
    linecolor="white",
    cbar=False,
    annot_kws={'size': 10, "color": "white"}
)

ax.set_title("Average Angular Distance Between Layers", pad=8, fontsize=18)
# ax.set_ylim(0, 1)

ax.set_xlabel("Layer", labelpad=10, fontsize=16)
ax.set_ylabel("Layer", labelpad=10, fontsize=16)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

plt.tight_layout()
plt.savefig("img/distances.pdf", dpi=300, bbox_inches="tight")

from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

scores = []
nl = model.cfg.n_layers

for k in tqdm(range(2, nl - 1)):
    dist = np.array(all_distances).mean(0)

    X = np.array(dist)
    X[np.isinf(X)] = 0
    X[np.isnan(X)] = 0
    X = np.nan_to_num(X)

    clustering = AgglomerativeClustering(n_clusters=k, linkage="complete").fit(X)
    print(f"K{k}:\t", clustering.labels_)

