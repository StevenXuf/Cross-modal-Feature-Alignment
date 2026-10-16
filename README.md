# Cross-modal Feature Alignment

Code for **Multimodal Representation Alignment for Cross-modal Information Retrieval** by Fan Xu and Luis A. Leiva ([paper](https://arxiv.org/abs/2506.08774v2)). We compare embedding geometry, fixed similarity metrics, and learned fusion networks for image–text retrieval on IMDB, Flickr30K, and MS-COCO.

## Experimental pipeline

![Experimental pipeline](figures/pipeline.png)

**Figure 1.** Extract image and text embeddings, analyze their geometry, compare retrieval metrics, and train fusion networks. The revised paper extends the figure's MLP branch to both MLP and transformer architectures, using MSE or a custom contrastive loss.

## Models and data

| Encoders | `model_name` |
| --- | --- |
| CLIP ViT-B/32 | `clip` |
| BLIP base retrieval COCO | `blip_official_itm` |
| Meta-Transformer B/16 | `metatransformer` |
| ResNet-34 / BERT-small | `resnetbert` |
| ConViT-base / RoBERTa-base | `convitroberta` |
| ConvNeXt-small / XLNet-base-cased | `convnextxlnet` |

Dataset identifiers are `imdb`, `flickr30k`, and `coco_train`. Each image has one text match: the movie title for IMDB or the first caption for Flickr30K/MS-COCO. Training uses all IMDB pairs or up to 20K pairs from the other datasets, split 80%/10%/10% for training/validation/testing with seed 0.

## Setup

For training on existing embeddings:

```bash
python -m pip install torch matplotlib scikit-learn torchmetrics tqdm
```

Use a recent PyTorch version supporting `torch.load(..., mmap=True, weights_only=True)`. Data, embeddings, and checkpoints must be obtained separately.

Provide two row-aligned `[N, D]` tensors with matching dimensions, following this naming convention:

```text
<feature_root>/<MODEL_NAME>/<model_name>_caption_embed_<dataset>.pt
<feature_root>/<MODEL_NAME>/<model_name>_img_embed_<dataset>.pt
```

For example: `alignment_features/CLIP/clip_caption_embed_imdb.pt` and its corresponding image tensor.

To extract features, configure dataset paths in `load_data.py` and `feature_extractor.py`, install encoder dependencies, and enable the extractor's `get_features` call. Extraction requires CUDA/NCCL and `torchrun`. BLIP needs its external source; Meta-Transformer also needs Data2Seq, pretrained weights, and its disabled extraction branch enabled. These external sources are not populated automatically by cloning.

## Train and evaluate

**`train_main.py` trains both MLPs and transformer-based fusion networks** on fixed embeddings. Run from the repository root:

```bash
# MLP configuration used in the paper
python train_main.py clip imdb --path /path/to/alignment_features \
  --network new_mlp --loss_func contrastive --task txt2img

# Transformer fusion
python train_main.py clip imdb --path /path/to/alignment_features \
  --network transformer --loss_func contrastive --task img2txt
```

| Option | Choices |
| --- | --- |
| `--network` | `new_mlp`: 512/256 hidden units; `transformer`: 128 hidden dimensions, 4 heads; `mlp`: earlier 500/500 variant (default) |
| `--loss_func` | `mse` (default): regress cosine similarity; `contrastive`: matched scores toward +1, unmatched toward −1 |
| `--task` | `txt2img` (default) or `img2txt` |

Defaults: batch size 64, 100 epochs, Adam learning rate `5e-5`, and cosine annealing. Override with `--batch_size`, `--n_epochs`, and `--lr`; select a GPU with `--gpu_id`. CPU is used when CUDA is unavailable. Contrastive batches must contain at least two pairs.

Training reports **Recall@1/5/10** for the final model. Checkpoints, loss histories, and plots are saved under:

```text
<feature_root>/<MODEL_NAME>/model_checkpoints/<network>_<dataset>_<loss_func>_<task>/
```

Checkpoints are saved every five epochs and on early stopping. Re-running the same configuration clears its output directory.

## Baselines and analysis

`compute_distance.py` compares Euclidean, cosine, Manhattan, and Chi-square metrics on up to 1K pairs. Update its feature path before running:

```bash
python compute_distance.py clip imdb yes --GPU_id 0
```

Use `yes` for text-to-image retrieval or `no` for image-to-text. `plot_*.py` provides analysis figures; `hyperparam_tuning.py` provides Optuna tuning. Review their experiment-specific paths and dependencies before use.

Bash launchers in local `scripts/`, logs, metric outputs, archived experiments, and generated figures are ignored; `figures/pipeline.png` is retained.

## Citation

```bibtex
@misc{xu2026multimodal,
  title         = {Multimodal Representation Alignment for Cross-modal Information Retrieval},
  author        = {Fan Xu and Luis A. Leiva},
  year          = {2026},
  eprint        = {2506.08774},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {https://arxiv.org/abs/2506.08774v2}
}
```
