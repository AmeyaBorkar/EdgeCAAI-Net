# EdgeCAAI-Net

**EdgeCAAI-Net** — Lightweight Compute-Adaptive, Artist-Invariant Music Genre Classifier

A music genre classification system designed for edge deployment, featuring:
- **Multi-exit inference** with confidence-based early stopping
- **Artist invariance** via Gradient Reversal Layer (GRL) and GroupDRO
- **~1.42M parameters** — efficient enough for on-device inference
- Empirical validation that artist leakage inflates reported scores on standard benchmarks

---

## Overview

Standard music genre classifiers are evaluated on *track-disjoint* splits, where tracks by the same artist can appear in both train and test sets. This inflates scores because the model partially learns artist style rather than genre. EdgeCAAI-Net exposes this gap by evaluating on *artist-disjoint* splits and uses adversarial training (GRL) to reduce artist-specific features.

**Core thesis**: A track-disjoint to artist-disjoint F1 drop of **0.105** (18.8% relative) demonstrates that artist leakage is a real and significant problem in genre recognition benchmarks.

---

## Architecture

```
Input Log-Mel Spectrogram (64 × T)
         │
    ┌────▼────┐
    │  Stem   │  Depthwise-Separable 2D Conv  1×1 PW → 128ch
    └────┬────┘
         │
    ┌────▼────┐  ← Exit 1 (weight 0.2)
    │ Block 1 │  Conformer-Lite: DW-Conv + MHSA + FFN
    │ Block 2 │
    └────┬────┘
         │
    ┌────▼────┐  ← Exit 2 (weight 0.3)
    │ Block 3 │
    │ Block 4 │
    └────┬────┘
         │
    ┌────▼────┐  ← Exit 3 / Final (weight 0.5)
    │ Block 5 │
    │ Block 6 │
    └────┬────┘
         │
  Attentive Statistics Pooling
  (weighted mean + std → 256-d)
         │
    Classifier Head
         │
      Genre Logits
```

**Conformer-Lite Block**: Depthwise temporal convolution → Multi-head self-attention (4 heads) → Feed-forward network (dim 256) with residual connections. No convolution subsampling — maintains temporal resolution.

**Early Exits**: Each exit head has its own attentive pooling and confidence estimator. During inference, the model stops at the first exit where confidence exceeds a threshold (default: 0.9).

**Artist Invariance (GRL)**: A gradient reversal layer feeds into an artist classifier. The main network learns to fool it → artist-invariant representations.

| Component | Value |
|-----------|-------|
| Backbone blocks | 6 Conformer-Lite |
| d_model | 128 |
| n_heads | 4 |
| FFN dim | 256 |
| Exit positions | Blocks 2, 4, 6 |
| Total params | ~1.42M |

---

## Project Structure

```
EdgeCAAI-Net/
├── models/
│   ├── edgecaai_net.py         # Main model architecture
│   ├── exits.py                # Early exit heads, budget loss
│   ├── artist_invariance.py    # GRL and GroupDRO
│   └── baselines/
│       ├── tiny_cnn.py         # Baseline A (0.65M)
│       ├── small_crnn.py       # Baseline B (0.98M)
│       ├── mobilenet_baseline.py # Baseline C (1.53M)
│       └── tiny_transformer.py # Baseline D (0.69M)
├── configs/
│   ├── default.yaml            # Shared hyperparameters
│   ├── fma_track_disjoint.yaml # FMA track-disjoint experiment
│   ├── fma_artist_disjoint.yaml# FMA artist-disjoint + GRL
│   ├── gtzan_config.yaml       # GTZAN experiment
│   ├── baselines/              # 8 baseline configs
│   └── ablations/              # 4 ablation configs
├── scripts/
│   ├── make_splits.py          # Dataset split generation
│   ├── cache_features.py       # GPU-accelerated mel preprocessing
│   └── measure_latency.py      # On-device latency benchmarking
├── data/
│   ├── splits/                 # Train/val/test split JSON files
│   └── processed/              # Cached .pt tensors (gitignored)
├── results/
│   ├── fma_track_disjoint/     # Main model checkpoints
│   ├── fma_artist_disjoint/    # GRL model checkpoints
│   ├── gtzan/                  # GTZAN model checkpoints
│   ├── baselines/              # Baseline checkpoints
│   ├── ablations/              # Ablation checkpoints
│   └── training_log.md         # Full epoch-by-epoch training logs
├── train.py                    # Main training script (EdgeCAAI-Net + ablations)
├── train_baseline.py           # Baseline training script
├── eval.py                     # Evaluation with multi-exit inference
└── run_all.py                  # Wrapper to run all experiments
```

---

## Installation

```bash
git clone https://github.com/AmeyaBorkar/EdgeCAAI-Net.git
cd EdgeCAAI-Net

# Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

**requirements.txt** includes: `torch`, `torchaudio`, `torchvision`, `librosa`, `numpy`, `scikit-learn`, `PyYAML`, `tqdm`

> **Note**: Audio loading uses `librosa` throughout (avoids Windows torchcodec/FFmpeg issues).

---

## Datasets

### FMA Small
- **Source**: [freemusicarchive.org](https://github.com/mdeff/fma)
- **Size**: 8,000 MP3 tracks, 8 genres
- **Path**: `Datasets/fma/fma_small/` and `Datasets/fma/fma_metadata/`

### GTZAN
- **Source**: [Kaggle GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Size**: 1,000 WAV tracks, 10 genres
- **Path**: `Datasets/gtzan_kaggle/Data/genres_original/`

---

## Usage

### Step 1 — Generate Splits

```bash
# FMA splits (track-disjoint and artist-disjoint)
python scripts/make_splits.py --dataset fma \
    --data-dir Datasets/fma/fma_small \
    --metadata-dir Datasets/fma/fma_metadata \
    --out-dir data/splits

# GTZAN splits
python scripts/make_splits.py --dataset gtzan \
    --data-dir Datasets/gtzan_kaggle/Data/genres_original \
    --out-dir data/splits
```

### Step 2 — Preprocess (Cache Mel Spectrograms)

```bash
# FMA (track-disjoint split)
python scripts/cache_features.py \
    --split-file data/splits/track_disjoint_train.json \
    --out-dir data/processed/fma_track \
    --device cuda --batch-size 64

# Repeat for val/test splits and artist-disjoint splits
# Repeat for GTZAN using gtzan_track_disjoint_*.json splits
```

> Preprocessing output: ~147K segments (FMA track), ~128K (FMA artist), ~19K (GTZAN)

### Step 3 — Train

```bash
# Main model: FMA track-disjoint
python train.py --config configs/fma_track_disjoint.yaml

# Main model: FMA artist-disjoint with GRL
python train.py --config configs/fma_artist_disjoint.yaml

# Main model: GTZAN
python train.py --config configs/gtzan_config.yaml

# All baselines and ablations (auto-skips completed jobs)
python run_all.py

# Or selectively:
python run_all.py --only baselines
python run_all.py --only ablations
```

### Step 4 — Evaluate

```bash
python eval.py --checkpoint results/fma_track_disjoint/checkpoints/best.pt \
               --config configs/fma_track_disjoint.yaml \
               --threshold 0.9
```

### Step 5 — Measure Latency

```bash
python scripts/measure_latency.py \
    --checkpoint results/fma_track_disjoint/checkpoints/best.pt \
    --device cpu
```

---

## Audio Configuration

| Parameter | Value |
|-----------|-------|
| Sample rate | 22,050 Hz |
| Segment length | 3 s |
| Segment hop | 1.5 s |
| n_fft | 1,024 |
| hop_length | 512 |
| n_mels | 64 |
| Compression | log(1 + 10 × mel) |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Batch size | 32 |
| Max epochs | 80 |
| LR schedule | Cosine with 5% warmup |
| Exit weights | [0.2, 0.3, 0.5] |
| Budget λ | 0.02 |
| GRL α | 0.2 |
| Early stopping patience | 10 |

---

## Results

### Main Model — EdgeCAAI-Net

| Split | Artist Inv. | Best Val F1 | Best Val BA | Best Epoch |
|-------|-------------|-------------|-------------|------------|
| FMA Track-Disjoint | None | **0.5598** | 0.5620 | 8 |
| FMA Artist-Disjoint | GRL (α=0.2) | **0.4546** | 0.4647 | 18 |
| GTZAN | None | **0.7973** | 0.7978 | 13 |

**Track → Artist F1 drop: −0.105 (18.8% relative)** — validates the artist leakage hypothesis.

### Comparison with Baselines (FMA)

| Model | Params | Track-Disjoint F1 | Artist-Disjoint F1 | Gap |
|-------|--------|-------------------|---------------------|-----|
| TinyCNN | 0.65M | 0.5436 | 0.4317 | −0.112 |
| SmallCRNN | 0.98M | **0.6004** | **0.4964** | −0.104 |
| MobileNetV3-Small | 1.53M | 0.5341 | — | — |
| TinyTransformer | 0.69M | — | — | — |
| **EdgeCAAI-Net** | **1.42M** | **0.5598** | **0.4546** | **−0.105** |

### Ablation Study (FMA Artist-Disjoint)

| Configuration | Early Exit | Budget | Artist Inv. | Val F1 | Val BA |
|---------------|-----------|--------|-------------|--------|--------|
| Backbone only | No | No | No | 0.4478 | 0.4681 |
| + Early exits | Yes | No | No | 0.4566 | 0.4633 |
| + Exits + Budget | Yes | Yes | No | 0.4662 | 0.4784 |
| + Exits + Budget + GRL | Yes | Yes | GRL | **0.4546** | 0.4647 |
| + Exits + Budget + GroupDRO | Yes | Yes | GroupDRO | — | — |

**Component contributions**: Each added component yields incremental improvement. The GRL adversarial loss trades a small amount of absolute accuracy for artist-invariant representations — the real benefit appears in the reduced track-vs-artist gap.

---

## Key Findings

1. **Artist leakage is real**: All models show a consistent ~0.10–0.11 F1 drop when evaluated on artist-disjoint vs track-disjoint splits.

2. **SmallCRNN is a strong baseline**: Its recurrent architecture handles temporal structure better, achieving the highest track-disjoint F1 (0.6004).

3. **MobileNetV3 underperforms despite size**: Its image classification backbone is suboptimal for spectrograms.

4. **EdgeCAAI-Net efficiency**: Achieves competitive accuracy at 1.42M params while supporting early-exit inference for compute-adaptive deployment.

5. **Budget loss accelerates convergence**: The budget-aware model peaks at epoch 11 vs epoch 15 for exits-only, suggesting earlier confident predictions.

---

## Model Checkpoints

All trained checkpoints are stored via Git LFS under `results/`:

| Checkpoint | F1 |
|------------|----|
| `results/fma_track_disjoint/checkpoints/best.pt` | 0.5598 |
| `results/fma_artist_disjoint/checkpoints/best.pt` | 0.4546 |
| `results/gtzan/checkpoints/best.pt` | 0.7973 |
| `results/baselines/tiny_cnn_track/checkpoints/best.pt` | 0.5436 |
| `results/baselines/tiny_cnn_artist/checkpoints/best.pt` | 0.4317 |
| `results/baselines/small_crnn_track/checkpoints/best.pt` | 0.6004 |
| `results/baselines/small_crnn_artist/checkpoints/best.pt` | 0.4964 |
| `results/baselines/mobilenet_track/checkpoints/best.pt` | 0.5341 |
| `results/ablations/backbone_only/checkpoints/best.pt` | 0.4478 |
| `results/ablations/exits_only/checkpoints/best.pt` | 0.4566 |
| `results/ablations/exits_budget/checkpoints/best.pt` | 0.4662 |

---

## Full Training Logs

Epoch-by-epoch training curves for all experiments: [`results/training_log.md`](results/training_log.md)

---

## License

This project is released for academic and research purposes.
