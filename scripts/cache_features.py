"""
Audio preprocessing pipeline: load audio, extract log-mel spectrograms,
and cache as .pt tensors.

GPU-accelerated: batches segments together and computes mel spectrograms
on GPU when available. Audio decoding remains on CPU.

Usage:
    python scripts/cache_features.py --config configs/default.yaml \
        --split-file data/splits/track_disjoint_train.json \
        --output-dir data/processed/fma_track
"""

import argparse
import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from tqdm import tqdm


def load_audio(file_path, target_sr=22050):
    """Load and preprocess audio file to mono at target sample rate.

    Uses librosa for all formats (handles MP3 via audioread, WAV via soundfile).
    Avoids torchaudio.load entirely to bypass torchcodec/FFmpeg issues on Windows.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    waveform = torch.from_numpy(y).float()
    return waveform, target_sr


def extract_segments(waveform, sr, segment_length=3.0, segment_hop=1.5):
    """Extract fixed-length segments from waveform."""
    seg_samples = int(segment_length * sr)
    hop_samples = int(segment_hop * sr)
    total_samples = waveform.shape[0]

    segments = []
    start = 0
    while start + seg_samples <= total_samples:
        segments.append(waveform[start:start + seg_samples])
        start += hop_samples

    if len(segments) == 0 and total_samples > 0:
        padded = torch.zeros(seg_samples)
        padded[:total_samples] = waveform
        segments.append(padded)

    return segments


def process_split(split_file, output_dir, config, device):
    """Process all tracks in a split file and cache log-mel tensors."""
    with open(split_file, "r") as f:
        records = json.load(f)

    audio_cfg = config["audio"]
    feat_cfg = config["features"]
    sr = audio_cfg["sample_rate"]
    batch_size = config.get("preprocessing_batch_size", 64)

    os.makedirs(output_dir, exist_ok=True)

    # Create mel transform ONCE on the target device
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=feat_cfg["n_fft"],
        hop_length=feat_cfg["hop_length"],
        n_mels=feat_cfg["n_mels"],
    ).to(device)

    log_scale = feat_cfg.get("log_scale", 10)
    stats = {"processed": 0, "skipped": 0, "segments_total": 0}

    # Accumulate segments into batches for GPU processing
    seg_batch = []       # list of (seg_samples,) tensors
    meta_batch = []      # list of (track_id, seg_idx, genre_idx) tuples

    def flush_batch():
        """Send accumulated segments through GPU mel transform."""
        if not seg_batch:
            return
        # Stack into (B, samples) and move to GPU
        batch_tensor = torch.stack(seg_batch).to(device)     # (B, samples)
        with torch.no_grad():
            mel_specs = mel_transform(batch_tensor)           # (B, n_mels, T)
            log_mels = torch.log1p(log_scale * mel_specs)     # (B, n_mels, T)

        # Move back to CPU and save individually
        log_mels = log_mels.cpu()
        for i, (track_id, seg_idx, genre_idx) in enumerate(meta_batch):
            out_path = os.path.join(output_dir, f"{track_id}_seg{seg_idx}.pt")
            torch.save({
                "log_mel": log_mels[i],
                "genre_idx": genre_idx,
                "track_id": track_id,
                "segment_idx": seg_idx,
            }, out_path)
            stats["segments_total"] += 1

        seg_batch.clear()
        meta_batch.clear()

    for record in tqdm(records, desc=f"Processing tracks ({device})"):
        file_path = record["file_path"]
        track_id = record["track_id"]

        if not os.path.exists(file_path):
            stats["skipped"] += 1
            continue

        try:
            waveform, _ = load_audio(file_path, target_sr=sr)
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            stats["skipped"] += 1
            continue

        # Normalize amplitude
        if audio_cfg.get("normalize", True):
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        # Extract segments
        segments = extract_segments(
            waveform, sr,
            segment_length=audio_cfg["segment_length"],
            segment_hop=audio_cfg.get("segment_hop", audio_cfg["segment_length"]),
        )

        # Add segments to batch
        for seg_idx, segment in enumerate(segments):
            seg_batch.append(segment)
            meta_batch.append((track_id, seg_idx, record["genre_idx"]))

            if len(seg_batch) >= batch_size:
                flush_batch()

        stats["processed"] += 1

    # Flush remaining segments
    flush_batch()

    print(f"\nProcessing complete:")
    print(f"  Device:           {device}")
    print(f"  Tracks processed: {stats['processed']}")
    print(f"  Tracks skipped:   {stats['skipped']}")
    print(f"  Total segments:   {stats['segments_total']}")


def main():
    parser = argparse.ArgumentParser(description="Cache log-mel features")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split-file", required=True,
                        help="Path to split JSON file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for cached .pt tensors")
    parser.add_argument("--device", default=None,
                        help="Device: cuda, cpu, or auto (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for GPU mel computation (default: 64)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config["preprocessing_batch_size"] = args.batch_size

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    process_split(args.split_file, args.output_dir, config, device)


if __name__ == "__main__":
    main()
