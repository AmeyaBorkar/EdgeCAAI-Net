"""
Split generation script for FMA and GTZAN datasets.

Generates track-disjoint and artist-disjoint splits with leakage verification.
Outputs JSON files with track ID, artist ID, genre label, and file path.

Usage:
    python scripts/make_splits.py --dataset fma --split-type track_disjoint
    python scripts/make_splits.py --dataset fma --split-type artist_disjoint
    python scripts/make_splits.py --dataset gtzan --split-type track_disjoint
"""

import argparse
import ast
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_fma_metadata(metadata_dir):
    """Load FMA tracks and genres metadata."""
    tracks_path = os.path.join(metadata_dir, "tracks.csv")
    genres_path = os.path.join(metadata_dir, "genres.csv")

    # Load tracks CSV (multi-level header)
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # Extract relevant columns
    track_info = pd.DataFrame({
        "track_id": tracks.index,
        "artist_id": tracks[("artist", "id")].values,
        "genre_top": tracks[("track", "genre_top")].values,
        "subset": tracks[("set", "subset")].values,
    })

    # Filter to 'small' subset and drop tracks without genre
    track_info = track_info[track_info["subset"] == "small"]
    track_info = track_info.dropna(subset=["genre_top"])
    track_info = track_info.reset_index(drop=True)

    return track_info


def get_fma_audio_path(track_id, audio_dir):
    """Get the audio file path for an FMA track ID."""
    tid_str = str(track_id).zfill(6)
    folder = tid_str[:3]
    return os.path.join(audio_dir, folder, f"{tid_str}.mp3")


def load_gtzan_metadata(audio_dir):
    """Load GTZAN metadata from directory structure."""
    records = []
    for genre_dir in sorted(Path(audio_dir).iterdir()):
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name
        for audio_file in sorted(genre_dir.glob("*.wav")):
            track_id = audio_file.stem
            records.append({
                "track_id": track_id,
                "artist_id": -1,  # GTZAN has no artist metadata
                "genre_top": genre,
                "file_path": str(audio_file),
            })
    return pd.DataFrame(records)


def make_track_disjoint_split(df, seed=42, train_ratio=0.7, val_ratio=0.15):
    """
    Create track-disjoint split. No track appears in multiple partitions.
    Stratified by genre for class balance.
    """
    # Encode genre labels
    genres = sorted(df["genre_top"].unique())
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    labels = df["genre_top"].map(genre_to_idx).values

    # Train/temp split
    train_idx, temp_idx = train_test_split(
        np.arange(len(df)), test_size=(1 - train_ratio),
        stratify=labels, random_state=seed
    )
    # Val/test split from temp
    temp_labels = labels[temp_idx]
    relative_val = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - relative_val),
        stratify=temp_labels, random_state=seed
    )

    return train_idx, val_idx, test_idx, genres


def make_artist_disjoint_split(df, seed=42, train_ratio=0.7, val_ratio=0.15):
    """
    Create artist-disjoint split. All tracks by the same artist go to one partition.
    Groups by artist, then splits artist groups.
    """
    rng = np.random.RandomState(seed)
    genres = sorted(df["genre_top"].unique())

    # Group tracks by artist
    artist_groups = df.groupby("artist_id").agg({
        "genre_top": "first",  # majority genre for stratification
        "track_id": "count",
    }).rename(columns={"track_id": "num_tracks"})

    # Shuffle artists
    artists = artist_groups.index.values.copy()
    rng.shuffle(artists)

    # Split artists (approximate stratification by genre)
    n_artists = len(artists)
    n_train = int(n_artists * train_ratio)
    n_val = int(n_artists * val_ratio)

    train_artists = set(artists[:n_train])
    val_artists = set(artists[n_train:n_train + n_val])
    test_artists = set(artists[n_train + n_val:])

    # Map back to track indices
    train_idx = df[df["artist_id"].isin(train_artists)].index.values
    val_idx = df[df["artist_id"].isin(val_artists)].index.values
    test_idx = df[df["artist_id"].isin(test_artists)].index.values

    return train_idx, val_idx, test_idx, genres


def verify_no_leakage(df, train_idx, val_idx, test_idx, split_type="track"):
    """Verify no track or artist leakage across splits."""
    train_tracks = set(df.iloc[train_idx]["track_id"])
    val_tracks = set(df.iloc[val_idx]["track_id"])
    test_tracks = set(df.iloc[test_idx]["track_id"])

    # Track leakage check
    assert train_tracks.isdisjoint(val_tracks), "Track leak: train/val overlap!"
    assert train_tracks.isdisjoint(test_tracks), "Track leak: train/test overlap!"
    assert val_tracks.isdisjoint(test_tracks), "Track leak: val/test overlap!"

    if split_type == "artist_disjoint":
        train_artists = set(df.iloc[train_idx]["artist_id"])
        val_artists = set(df.iloc[val_idx]["artist_id"])
        test_artists = set(df.iloc[test_idx]["artist_id"])

        assert train_artists.isdisjoint(val_artists), "Artist leak: train/val!"
        assert train_artists.isdisjoint(test_artists), "Artist leak: train/test!"
        assert val_artists.isdisjoint(test_artists), "Artist leak: val/test!"

    print(f"  Leakage check PASSED ({split_type})")


def build_split_records(df, indices, genres, audio_dir=None, dataset="fma"):
    """Build JSON-serializable records for a split."""
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    records = []
    for idx in indices:
        row = df.iloc[idx]
        if dataset == "fma" and audio_dir:
            file_path = get_fma_audio_path(row["track_id"], audio_dir)
        else:
            file_path = row.get("file_path", "")

        records.append({
            "track_id": int(row["track_id"]) if dataset == "fma" else str(row["track_id"]),
            "artist_id": int(row["artist_id"]),
            "genre": row["genre_top"],
            "genre_idx": genre_to_idx[row["genre_top"]],
            "file_path": str(file_path),
        })
    return records


def save_split(records, output_path):
    """Save split to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset splits")
    parser.add_argument("--dataset", choices=["fma", "gtzan"], required=True)
    parser.add_argument("--split-type", choices=["track_disjoint", "artist_disjoint"],
                        required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/splits")
    # FMA paths
    parser.add_argument("--fma-audio-dir", default="Datasets/fma/fma_small")
    parser.add_argument("--fma-metadata-dir", default="Datasets/fma/fma_metadata")
    # GTZAN paths
    parser.add_argument("--gtzan-audio-dir",
                        default="Datasets/gtzan_kaggle/Data/genres_original")
    args = parser.parse_args()

    print(f"Generating {args.split_type} splits for {args.dataset}...")

    if args.dataset == "fma":
        df = load_fma_metadata(args.fma_metadata_dir)
        audio_dir = args.fma_audio_dir
        print(f"  Loaded {len(df)} FMA tracks")
    else:
        df = load_gtzan_metadata(args.gtzan_audio_dir)
        audio_dir = args.gtzan_audio_dir
        print(f"  Loaded {len(df)} GTZAN tracks")

    # Generate splits
    if args.split_type == "track_disjoint":
        train_idx, val_idx, test_idx, genres = make_track_disjoint_split(
            df, seed=args.seed
        )
    else:
        if args.dataset == "gtzan":
            print("  WARNING: GTZAN has no artist metadata. Using track-disjoint.")
            train_idx, val_idx, test_idx, genres = make_track_disjoint_split(
                df, seed=args.seed
            )
        else:
            train_idx, val_idx, test_idx, genres = make_artist_disjoint_split(
                df, seed=args.seed
            )

    # Verify no leakage
    verify_no_leakage(df, train_idx, val_idx, test_idx, args.split_type)

    # Print split statistics
    print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, "
          f"test={len(test_idx)}")
    print(f"  Genres ({len(genres)}): {genres}")

    # Build and save
    prefix = f"{args.dataset}_{args.split_type}" if args.dataset == "gtzan" else args.split_type
    for split_name, indices in [("train", train_idx), ("val", val_idx),
                                 ("test", test_idx)]:
        records = build_split_records(df, indices, genres, audio_dir, args.dataset)
        save_split(records, os.path.join(args.output_dir, f"{prefix}_{split_name}.json"))

    # Save genre mapping
    genre_map = {g: i for i, g in enumerate(genres)}
    genre_map_path = os.path.join(args.output_dir, f"{args.dataset}_genre_map.json")
    with open(genre_map_path, "w") as f:
        json.dump(genre_map, f, indent=2)
    print(f"  Genre mapping saved to {genre_map_path}")

    print("Done!")


if __name__ == "__main__":
    main()
