import argparse
import json
import logging
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# Set HuggingFace cache to local directory if not already set
if "HF_DATASETS_CACHE" not in os.environ:
    local_cache = Path.cwd() / ".hf_cache"
    os.environ["HF_DATASETS_CACHE"] = str(local_cache)
    local_cache.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SR = 16000  # NeMo requires 16 kHz for this model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NeMo ASR manifest from Hugging Face Quran dataset. "
        "Audio is resampled to 16 kHz, validated, and output as JSONL manifests."
    )
    parser.add_argument("--dataset_name", default="hifzyml/quran_dataset_v0", help="HF dataset name")
    parser.add_argument("--output_dir", default="data", help="Output directory for manifests and audio")
    parser.add_argument("--audio_column", default="audio", help="Audio column name in dataset")
    parser.add_argument("--text_column", default="transcript", help="Text/transcript column name in dataset")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Test split ratio")
    parser.add_argument("--copy_audio", action="store_true", help="Copy/resample audio into output_dir/audio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit to N samples (for testing)")
    return parser.parse_args()


def ensure_dirs(base_dir: Path) -> dict:
    """Create output directories."""
    dirs = {
        "audio": base_dir / "audio",
        "manifests": base_dir / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_splits(dataset, val_ratio: float, test_ratio: float, seed: int, num_samples: int | None) -> dict:
    """Split dataset into train/val/test."""
    # Load all splits if available, else use 'train' and split it
    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split.")

    train_data = dataset["train"]
    if num_samples:
        train_data = train_data.select(range(min(num_samples, len(train_data))))

    # If separate val/test exist, use them
    if "validation" in dataset and "test" in dataset:
        return {
            "train": train_data,
            "val": dataset["validation"],
            "test": dataset["test"],
        }

    # Otherwise split train into train/val/test
    test_split = train_data.train_test_split(test_size=test_ratio, seed=seed)
    train_data = test_split["train"]
    test_data = test_split["test"]

    val_split = train_data.train_test_split(test_size=val_ratio / (1 - test_ratio), seed=seed)
    return {
        "train": val_split["train"],
        "val": val_split["test"],
        "test": test_data,
    }


def resample_audio(audio_data: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Resample audio to target sample rate."""
    if sr == target_sr:
        return audio_data, sr
    return librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr), target_sr


def prepare_split(
    split_name: str,
    dataset,
    output_dir: Path,
    audio_column: str,
    text_column: str,
    copy_audio: bool,
) -> set:
    """Process a single split (train/val/test) and write NeMo manifest."""
    manifest_path = output_dir / "manifests" / f"{split_name}.json"
    audio_dir = output_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    vocab = set()
    valid_count = 0
    skipped_count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            text = str(example[text_column]).strip()

            # Skip empty text
            if not text:
                skipped_count += 1
                continue

            # Extract characters for vocabulary
            for char in text:
                vocab.add(char)

            # Get audio
            audio_info = example[audio_column]
            audio_path = None
            audio_array = None
            sr = None

            try:
                # Handle AudioDecoder from torchcodec (datasets library)
                if hasattr(audio_info, 'get_all_samples'):
                    # It's an AudioDecoder - decode it
                    samples = audio_info.get_all_samples()
                    metadata = audio_info.metadata
                    
                    # Extract tensor and convert to numpy
                    tensor_data = samples.data
                    audio_array = tensor_data.cpu().numpy() if hasattr(tensor_data, 'cpu') else np.array(tensor_data)
                    
                    # Squeeze mono (shape [1, N] -> [N])
                    if audio_array.ndim == 2 and audio_array.shape[0] == 1:
                        audio_array = audio_array.squeeze(0)
                    elif audio_array.ndim == 2:
                        # Multi-channel, take first channel
                        audio_array = audio_array[0]
                    
                    # Ensure float32
                    audio_array = audio_array.astype(np.float32)
                    sr = metadata.sample_rate
                    
                elif isinstance(audio_info, dict) and "array" in audio_info:
                    # Dictionary format with pre-decoded array
                    audio_array = np.array(audio_info["array"], dtype=np.float32)
                    sr = audio_info.get("sampling_rate", TARGET_SR)
                    
                elif isinstance(audio_info, dict) and "path" in audio_info:
                    # Dictionary format with file path
                    audio_array, sr = sf.read(audio_info["path"])
                    audio_array = audio_array.astype(np.float32)
                    
                else:
                    logger.warning(f"Skipping sample {idx}: unsupported audio format {type(audio_info)}")
                    skipped_count += 1
                    continue

                if audio_array is None:
                    logger.warning(f"Skipping sample {idx}: failed to load audio")
                    skipped_count += 1
                    continue

                # Resample if needed
                if sr != TARGET_SR:
                    audio_array, sr = resample_audio(audio_array, sr, TARGET_SR)

                # Save audio
                if copy_audio:
                    audio_path = audio_dir / f"{idx:06d}.wav"
                    sf.write(str(audio_path), audio_array, TARGET_SR)
                else:
                    # Use original path if not copying
                    if isinstance(audio_info, dict) and "path" in audio_info:
                        audio_path = Path(audio_info["path"])
                    else:
                        audio_path = audio_dir / f"{idx:06d}.wav"
                        sf.write(str(audio_path), audio_array, TARGET_SR)

                # Compute duration
                duration = len(audio_array) / TARGET_SR

                # Write manifest record
                record = {
                    "audio_filepath": str(audio_path),
                    "duration": float(duration),
                    "text": text,
                }
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                valid_count += 1

            except Exception as e:
                logger.warning(f"Skipping sample {idx}: {e}")
                skipped_count += 1
                continue

    logger.info(f"{split_name}: {valid_count} valid, {skipped_count} skipped")
    return vocab


def save_vocab(vocab: set, vocab_path: Path) -> None:
    """Save vocabulary to file."""
    with vocab_path.open("w", encoding="utf-8") as f:
        for char in sorted(vocab):
            if char.strip():  # Skip pure whitespace
                f.write(char + "\n")
    logger.info(f"Vocabulary ({len(vocab)} chars) saved to {vocab_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)

    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    # Resolve splits
    splits = resolve_splits(dataset, args.val_ratio, args.test_ratio, args.seed, args.num_samples)
    logger.info(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Process each split
    vocab = set()
    for split_name, split_data in splits.items():
        split_vocab = prepare_split(
            split_name,
            split_data,
            output_dir,
            args.audio_column,
            args.text_column,
            args.copy_audio,
        )
        vocab.update(split_vocab)

    # Save vocab
    vocab_path = output_dir / "vocab.txt"
    save_vocab(vocab, vocab_path)

    logger.info(f"✓ All manifests saved to {output_dir / 'manifests'}")
    logger.info(f"✓ All audio saved to {output_dir / 'audio'}" if args.copy_audio else "✓ No audio copied (use --copy_audio to copy)")


if __name__ == "__main__":
    main()
