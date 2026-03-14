# NeMo Experiments Folder Navigation Algorithm

## Problem Statement

The `nemo_experiments` folder contains multiple model architectures, each with multiple training runs, and each run potentially has multiple checkpoints. We need to:

1. **Identify all architectures** (first-level directories)
2. **Find the latest run** for each architecture (newest timestamp)
3. **Select the best checkpoint** from that run (latest saved model)
4. **Detect model type** (streaming vs standard)
5. **Extract configuration** (decoder type, context size, etc.)

## Folder Structure

Two possible layouts:

### Layout 1: Flat (Most common)
```
nemo_experiments/
└── FastConformer-Custom-Tokenizer/
    ├── 2026-02-14_08-36-37/        ← LATEST RUN
    │   ├── checkpoints/
    │   │   └── FastConformer-Custom-Tokenizer.nemo  ← BEST CHECKPOINT
    │   ├── hparams.yaml
    │   └── nemo_log_globalrank-0_localrank-0.txt
    ├── 2026-02-14_08-19-54/        ← OLDER RUN
    │   └── ...
    └── ...
```

### Layout 2: Nested (Special case)
```
nemo_experiments/
└── FastConformer-Streaming-Custom-Tokenizer-Official-Seed/
    ├── finetune/
    │   ├── run_0/
    │   └── run_1/
    └── finetune_streaming_quran/
        └── 2026-02-16_12-24-08/    ← LATEST RUN
            ├── checkpoints/
            │   └── finetune_streaming_quran.nemo  ← BEST CHECKPOINT
            └── hparams.yaml
```

## Algorithm

### Step 1: Parse Timestamp

```python
def parse_timestamp(name: str) -> datetime:
    """Convert folder name to datetime.
    
    Format: 2026-02-17_05-27-32
    
    This allows sorting runs by date (latest first).
    """
    try:
        return datetime.strptime(name, '%Y-%m-%d_%H-%M-%S')
    except:
        return datetime.min  # Invalid timestamps sort to front
```

### Step 2: Find Latest Run (Per Architecture)

```python
def find_latest_run(arch_path: Path) -> Optional[Path]:
    """Find the most recent run folder for an architecture.
    
    Process:
    1. Scan immediate subdirectories
    2. Find those matching timestamp pattern (202*)
    3. Also scan nested directories (e.g., finetune/2026-*)
    4. Compare all timestamps
    5. Return maximum (newest)
    
    Handles:
    - arch_path/2026-02-14_08-36-37/  (direct)
    - arch_path/finetune_streaming_quran/2026-02-16_12-24-08/  (nested)
    """
    runs = []
    
    # Direct timestamped folders
    for item in arch_path.iterdir():
        if item.is_dir() and item.name[0].isdigit():  # 202* format
            runs.append(item)
    
    # Nested timestamped folders  (experiment_name/2026-*)
    for item in arch_path.iterdir():
        if item.is_dir() and not item.name[0].isdigit():
            for subitem in item.iterdir():
                if subitem.is_dir() and subitem.name[0].isdigit():
                    runs.append(subitem)
    
    if not runs:
        return None
    
    # Max by timestamp = newest
    return max(runs, key=lambda x: parse_timestamp(x.name))
```

**Why this works:**
- `item.name[0].isdigit()` identifies timestamps (start with '2')
- Nested loop finds subfolders like `finetune/2026-02-16_12-24-08`
- `max(..., key=parse_timestamp)` selects newest by date

### Step 3: Find Best Checkpoint

```python
def find_best_checkpoint(run_path: Path) -> Optional[Path]:
    """Find the best (latest saved) model checkpoint in a run.
    
    Process:
    1. Look in run_path/checkpoints/
    2. Find all *.nemo files
    3. Sort by modification time (latest first)
    4. Return first (best/latest)
    """
    checkpoints = run_path / "checkpoints"
    
    if not checkpoints.exists():
        return None
    
    nemo_files = list(checkpoints.glob("*.nemo"))
    if not nemo_files:
        return None
    
    # Latest saved = best model (by modification time)
    nemo_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return nemo_files[0]
```

**Why modification time:**
- Models saved during training have increasing modification times
- Latest = most recent epoch trained
- Assumption: Later epochs = better performance (or explicitly selected)

### Step 4: Detect Streaming

```python
def check_streaming(run_path: Path, arch_name: str) -> bool:
    """Detect if model is streaming or standard.
    
    Methods:
    1. Check hparams.yaml for att_context_size or "streaming"
    2. Fall back to architecture name check
    """
    hparams_path = run_path / "hparams.yaml"
    if not hparams_path.exists():
        hparams_path = run_path.parent / "hparams.yaml"
    
    if hparams_path.exists():
        try:
            with open(hparams_path) as f:
                config = yaml.safe_load(f)
            
            config_str = json.dumps(config, default=str).lower()
            if "streaming" in config_str or "att_context_size" in config_str:
                return True
        except:
            pass
    
    # Fallback: name-based detection
    return "streaming" in arch_name.lower()
```

**Config Detection:**
- YAML config contains model architecture details
- `att_context_size` = streaming attention cache size
- Keyword "streaming" appears in config if enabled

**Fallback:**
- "streaming" in folder name → streaming model
- Catches cases where hparams.yaml missing or unreadable

### Step 5: Get All Architectures

```python
def get_model_architectures() -> Dict[str, Dict]:
    """Enumerate all architectures with their best runs."""
    nemo_root = Path("nemo_experiments")
    architectures = {}
    
    for arch_path in sorted(nemo_root.iterdir()):
        if not arch_path.is_dir():
            continue
        
        arch_name = arch_path.name
        latest_run = find_latest_run(arch_path)
        
        if latest_run is None:  # No runs found
            continue
        
        best_ckpt = find_best_checkpoint(latest_run)
        
        if best_ckpt is None:  # No checkpoints saved
            continue
        
        is_streaming = check_streaming(latest_run, arch_name)
        
        architectures[arch_name] = {
            'latest_run': latest_run.name,
            'latest_run_path': str(latest_run),
            'best_ckpt': str(best_ckpt),
            'best_ckpt_name': best_ckpt.name,
            'is_streaming': is_streaming,
            'size_mb': best_ckpt.stat().st_size / (1024**2)
        }
    
    return architectures
```

## Example Walkthrough

### Architecture: FastConformer-Streaming-Custom-Tokenizer

```
Step 1: Parse timestamps
  nemo_experiments/FastConformer-Streaming-Custom-Tokenizer/
  ├─ 2026-02-15_13-35-12  →  2026-02-15 13:35:12
  ├─ 2026-02-14_20-30-54  →  2026-02-14 20:30:54
  └─ 2026-02-14_20-15-48  →  2026-02-14 20:15:48

Step 2: Find latest
  max(timestamps) = 2026-02-15 13:35:12
  Latest run: /path/FastConformer-Streaming-Custom-Tokenizer/2026-02-15_13-35-12

Step 3: Find best checkpoint
  ls 2026-02-15_13-35-12/checkpoints/
  FastConformer-Streaming-Custom-Tokenizer.nemo (modified 2026-02-15 13:35:12)
  
  Best: /path/.../FastConformer-Streaming-Custom-Tokenizer.nemo (438.5 MB)

Step 4: Detect streaming
  Check hparams.yaml:
    ✓ Found "att_context_size" in config
    → is_streaming = True
  
  Fallback check:
    ✓ "streaming" in "FastConformer-Streaming-Custom-Tokenizer"
    → is_streaming = True

Result:
  {
    'latest_run': '2026-02-15_13-35-12',
    'best_ckpt': '/path/...FastConformer-Streaming-Custom-Tokenizer.nemo',
    'best_ckpt_name': 'FastConformer-Streaming-Custom-Tokenizer.nemo',
    'is_streaming': True,
    'size_mb': 438.5
  }
```

### Architecture: FastConformer-Streaming-Custom-Tokenizer-Official-Seed (Nested)

```
Step 1: Parse timestamps (includes nested search)
  nemo_experiments/FastConformer-Streaming-Custom-Tokenizer-Official-Seed/
  ├─ finetune/
  │  ├─ run_0/
  │  └─ run_1/
  └─ finetune_streaming_quran/
     ├─ 2026-02-16_11-51-07  →  2026-02-16 11:51:07
     ├─ 2026-02-16_11-58-22  →  2026-02-16 11:58:22
     ├─ 2026-02-16_12-08-08  →  2026-02-16 12:08:08
     ├─ 2026-02-16_12-14-42  →  2026-02-16 12:14:42
     └─ 2026-02-16_12-24-08  →  2026-02-16 12:24:08

Step 2: Find latest
  max(timestamps) = 2026-02-16 12:24:08
  Latest run: /path/.../finetune_streaming_quran/2026-02-16_12-24-08

Step 3: Find best checkpoint
  ls 2026-02-16_12-24-08/checkpoints/
  finetune_streaming_quran.nemo
  
  Best: /path/.../finetune_streaming_quran.nemo (438.4 MB)

Step 4: Detect streaming
  Check hparams.yaml:
    ✓ Found "att_context_size" in config
    → is_streaming = True

Result:
  {
    'latest_run': '2026-02-16_12-24-08',
    'best_ckpt': '/path/.../finetune_streaming_quran/2026-02-16_12-24-08/checkpoints/finetune_streaming_quran.nemo',
    'best_ckpt_name': 'finetune_streaming_quran.nemo',
    'is_streaming': True,
    'size_mb': 438.4
  }
```

## Edge Cases & Error Handling

### Case 1: No runs found
```python
latest_run = find_latest_run(arch_path)
if latest_run is None:
    continue  # Skip this architecture
```
→ Occurs when folder exists but has no timestamp-named subdirectories

### Case 2: No checkpoints saved
```python
best_ckpt = find_best_checkpoint(latest_run)
if best_ckpt is None:
    continue  # Skip this architecture
```
→ Occurs for recent/incomplete experiments
→ Examples: CTC-BPE-Streaming-Quran, English-Quran-Tokenizer

### Case 3: hparams.yaml missing
```python
if not hparams_path.exists():
    hparams_path = run_path.parent / "hparams.yaml"
```
→ Check parent directory (sometimes config saved there)

### Case 4: Invalid timestamp format
```python
def parse_timestamp(name: str) -> datetime:
    try:
        return datetime.strptime(name, '%Y-%m-%d_%H-%M-%S')
    except:
        return datetime.min  # Invalid sorts to front
```
→ Returns minimum datetime, won't be selected as latest

## Time Complexity

- **Total time**: O(n * m)
  - n = number of architectures (~7)
  - m = average runs per architecture (~5)
  - Total filesystem operations: ~70 (negligible)

## Advantages of This Approach

1. **Automatic**: No hardcoded paths
2. **Scalable**: Works with any number of runs/architectures
3. **Flexible**: Handles both flat and nested layouts
4. **Deterministic**: Timestamp-based (reproducible)
5. **Robust**: Fallback detection methods
6. **Informative**: Returns full metadata (size, type, path)
7. **User-Friendly**: Interactive selection menu

## Files Implementing This

- `evaluate.py`: Main evaluation script with all functions
- `MODELS_OVERVIEW.md`: Documentation of discovered models

## Testing the Logic

```bash
python /tmp/detailed_analysis.py

# Shows all architectures found, latest runs, best checkpoints, and full paths
```

## Result: Discovered Models

✅ **4 Ready for Evaluation:**
1. FastConformer-Custom-Tokenizer (STANDARD)
2. FastConformer-Hybrid-Transducer-CTC-BPE-Streaming (STREAMING)
3. FastConformer-Streaming-Custom-Tokenizer (STREAMING)
4. FastConformer-Streaming-Custom-Tokenizer-Official-Seed (STREAMING - has issues)

⚠️ **3 Incomplete:**
1. FastConformer-CTC-BPE-Streaming-Quran
2. FastConformer-English-Quran-Tokenizer
3. FastConformer-Streaming-From-Pretrained
