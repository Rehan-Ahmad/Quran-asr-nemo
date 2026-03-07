# Why Auto-Conversion to .nemo Didn't Happen During Finetuning

## Problem Analysis

The finetuning script (`finetune_multilingual_simple.py`) completes training but **does not automatically convert the final checkpoint to a deployable `.nemo` package**. 

### Root Cause

The script uses PyTorch Lightning's trainer which saves checkpoints as PyTorch Lightning `.ckpt` files (which are just pickled Python objects). NeMo doesn't automatically convert these to `.nemo` format. Here's why:

1. **`.ckpt` format** (PyTorch Lightning checkpoint):
   - Contains: `state_dict`, `hyper_parameters`, training metadata
   - Used for: Resuming training
   - Size: Larger (includes optimizer state, training state)
   - Backward compatible: Can load old checkpoints with different NeMo versions

2. **`.nemo` format** (NeMo package):
   - Contains: Model weights, config, tokenizer
   - Used for: Inference/deployment
   - Size: Smaller (weights + config only)
   - Auto-loading: No manual config required

### Why NeMo Doesn't Auto-Convert

**The design philosophy:**
- Training typically produces many checkpoints (one per epoch)
- Converting all of them to `.nemo` would waste disk space and time
- Practitioners should manually convert only the **best checkpoint** (usually the one with lowest validation loss)
- This gives explicit control over which checkpoint to promote to production

## Solution: Add Conversion to Finetuning Script

### Step 1: Update `finetune_multilingual_simple.py` 

Add automatic conversion after training:

```python
# [7] Start training
print("\n[7/7] Starting training...")
trainer.fit(model)

# [8] Convert best checkpoint to .nemo format
print("\n[8/8] Converting best checkpoint to .nemo format...")
best_checkpoint = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None

if best_checkpoint:
    nemo_output = best_checkpoint.replace('.ckpt', '.nemo')
    print(f"  Best checkpoint: {best_checkpoint}")
    print(f"  Converting to: {nemo_output}")
    
    model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(best_checkpoint, map_location='cpu')
    model.save_to(nemo_output)
    print(f"✓ Conversion complete: {nemo_output}")
```

### Step 2: Use Dedicated Conversion Script

For manual conversions (or converting old checkpoints), use the provided script:

```bash
python convert_ckpt_to_nemo.py \
    --ckpt ./nemo_experiments/.../epoch=49-step=45200.ckpt \
    --output ./nemo_experiments/.../FastConformer-Custom.nemo \
    --device cuda
```

## Best Practices Going Forward

1. **During training**: Train normally, let NeMo save `.ckpt` files
2. **After training**: Convert the best checkpoint to `.nemo` using the conversion script
3. **Deployment**: Ship only the `.nemo` file (smaller, cleaner, no training artifacts)

## Why Your Conversion Failed Before

The two old `.nemo` files (`epoch=49-step=45200_converted.nemo` and `epoch=49-step=45200_converted2.nemo`) likely failed because:

1. **API change**: NeMo's ConformerEncoder was updated - old conversion script used incompatible parameters
2. **Tokenizer issues**: The converted packages had broken references to external tokenizer files
3. **Version mismatch**: Checkpoint saved with NeMo v1.X loaded with v2.0 (API changed)

The new conversion script works because it uses the current NeMo API and properly handles the entire model restoration cycle.

## Files Involved

- **Training**: `finetune_multilingual_simple.py` (generates `.ckpt` files)
- **Conversion**: `convert_ckpt_to_nemo.py` (converts `.ckpt` → `.nemo`)
- **Inference**: `app_streaming_comparison.py` (loads `.nemo` files)

## Recommendation

Update `finetune_multilingual_simple.py` to:
1. Keep all training logic as-is
2. Add automatic `.nemo` conversion after training completes
3. Optionally delete old `.ckpt` files to save disk space
