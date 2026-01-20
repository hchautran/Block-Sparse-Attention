# Fix for "undefined symbol: _ZN3c104cuda9SetDeviceEab"

This error means the compiled extension can't find PyTorch's C10 CUDA library at runtime.

## Quick Fix

### Option 1: Set LD_LIBRARY_PATH (Recommended)

```bash
# Find PyTorch library path
python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"

# Set it before running (replace path with output from above)
export LD_LIBRARY_PATH=/path/to/torch/lib:$LD_LIBRARY_PATH

# Now import should work
python -c "import block_sparse_attn_sam_cuda; print('Success!')"
```

### Option 2: Add to your script/environment permanently

Add to your `~/.bashrc` or `~/.bash_profile`:

```bash
# Add PyTorch libraries to path
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH
```

Then:
```bash
source ~/.bashrc
```

### Option 3: Clean rebuild with proper RPATH

```bash
# Clean everything
pip uninstall block-sparse-attn-sam -y
rm -rf build dist *.egg-info

# Rebuild with RPATH embedding
python setup_sam_fixed.py install
```

## Why This Happens

The symbol `_ZN3c104cuda9SetDeviceEab` decodes to `c10::cuda::SetDevice`, which is part of PyTorch's C10 CUDA library. The extension was compiled successfully but can't find `libc10_cuda.so` at runtime.

This typically happens when:
1. PyTorch libraries aren't in the system library search path
2. The extension wasn't built with RPATH to find PyTorch libraries
3. Different PyTorch versions at build vs runtime

## Verify the Issue

Check if libraries exist:
```bash
# Find torch library path
TORCH_LIB=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
echo "PyTorch lib path: $TORCH_LIB"

# Check if c10_cuda exists
ls -la $TORCH_LIB/libc10_cuda.so*

# Check what your extension is looking for
python -c "import block_sparse_attn_sam_cuda" 2>&1 | grep "undefined symbol"
```

## Permanent Fix

Use the fixed setup script that embeds RPATH:
```bash
python setup_sam_fixed.py install
```

This ensures the extension knows where to find PyTorch libraries at runtime.
