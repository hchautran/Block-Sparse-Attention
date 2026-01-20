# SAM-Only Installation Guide

This guide will help you install and test the simplified SAM-only version of Block Sparse Attention.

## Quick Install (Recommended)

```bash
# 1. Install dependencies
pip install torch einops packaging psutil

# 2. Compile SAM-only version
pip install -e . -f setup_sam.py

# 3. Test installation
python -c "import block_sparse_attn_sam_cuda; print('✅ Installation successful!')"

# 4. Run examples
python examples/sam_usage_example.py
```

**Expected compile time: 1-2 minutes** (compared to 10-15 min for full version)

## Step-by-Step Guide

### Step 1: Check Prerequisites

```bash
# Check Python version (need >= 3.9)
python --version

# Check CUDA version (need >= 11.7)
nvcc --version

# Check GPU compute capability (need >= 8.0)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Compute capability: {torch.cuda.get_device_capability()}')"
```

**Expected output:**
```
Python 3.9.0 (or higher)
CUDA compilation tools, release 11.7 (or higher)
CUDA available: True
Compute capability: (8, 0)  # or higher (8,6), (9,0), etc.
```

### Step 2: Install PyTorch

If you don't have PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 3: Install Other Dependencies

```bash
pip install einops packaging psutil
```

### Step 4: Compile the Extension

```bash
# Navigate to the repository root
cd /Users/admin/Block-Sparse-Attention

# Compile SAM-only version
pip install -e . -f setup_sam.py
```

**What happens during compilation:**
1. Git submodule init (cutlass)
2. Compile 3 CUDA kernel files:
   - `flash_fwd_sam_hdim64_fp16.cu`
   - `flash_fwd_sam_hdim64_bf16.cu`
   - `flash_fwd_sam_hdim128_fp16.cu`
3. Compile C++ wrapper: `flash_api_sam.cpp`
4. Create Python package: `block_sparse_attn_sam_cuda`

**Expected output:**
```
Building wheel for block-sparse-attn-sam (setup.py) ... done
Successfully installed block-sparse-attn-sam-0.0.2+sam
```

### Step 5: Verify Installation

```bash
# Test import
python -c "import block_sparse_attn_sam_cuda; print('✅ C++ extension loaded')"

# Test Python interface
python -c "from block_sparse_attn.sam_attention import sam_block_sparse_attn_simple; print('✅ Python interface loaded')"

# Run quick test
python << 'EOF'
import torch
from block_sparse_attn.sam_attention import sam_block_sparse_attn_simple

if torch.cuda.is_available():
    q = torch.randn(2, 256, 12, 64, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = sam_block_sparse_attn_simple(q, k, v, base_blockmask=None)

    print(f'✅ SAM attention works! Output shape: {output.shape}')
else:
    print('⚠️  CUDA not available, skipping GPU test')
EOF
```

### Step 6: Run Examples

```bash
# Run comprehensive examples
python examples/sam_usage_example.py

# Run specific example
python examples/sam_masks.py
```

## Advanced: Custom Build Options

### Specify CUDA Architectures

By default, the build compiles for sm80, sm90, and sm100. To customize:

```bash
# For A100 only (sm80)
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"
pip install -e . -f setup_sam.py

# For A100 and H100 (sm80, sm90)
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;90"
pip install -e . -f setup_sam.py

# For consumer GPUs (RTX 3090=sm86, RTX 4090=sm89)
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="86;89"
pip install -e . -f setup_sam.py
```

### Adjust Parallel Compilation

If you run out of memory during compilation:

```bash
# Reduce parallel jobs
export MAX_JOBS=2
export NVCC_THREADS=2
pip install -e . -f setup_sam.py
```

### Force Rebuild

```bash
# Force complete rebuild
export BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE
pip install -e . -f setup_sam.py --force-reinstall --no-cache-dir
```

### Enable C++11 ABI

For compatibility with some Docker images:

```bash
export BLOCK_SPARSE_ATTN_FORCE_CXX11_ABI=TRUE
pip install -e . -f setup_sam.py
```

## Troubleshooting

### Issue: "cannot open source file 'cuda.h'"

**Solution**: CUDA toolkit not installed or not in PATH

```bash
# Check CUDA installation
which nvcc

# If not found, install CUDA toolkit
# On Ubuntu:
sudo apt-get install nvidia-cuda-toolkit

# Then verify
nvcc --version
```

### Issue: "error: namespace 'c10' has no member 'cuda'"

**Solution**: PyTorch version mismatch or CPU-only PyTorch

```bash
# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.version.cuda)"
```

### Issue: "CUDA out of memory" during compilation

**Solution**: Reduce parallel jobs

```bash
export MAX_JOBS=1
export NVCC_THREADS=1
pip install -e . -f setup_sam.py
```

### Issue: "requires Ampere GPUs or newer (sm80+)"

**Solution**: Your GPU is too old

```bash
# Check your GPU
python -c "import torch; print(torch.cuda.get_device_capability())"

# If output is (7,5) or lower, you need a newer GPU
# Minimum supported: RTX 3090, A100, A6000, H100, etc.
```

### Issue: Compilation succeeds but import fails

**Solution**: Clean rebuild

```bash
# Clean everything
pip uninstall block-sparse-attn-sam -y
rm -rf build dist *.egg-info
find . -name "*.so" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Rebuild
pip install -e . -f setup_sam.py
```

## Development Mode

If you're modifying the code:

```bash
# Install in development mode
pip install -e . -f setup_sam.py

# After modifying C++/CUDA code, rebuild:
python setup_sam.py build_ext --inplace

# After modifying Python code, no rebuild needed
```

## Uninstall

```bash
pip uninstall block-sparse-attn-sam
```

## Compare: Full vs SAM-Only Build

### Full Version

```bash
pip install -e .
# ⏱️  Compile time: 10-15 minutes
# 💾 Binary size: ~200MB
# 📦 Kernel files: 24
# ✨ Features: Training + Inference
```

### SAM-Only Version

```bash
pip install -e . -f setup_sam.py
# ⏱️  Compile time: 1-2 minutes ⚡
# 💾 Binary size: ~20-50MB 🎯
# 📦 Kernel files: 3 ✅
# ✨ Features: Inference only (SAM)
```

## Next Steps

1. Read `README_SAM.md` for usage guide
2. Run `examples/sam_usage_example.py` for comprehensive examples
3. Check `examples/sam_masks.py` for mask generation patterns
4. Integrate into your SAM model!

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review the error message carefully
3. Check GPU compatibility
4. Try a clean rebuild
5. Open an issue on GitHub with:
   - Full error message
   - Output of diagnostic commands
   - PyTorch and CUDA versions
