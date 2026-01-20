"""
SAM-Only Block Sparse Attention Setup (with fixed linking)

This version ensures proper linking to PyTorch libraries by embedding RPATH.
"""

import sys
import functools
import warnings
import os
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "block_sparse_attn_sam"

FORCE_BUILD = os.getenv("BLOCK_SPARSE_ATTN_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("BLOCK_SPARSE_ATTN_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
FORCE_CXX11_ABI = os.getenv("BLOCK_SPARSE_ATTN_FORCE_CXX11_ABI", "FALSE") == "TRUE"

@functools.lru_cache(maxsize=None)
def cuda_archs() -> list:
    return os.getenv("BLOCK_SPARSE_ATTN_CUDA_ARCHS", "80;90;100").split(";")


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    """Add -gencode flags for requested architectures."""
    supported_ptx_archs = []

    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
        supported_ptx_archs.append("80")

    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
        supported_ptx_archs.append("90")

    if bare_metal_version >= Version("12.8") and "100" in archs:
        if bare_metal_version >= Version("12.9"):
            cc_flag += ["-gencode", "arch=compute_100f,code=sm_100"]
            supported_ptx_archs.append("100f")
        else:
            cc_flag += ["-gencode", "arch=compute_100,code=sm_100"]
            supported_ptx_archs.append("100")

    if supported_ptx_archs:
        newest = max(
            supported_ptx_archs,
            key=lambda arch: int("".join(ch for ch in arch if ch.isdigit())),
        )
        cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    return cc_flag


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found. "
        "Are you sure your environment has nvcc available?"
    )


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


cmdclass = {}
ext_modules = []

subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=False)

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("block_sparse_attn_sam")

    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.7"):
            raise RuntimeError(
                "SAM Block Sparse Attention requires CUDA 11.7 or above. "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
        add_cuda_gencodes(cc_flag, set(cuda_archs()), bare_metal_version)

    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]

    compiler_c17_flag = ["-O3", "-std=c++17"]

    if sys.platform == "win32" and os.getenv('DISTUTILS_USE_SDK') == '1':
        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus"])
        compiler_c17_flag = ["-O2", "/std:c++17", "/Zc:__cplusplus"]

    sam_sources = [
        "csrc/block_sparse_attn/flash_api_sam.cpp",
        "csrc/block_sparse_attn/src/flash_fwd_sam_hdim64_fp16.cu",
        "csrc/block_sparse_attn/src/flash_fwd_sam_hdim64_bf16.cu",
        "csrc/block_sparse_attn/src/flash_fwd_sam_hdim32_fp16.cu",
        "csrc/block_sparse_attn/src/flash_fwd_sam_hdim32_bf16.cu",
        "csrc/block_sparse_attn/src/flash_fwd_sam_hdim128_fp16.cu",
        "csrc/block_sparse_attn/src/flash_fwd_sam_hdim128_bf16.cu",
        "csrc/block_sparse_attn/src/flash_fwd_sam_hdim256_fp16.cu",
        "csrc/block_sparse_attn/src/flash_fwd_sam_hdim256_bf16.cu",
    ]

    # Get PyTorch library path for RPATH embedding
    torch_lib_path = Path(torch.__file__).parent / "lib"

    # Build extra link arguments with RPATH
    extra_link_args = []
    if sys.platform.startswith('linux'):
        # On Linux, embed RPATH so the extension can find PyTorch libraries
        extra_link_args = [
            f"-Wl,-rpath,{torch_lib_path}",
            f"-L{torch_lib_path}",
        ]
    elif sys.platform == 'darwin':
        # On macOS, use @loader_path
        extra_link_args = [
            f"-Wl,-rpath,{torch_lib_path}",
            f"-L{torch_lib_path}",
        ]

    ext_modules.append(
        CUDAExtension(
            name="block_sparse_attn_sam_cuda",
            sources=sam_sources,
            extra_compile_args={
                "cxx": compiler_c17_flag,
                "nvcc": append_nvcc_threads(nvcc_flags + cc_flag),
            },
            extra_link_args=extra_link_args,
            include_dirs=[
                Path(this_dir) / "csrc" / "block_sparse_attn",
                Path(this_dir) / "csrc" / "block_sparse_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )


def get_package_version():
    with open(Path(this_dir) / "block_sparse_attn" / "__init__.py", "r") as f:
        import re
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = version_match.group(1).strip('"\'')
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+sam.{local_version}"
    else:
        return f"{public_version}+sam"


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        if not os.environ.get("MAX_JOBS"):
            import psutil
            max_num_jobs_cores = max(1, os.cpu_count() // 2)
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
            max_num_jobs_memory = int(free_memory_gb / 9)
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)
        super().__init__(*args, **kwargs)


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "block_sparse_attn.egg-info",
        )
    ),
    author="Junxian Guo",
    author_email="junxian@mit.edu",
    description="SAM Block Sparse Attention (Inference Only)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mit-han-lab/Block-Sparse-Attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension} if ext_modules else {},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "einops",
    ],
)
