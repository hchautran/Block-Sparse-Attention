#include <cstdio>

#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

using namespace cute;


template<class Shape, class  Stride>
__device__ void print2D(Layout<Shape, Stride> const& layout) {
    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            printf("%3d ", layout(i, j));
        }
        printf("\n");
    }
}


template<class Shape, class  Stride>
__device__ void print1D(Layout<Shape, Stride> const& layout) {
    for (int i = 0; i < size(layout); ++i) {
        printf("%3d ", layout(i));
    }
    printf("\n");
}


__global__ void dummy_kernel(float* storage) {
    // 4x4 row-major layout composed with a simple swizzle.
    using BaseLayout = Layout<Shape<_4, _4>, Stride<_4, _1>>;
    auto base = BaseLayout{};
    auto swizzled = composition(Swizzle<2, 2, 2>{}, base);

    Tensor base_matrix = make_tensor(make_gmem_ptr(storage), base);
    Tensor swz_matrix = make_tensor(make_gmem_ptr(storage), swizzled);

    if (thread0()) {
        printf("Base layout (row-major offsets):\n");
        print_latex(base);
        printf("Swizzled layout offsets:\n");
        print_latex(swizzled);
        printf("Base values:\n");
        for (int i = 0; i < size<0>(base); ++i) {
            for (int j = 0; j < size<1>(base); ++j) {
                printf("A(%d,%d)=%.1f\n", i, j, base_matrix(i, j));
            }
        }
        printf("Swizzled values (same storage, different view):\n");
        for (int i = 0; i < size<0>(swizzled); ++i) {
            for (int j = 0; j < size<1>(swizzled); ++j) {
                printf("S(%d,%d)=%.1f\n", i, j, swz_matrix(i, j));
            }
        }
    }
}

int main() {
    float A[16] = {
        0.0,  1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,  7.0,
        8.0,  9.0, 10.0, 11.0,
        12.0, 13.0, 14.0, 15.0
    };
    float* d_A;

    cudaMalloc((void**)& d_A, 16 * sizeof(float));
    cudaMemcpy(d_A, A, 16 * sizeof(float), cudaMemcpyHostToDevice);
    dummy_kernel<<<1, 32>>>(d_A);
    cudaDeviceSynchronize();
    cudaFree(d_A);
    return 0;
}
