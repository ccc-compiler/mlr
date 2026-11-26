// Runtime for MLR programs that uses the Metal Performance Shaders.
//
// The interface is the same as the one of `mlr_runtime.c`, so the very same
// program can be linked against the plain C runtime or against this one. The
// matrix multiplication and the convolution are computed on the GPU with
// `MPSMatrixMultiplication`; the convolution is turned into a matrix
// multiplication with the im2col transformation. The remaining operations
// are simple enough to stay on the CPU.
//
// Build it with
//
//   clang++ -std=c++17 -isysroot $(xcrun --show-sdk-path) \
//       -framework Foundation -framework Metal -framework MetalPerformanceShaders \
//       runtime/mlr_runtime_mps.mm -o runtime
//
// (the plain C runtime is included here so that both runtimes implement the
// operations in exactly the same way).

extern "C" {
#define mlr_matmul mlr_matmul_cpu
#define mlr_conv2d mlr_conv2d_cpu
#define mlr_matmul_grad mlr_matmul_grad_cpu
#include "mlr_runtime.c"
#undef mlr_matmul
#undef mlr_conv2d
#undef mlr_matmul_grad
}

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <vector>

namespace {

// The Metal device and the command queue are created once.
struct Ctx {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    bool failed = false;
};

Ctx &ctx() {
    static Ctx c;
    if (c.device == nil && !c.failed) {
        c.device = MTLCreateSystemDefaultDevice();
        if (c.device == nil) {
            c.failed = true;
            fprintf(stderr, "mlr: no Metal device, falling back to the CPU\n");
        } else {
            c.queue = [c.device newCommandQueue];
            fprintf(stderr, "mlr: Metal Performance Shaders on %s\n",
                    [[c.device name] UTF8String]);
        }
    }
    return c;
}

// The row stride of the matrices is aligned to 64 bytes, the alignment that
// the Metal Performance Shaders expect.
const size_t kAlign = 64;

size_t padded_row_bytes(size_t columns) {
    size_t bytes = columns * sizeof(float);
    return ((bytes + kAlign - 1) / kAlign) * kAlign;
}

// C = A * B with A being M x K and B being K x N, all in row major order.
void gemm(const float *A, size_t M, size_t K, const float *B, size_t N, float *C) {
    Ctx &c = ctx();
    if (c.device == nil) {
        // Without a GPU the program still runs, only slower.
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        return;
    }

    const size_t lda = padded_row_bytes(K);
    const size_t ldb = padded_row_bytes(N);
    const size_t ldc = padded_row_bytes(N);

    id<MTLBuffer> bufA = [c.device newBufferWithLength:lda * M options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [c.device newBufferWithLength:ldb * K options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [c.device newBufferWithLength:ldc * M options:MTLResourceStorageModeShared];

    float *pa = (float *)[bufA contents];
    float *pb = (float *)[bufB contents];
    float *pc = (float *)[bufC contents];
    for (size_t i = 0; i < M; i++) {
        memcpy(pa + i * (lda / sizeof(float)), A + i * K, K * sizeof(float));
    }
    for (size_t i = 0; i < K; i++) {
        memcpy(pb + i * (ldb / sizeof(float)), B + i * N, N * sizeof(float));
    }
    memset(pc, 0, ldc * M);

    MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                    columns:K
                                                                   rowBytes:lda
                                                                   dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *db = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                    columns:N
                                                                   rowBytes:ldb
                                                                   dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                    columns:N
                                                                   rowBytes:ldc
                                                                   dataType:MPSDataTypeFloat32];

    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bufA offset:0 descriptor:da];
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bufB offset:0 descriptor:db];
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bufC offset:0 descriptor:dc];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:c.device
                                                                      transposeLeft:NO
                                                                     transposeRight:NO
                                                                         resultRows:M
                                                                      resultColumns:N
                                                                   interiorColumns:K
                                                                               alpha:1.0
                                                                                beta:0.0];

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    [cmd commit];
    [cmd waitUntilCompleted];

    for (size_t i = 0; i < M; i++) {
        memcpy(C + i * N, pc + i * (ldc / sizeof(float)), N * sizeof(float));
    }
}

} // namespace

// The matrix multiplication runs on the GPU.
extern "C" void mlr_matmul(const void *a, const void *b, void *c, int64_t M, int64_t N, int64_t K) {
    gemm((const float *)a, (size_t)M, (size_t)K, (const float *)b, (size_t)N, (float *)c);
}

// Transposes a row major matrix.
static void transpose_f32(const float *a, size_t rows, size_t cols, float *out) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            out[j * rows + i] = a[i * cols + j];
        }
    }
}

// The gradient of the matrix multiplication: two matrix multiplications on
// the GPU, with the transposed operand packed into a temporary buffer.
// dx (M x K) = dy (M x N) @ w^T; dw (K x N) = x^T @ dy.
extern "C" void mlr_matmul_grad(const void *x, const void *w, const void *dy, void *dx, void *dw,
                                int64_t M, int64_t K, int64_t N) {
    Ctx &c = ctx();
    if (c.device == nil) {
        mlr_matmul_grad_cpu(x, w, dy, dx, dw, M, K, N);
        return;
    }
    float *wt = (float *)malloc((size_t)(K * N) * sizeof(float));
    float *xt = (float *)malloc((size_t)(K * M) * sizeof(float));
    if (wt == nullptr || xt == nullptr) {
        free(wt);
        free(xt);
        mlr_matmul_grad_cpu(x, w, dy, dx, dw, M, K, N);
        return;
    }
    transpose_f32((const float *)w, (size_t)K, (size_t)N, wt);
    transpose_f32((const float *)x, (size_t)M, (size_t)K, xt);
    gemm((const float *)dy, (size_t)M, (size_t)N, wt, (size_t)K, (float *)dx);
    gemm(xt, (size_t)K, (size_t)M, (const float *)dy, (size_t)N, (float *)dw);
    free(wt);
    free(xt);
}

// The convolution is an im2col transformation followed by a matrix
// multiplication on the GPU.
extern "C" void mlr_conv2d(const void *x, const void *w, const void *bias, void *out, int64_t O,
                            int64_t H, int64_t W, int64_t K) {
    const float *X = (const float *)x;
    const float *Wt = (const float *)w;
    const float *B = (const float *)bias;
    float *Y = (float *)out;

    const int64_t pad = K / 2;
    const int64_t pixels = H * W;
    const int64_t patch = K * K;

    // The patches of the image, one row per output pixel.
    std::vector<float> cols((size_t)(pixels * patch), 0.0f);
    for (int64_t p = 0; p < pixels; p++) {
        const int64_t y = p / W;
        const int64_t x = p % W;
        for (int64_t ky = 0; ky < K; ky++) {
            const int64_t iy = y + ky - pad;
            for (int64_t kx = 0; kx < K; kx++) {
                const int64_t ix = x + kx - pad;
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                cols[(size_t)(p * patch + ky * K + kx)] = X[iy * W + ix];
            }
        }
    }

    // The filters, transposed: one row per patch element.
    std::vector<float> wt((size_t)(patch * O), 0.0f);
    for (int64_t o = 0; o < O; o++) {
        for (int64_t q = 0; q < patch; q++) {
            wt[(size_t)(q * O + o)] = Wt[o * patch + q];
        }
    }

    std::vector<float> result((size_t)(pixels * O), 0.0f);
    gemm(cols.data(), (size_t)pixels, (size_t)patch, wt.data(), (size_t)O, result.data());

    for (int64_t p = 0; p < pixels; p++) {
        const int64_t y = p / W;
        const int64_t x = p % W;
        for (int64_t o = 0; o < O; o++) {
            Y[(o * H + y) * W + x] = result[(size_t)(p * O + o)] + B[o];
        }
    }
}
