/* Runtime for MLR programs.
 *
 * The compiler lowers `print(x)` into a call to the `mlr_print_*` functions
 * below and the domain operations (`matmul`, `conv2d`, ...) into calls to the
 * `mlr_*` functions at the bottom of this file.
 *
 * The tensors are plain row major float buffers; the addresses are passed as
 * 64 bit integers so that the interface is the same for the CPU and for the
 * Metal Performance Shaders implementation (`mlr_runtime_mps.mm`).
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void mlr_print_i32(int x) { printf("%d\n", x); }
void mlr_print_i64(long long x) { printf("%lld\n", x); }
void mlr_print_f32(float x) { printf("%f\n", (double)x); }
void mlr_print_f64(double x) { printf("%f\n", x); }
void mlr_print_bool(bool x) { printf("%s\n", x ? "true" : "false"); }

/* C = A (M x K) * B (K x N), all row major. */
void mlr_matmul(const void *a, const void *b, void *c, int64_t M, int64_t N, int64_t K) {
    const float *A = (const float *)a;
    const float *B = (const float *)b;
    float *C = (float *)c;
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* A bias is added to every row of a matrix. */
void mlr_add_bias(const void *x, const void *b, void *out, int64_t M, int64_t N) {
    const float *X = (const float *)x;
    const float *B = (const float *)b;
    float *Y = (float *)out;
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            Y[i * N + j] = X[i * N + j] + B[j];
        }
    }
}

/* Convolution of a single channel image with O filters of size K x K.
 * The padding is K / 2, the stride is one. */
void mlr_conv2d(const void *x, const void *w, const void *bias, void *out, int64_t O,
                int64_t H, int64_t W, int64_t K) {
    const float *X = (const float *)x;
    const float *Wt = (const float *)w;
    const float *B = (const float *)bias;
    float *Y = (float *)out;
    int64_t pad = K / 2;
    for (int64_t o = 0; o < O; o++) {
        for (int64_t y = 0; y < H; y++) {
            for (int64_t x = 0; x < W; x++) {
                float sum = B[o];
                for (int64_t ky = 0; ky < K; ky++) {
                    int64_t iy = y + ky - pad;
                    if (iy < 0 || iy >= H) continue;
                    for (int64_t kx = 0; kx < K; kx++) {
                        int64_t ix = x + kx - pad;
                        if (ix < 0 || ix >= W) continue;
                        sum += Wt[o * K * K + ky * K + kx] * X[iy * W + ix];
                    }
                }
                Y[(o * H + y) * W + x] = sum;
            }
        }
    }
}

/* Max pooling with a k x k window (stride k). */
void mlr_maxpool2d(const void *x, void *out, int64_t O, int64_t H, int64_t W, int64_t k) {
    const float *X = (const float *)x;
    float *Y = (float *)out;
    int64_t ho = H / k, wo = W / k;
    for (int64_t o = 0; o < O; o++) {
        for (int64_t y = 0; y < ho; y++) {
            for (int64_t x = 0; x < wo; x++) {
                float best = -3.402823466e+38f;
                for (int64_t dy = 0; dy < k; dy++) {
                    for (int64_t dx = 0; dx < k; dx++) {
                        float v = X[(o * H + y * k + dy) * W + x * k + dx];
                        if (v > best) best = v;
                    }
                }
                Y[(o * ho + y) * wo + x] = best;
            }
        }
    }
}

/* Rectified linear unit. */
void mlr_relu(const void *x, void *out, int64_t n) {
    const float *X = (const float *)x;
    float *Y = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
    }
}

/* A flat copy of the tensor. */
void mlr_flatten(const void *x, void *out, int64_t n) {
    memcpy(out, x, (size_t)n * sizeof(float));
}

/* The `index`-th row of a matrix. */
void mlr_row(const void *t, void *out, int64_t n, int64_t stride, int64_t index) {
    memcpy(out, (const float *)t + index * stride, (size_t)n * sizeof(float));
}

/* The index of the largest element. */
int32_t mlr_argmax(const void *x, int64_t n) {
    const float *X = (const float *)x;
    int32_t best = 0;
    for (int64_t i = 1; i < n; i++) {
        if (X[i] > X[best]) best = (int32_t)i;
    }
    return best;
}
