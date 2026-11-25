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
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mlr_print_i32(int x) { printf("%d\n", x); }
void mlr_print_i64(long long x) { printf("%lld\n", x); }
void mlr_print_f32(float x) { printf("%f\n", (double)x); }
void mlr_print_f64(double x) { printf("%f\n", x); }
void mlr_print_bool(int32_t x) { printf("%s\n", x ? "true" : "false"); }

/* Writes a single byte, used by the language models that generate text. */
void mlr_print_byte(int32_t x) { fputc((int)(x & 0xff), stdout); }

/* The scalar functions of the standard library. */
float mlr_exp(float x) { return expf(x); }
float mlr_ln(float x) { return logf(x); }
float mlr_sqrt(float x) { return sqrtf(x); }
float mlr_tanh(float x) { return tanhf(x); }
float mlr_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float mlr_gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118f)); }

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

/* The row of a table that belongs to an index. */
void mlr_embed(const void *table, int32_t id, void *out, int64_t rows, int64_t width) {
    const float *T = (const float *)table;
    float *O = (float *)out;
    if (id < 0 || id >= rows) {
        for (int64_t i = 0; i < width; i++) O[i] = 0.0f;
        return;
    }
    memcpy(O, T + (int64_t)id * width, (size_t)width * sizeof(float));
}

/* The rows of a table that belong to a tensor of indices. */
void mlr_embed_seq(const void *table, const void *ids, void *out, int64_t rows, int64_t width,
                   int64_t count) {
    const float *T = (const float *)table;
    const int32_t *I = (const int32_t *)ids;
    float *O = (float *)out;
    for (int64_t i = 0; i < count; i++) {
        int32_t id = I[i];
        if (id < 0 || id >= rows) {
            for (int64_t j = 0; j < width; j++) O[i * width + j] = 0.0f;
            continue;
        }
        memcpy(O + i * width, T + (int64_t)id * width, (size_t)width * sizeof(float));
    }
}

/* Every row of a matrix is normalized to zero mean and unit variance and is
 * then scaled and shifted. */
void mlr_layernorm(const void *x, const void *gamma, const void *beta, void *out, int64_t rows,
                   int64_t width) {
    const float *X = (const float *)x;
    const float *G = (const float *)gamma;
    const float *B = (const float *)beta;
    float *Y = (float *)out;
    for (int64_t r = 0; r < rows; r++) {
        const float *row = X + r * width;
        float *dst = Y + r * width;
        double mean = 0.0;
        for (int64_t i = 0; i < width; i++) mean += row[i];
        mean /= (double)width;
        double var = 0.0;
        for (int64_t i = 0; i < width; i++) {
            double d = row[i] - mean;
            var += d * d;
        }
        var /= (double)width;
        float inv = 1.0f / sqrtf((float)(var + 1e-5));
        for (int64_t i = 0; i < width; i++) {
            dst[i] = ((float)((row[i] - mean) * (double)inv)) * G[i] + B[i];
        }
    }
}

/* The softmax of the rows of a matrix; the columns beyond the length are
 * masked out. */
void mlr_softmax_causal(const void *scores, int32_t length, void *out, int64_t size) {
    const float *S = (const float *)scores;
    float *Y = (float *)out;
    for (int64_t r = 0; r < size; r++) {
        const float *row = S + r * size;
        float *dst = Y + r * size;
        int64_t n = length < size ? length : size;
        float best = -3.402823466e+38f;
        for (int64_t j = 0; j < n; j++) {
            if (row[j] > best) best = row[j];
        }
        double sum = 0.0;
        for (int64_t j = 0; j < n; j++) {
            float e = expf(row[j] - best);
            dst[j] = e;
            sum += e;
        }
        for (int64_t j = 0; j < n; j++) dst[j] = (float)(dst[j] / sum);
        for (int64_t j = n; j < size; j++) dst[j] = 0.0f;
    }
}

/* A small random number generator, so that the sampling is reproducible. */
static uint32_t mlr_random_state = 42u;

static float mlr_random_float(void) {
    uint32_t x = mlr_random_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    mlr_random_state = x;
    return (float)(x >> 8) / 16777216.0f;
}

/* A token is drawn from the distribution of the logits. */
int32_t mlr_sample(const void *logits, float temperature, int32_t seed, int64_t n) {
    const float *L = (const float *)logits;
    if (n <= 0) return 0;
    mlr_random_state = (uint32_t)seed * 2654435761u + 1u;
    float best = -3.402823466e+38f;
    for (int64_t i = 0; i < n; i++) {
        if (L[i] > best) best = L[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) sum += expf((L[i] - best) / temperature);
    float u = mlr_random_float() * sum;
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        acc += expf((L[i] - best) / temperature);
        if (u < acc) return (int32_t)i;
    }
    return (int32_t)(n - 1);
}

/* The transpose of a matrix. */
void mlr_transpose(const void *x, void *out, int64_t rows, int64_t width) {
    const float *X = (const float *)x;
    float *Y = (float *)out;
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < width; c++) {
            Y[c * rows + r] = X[r * width + c];
        }
    }
}

/* -------------------------------------------------------------------------
 * Writing PyTorch checkpoints
 *
 * `mlr.save_pt` is lowered into a call to `mlr_save_pt_begin`, one call to
 * `mlr_save_pt_add` per tensor and a final call to `mlr_save_pt_end`. The
 * tensors are collected in memory and written as a `.pt` archive (a ZIP file
 * that holds a pickle and the raw storage of every tensor) when the
 * checkpoint is closed.
 * ------------------------------------------------------------------------- */

typedef struct {
    unsigned char *data;
    size_t len;
    size_t cap;
} mlr_buf;

static void mlr_buf_reserve(mlr_buf *b, size_t n) {
    if (b->len + n <= b->cap) {
        return;
    }
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + n) {
        cap *= 2;
    }
    unsigned char *p = (unsigned char *)realloc(b->data, cap);
    if (!p) {
        fprintf(stderr, "mlr: out of memory\n");
        abort();
    }
    b->data = p;
    b->cap = cap;
}

static void mlr_buf_put(mlr_buf *b, const void *data, size_t n) {
    mlr_buf_reserve(b, n);
    memcpy(b->data + b->len, data, n);
    b->len += n;
}

static void mlr_buf_byte(mlr_buf *b, unsigned char v) { mlr_buf_put(b, &v, 1); }

static void mlr_buf_u16(mlr_buf *b, uint16_t v) {
    unsigned char t[2] = {(unsigned char)(v & 0xff), (unsigned char)((v >> 8) & 0xff)};
    mlr_buf_put(b, t, 2);
}

static void mlr_buf_u32(mlr_buf *b, uint32_t v) {
    unsigned char t[4] = {(unsigned char)(v & 0xff), (unsigned char)((v >> 8) & 0xff),
                          (unsigned char)((v >> 16) & 0xff), (unsigned char)((v >> 24) & 0xff)};
    mlr_buf_put(b, t, 4);
}

/* The CRC32 of the ZIP format. */
static uint32_t mlr_crc32(const unsigned char *data, size_t len) {
    static uint32_t table[256];
    static int ready = 0;
    if (!ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        ready = 1;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/* A string of a pickle. Torch only accepts BINUNICODE in the weights only
 * mode, so every string uses it. */
static void mlr_pkl_string(mlr_buf *b, const char *s) {
    size_t n = strlen(s);
    mlr_buf_byte(b, 0x58); /* BINUNICODE */
    mlr_buf_u32(b, (uint32_t)n);
    mlr_buf_put(b, s, n);
}

static void mlr_pkl_global(mlr_buf *b, const char *module, const char *name) {
    mlr_buf_byte(b, 0x63); /* GLOBAL */
    mlr_buf_put(b, module, strlen(module));
    mlr_buf_byte(b, '\n');
    mlr_buf_put(b, name, strlen(name));
    mlr_buf_byte(b, '\n');
}

static void mlr_pkl_int(mlr_buf *b, int64_t v) {
    if (v >= 0 && v < 256) {
        mlr_buf_byte(b, 0x4b); /* BININT1 */
        mlr_buf_byte(b, (unsigned char)v);
    } else if (v >= -2147483648LL && v <= 2147483647LL) {
        mlr_buf_byte(b, 0x4a); /* BININT */
        mlr_buf_u32(b, (uint32_t)(int32_t)v);
    } else {
        mlr_buf_byte(b, 0x49); /* LONG1 */
        mlr_buf_byte(b, 8);
        mlr_buf_put(b, &v, 8);
    }
}

static void mlr_pkl_tuple(mlr_buf *b, const int64_t *values, int n) {
    mlr_buf_byte(b, 0x28); /* MARK */
    for (int i = 0; i < n; i++) {
        mlr_pkl_int(b, values[i]);
    }
    mlr_buf_byte(b, 0x74); /* TUPLE */
}

typedef struct {
    char name[16];
    unsigned char *data;
    size_t len;
} mlr_storage;

static mlr_buf mlr_ckpt_pickle;
static char *mlr_ckpt_path;
static mlr_storage *mlr_ckpt_storages;
static int mlr_ckpt_tensors;

static void mlr_ckpt_reset(void) {
    mlr_ckpt_pickle.len = 0;
    for (int i = 0; i < mlr_ckpt_tensors; i++) {
        free(mlr_ckpt_storages[i].data);
    }
    mlr_ckpt_tensors = 0;
    free(mlr_ckpt_path);
    mlr_ckpt_path = NULL;
}

void mlr_save_pt_begin(int64_t path) {
    mlr_ckpt_reset();
    const char *p = (const char *)(uintptr_t)path;
    size_t n = strlen(p);
    mlr_ckpt_path = (char *)malloc(n + 1);
    memcpy(mlr_ckpt_path, p, n + 1);
    mlr_buf_byte(&mlr_ckpt_pickle, 0x80); /* PROTO 2 */
    mlr_buf_byte(&mlr_ckpt_pickle, 0x02);
    mlr_buf_byte(&mlr_ckpt_pickle, 0x7d); /* EMPTY_DICT */
    mlr_buf_byte(&mlr_ckpt_pickle, 0x28); /* MARK */
}

/* The element kinds of `mlr_save_pt_add`. */
#define MLR_ELEM_F32 0
#define MLR_ELEM_I32 1
#define MLR_ELEM_I64 2

static const char *mlr_storage_class(int64_t elem) {
    switch (elem) {
    case MLR_ELEM_I32: return "IntStorage";
    case MLR_ELEM_I64: return "LongStorage";
    default: return "FloatStorage";
    }
}

static size_t mlr_elem_size(int64_t elem) {
    switch (elem) {
    case MLR_ELEM_I32: return 4;
    case MLR_ELEM_I64: return 8;
    default: return 4;
    }
}

void mlr_save_pt_add(int64_t key, int64_t data, int64_t count, int64_t ndim, int64_t d0,
                     int64_t d1, int64_t d2, int64_t d3, int64_t elem) {
    const char *name = (const char *)(uintptr_t)key;
    const void *src = (const void *)(uintptr_t)data;
    int64_t shape[4] = {d0, d1, d2, d3};
    size_t bytes = (size_t)count * mlr_elem_size(elem);

    /* The storage of the tensor, copied so that it outlives the call. */
    char storage[16];
    snprintf(storage, sizeof(storage), "%d", mlr_ckpt_tensors);
    mlr_ckpt_storages = (mlr_storage *)realloc(
        mlr_ckpt_storages, (size_t)(mlr_ckpt_tensors + 1) * sizeof(mlr_storage));
    mlr_storage *s = &mlr_ckpt_storages[mlr_ckpt_tensors];
    snprintf(s->name, sizeof(s->name), "%s", storage);
    s->data = (unsigned char *)malloc(bytes ? bytes : 1);
    memcpy(s->data, src, bytes);
    s->len = bytes;
    mlr_ckpt_tensors++;

    /* The row major strides of the tensor. */
    int64_t stride[4] = {0, 0, 0, 0};
    int64_t acc = 1;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        stride[i] = acc;
        acc *= shape[i];
    }

    mlr_buf *p = &mlr_ckpt_pickle;
    mlr_pkl_string(p, name);
    mlr_pkl_global(p, "torch._utils", "_rebuild_tensor_v2");
    mlr_buf_byte(p, 0x28); /* MARK (the arguments) */
    /* The persistent id of the storage. */
    mlr_buf_byte(p, 0x28); /* MARK */
    mlr_pkl_string(p, "storage");
    mlr_pkl_global(p, "torch", mlr_storage_class(elem));
    mlr_pkl_string(p, storage);
    mlr_pkl_string(p, "cpu");
    mlr_pkl_int(p, count);
    mlr_buf_byte(p, 0x74); /* TUPLE */
    mlr_buf_byte(p, 0x51); /* BINPERSID */
    mlr_pkl_int(p, 0);     /* the storage offset */
    mlr_pkl_tuple(p, shape, (int)ndim);
    mlr_pkl_tuple(p, stride, (int)ndim);
    mlr_buf_byte(p, 0x89); /* NEWFALSE (requires_grad) */
    /* The empty ordered dictionary of the hooks. */
    mlr_pkl_global(p, "collections", "OrderedDict");
    mlr_buf_byte(p, 0x28); /* MARK */
    mlr_buf_byte(p, 0x7d); /* EMPTY_DICT */
    mlr_buf_byte(p, 0x74); /* TUPLE */
    mlr_buf_byte(p, 0x52); /* REDUCE */
    mlr_buf_byte(p, 0x74); /* TUPLE (the arguments) */
    mlr_buf_byte(p, 0x52); /* REDUCE */
}

/* A ZIP entry of a checkpoint. */
typedef struct {
    const char *name;
    const unsigned char *data;
    size_t len;
    uint32_t crc;
    uint32_t offset;
} mlr_zip_entry;

static void mlr_zip_add(mlr_buf *out, mlr_zip_entry *e) {
    e->crc = mlr_crc32(e->data, e->len);
    e->offset = (uint32_t)out->len;
    uint16_t n = (uint16_t)strlen(e->name);
    mlr_buf_u32(out, 0x04034b50u); /* the local file header */
    mlr_buf_u16(out, 20);          /* the version */
    mlr_buf_u16(out, 0);           /* the flags */
    mlr_buf_u16(out, 0);           /* the method: stored */
    mlr_buf_u16(out, 0);           /* the time */
    mlr_buf_u16(out, 0);           /* the date */
    mlr_buf_u32(out, e->crc);
    mlr_buf_u32(out, (uint32_t)e->len);
    mlr_buf_u32(out, (uint32_t)e->len);
    mlr_buf_u16(out, n);
    mlr_buf_u16(out, 0); /* the extra field */
    mlr_buf_put(out, e->name, n);
    mlr_buf_put(out, e->data, e->len);
}

void mlr_save_pt_end(void) {
    /* The empty dictionary of a checkpoint without tensors is still a valid
     * archive. */
    mlr_buf_byte(&mlr_ckpt_pickle, 0x75); /* SETITEMS */
    mlr_buf_byte(&mlr_ckpt_pickle, 0x2e); /* STOP */

    static const unsigned char version[] = "3\n";
    static const unsigned char byteorder[] = "little";
    int count = mlr_ckpt_tensors + 3;
    mlr_zip_entry *entries = (mlr_zip_entry *)malloc((size_t)count * sizeof(mlr_zip_entry));
    char (*names)[32] = (char (*)[32])malloc((size_t)mlr_ckpt_tensors * 32);
    entries[0].name = "archive/data.pkl";
    entries[0].data = mlr_ckpt_pickle.data;
    entries[0].len = mlr_ckpt_pickle.len;
    for (int i = 0; i < mlr_ckpt_tensors; i++) {
        snprintf(names[i], 32, "archive/data/%s", mlr_ckpt_storages[i].name);
        entries[i + 1].name = names[i];
        entries[i + 1].data = mlr_ckpt_storages[i].data;
        entries[i + 1].len = mlr_ckpt_storages[i].len;
    }
    entries[count - 2].name = "archive/version";
    entries[count - 2].data = version;
    entries[count - 2].len = 2;
    entries[count - 1].name = "archive/byteorder";
    entries[count - 1].data = byteorder;
    entries[count - 1].len = 6;

    mlr_buf out = {0};
    for (int i = 0; i < count; i++) {
        mlr_zip_add(&out, &entries[i]);
    }
    uint32_t directory = (uint32_t)out.len;
    for (int i = 0; i < count; i++) {
        mlr_zip_entry *e = &entries[i];
        uint16_t n = (uint16_t)strlen(e->name);
        mlr_buf_u32(&out, 0x02014b50u); /* the central directory header */
        mlr_buf_u16(&out, 20);          /* the version made by */
        mlr_buf_u16(&out, 20);          /* the version needed */
        mlr_buf_u16(&out, 0);           /* the flags */
        mlr_buf_u16(&out, 0);           /* the method */
        mlr_buf_u16(&out, 0);           /* the time */
        mlr_buf_u16(&out, 0);           /* the date */
        mlr_buf_u32(&out, e->crc);
        mlr_buf_u32(&out, (uint32_t)e->len);
        mlr_buf_u32(&out, (uint32_t)e->len);
        mlr_buf_u16(&out, n);
        mlr_buf_u16(&out, 0); /* the extra field */
        mlr_buf_u16(&out, 0); /* the comment */
        mlr_buf_u16(&out, 0); /* the disk */
        mlr_buf_u16(&out, 0); /* the internal attributes */
        mlr_buf_u32(&out, 0); /* the external attributes */
        mlr_buf_u32(&out, e->offset);
        mlr_buf_put(&out, e->name, n);
    }
    uint32_t size = (uint32_t)out.len - directory;
    mlr_buf_u32(&out, 0x06054b50u); /* the end of the central directory */
    mlr_buf_u16(&out, 0);
    mlr_buf_u16(&out, 0);
    mlr_buf_u16(&out, (uint16_t)count);
    mlr_buf_u16(&out, (uint16_t)count);
    mlr_buf_u32(&out, size);
    mlr_buf_u32(&out, directory);
    mlr_buf_u16(&out, 0);

    if (mlr_ckpt_path) {
        FILE *f = fopen(mlr_ckpt_path, "wb");
        if (!f) {
            fprintf(stderr, "mlr: cannot write `%s`\n", mlr_ckpt_path);
        } else {
            fwrite(out.data, 1, out.len, f);
            fclose(f);
        }
    }
    free(entries);
    free(names);
    free(out.data);
    mlr_ckpt_reset();
}

/* -------------------------------------------------------------------------
 * Training: the loss and the gradients of the forward operations.
 * ------------------------------------------------------------------------- */

/* The mean cross entropy of the rows of the logits. */
float mlr_cross_entropy(const void *logits, const void *labels, int64_t rows, int64_t vocab) {
    const float *L = (const float *)logits;
    const int32_t *Y = (const int32_t *)labels;
    double total = 0.0;
    for (int64_t r = 0; r < rows; r++) {
        const float *row = L + r * vocab;
        float best = row[0];
        for (int64_t j = 1; j < vocab; j++) {
            if (row[j] > best) best = row[j];
        }
        double sum = 0.0;
        for (int64_t j = 0; j < vocab; j++) {
            sum += expf(row[j] - best);
        }
        int32_t label = Y[r];
        if (label < 0 || label >= vocab) {
            fprintf(stderr, "mlr: the label %d is outside the vocabulary\n", label);
            abort();
        }
        total -= log((double)expf(row[label] - best) / sum);
    }
    return (float)(total / (double)rows);
}

/* The gradient of the cross entropy: scale * (softmax - one hot) / rows. */
void mlr_cross_entropy_grad(const void *logits, const void *labels, float scale, void *out,
                            int64_t rows, int64_t vocab) {
    const float *L = (const float *)logits;
    const int32_t *Y = (const int32_t *)labels;
    float *D = (float *)out;
    scale = scale * (rows > 0 ? 1.0f / (float)rows : 0.0f);
    for (int64_t r = 0; r < rows; r++) {
        const float *row = L + r * vocab;
        float *d = D + r * vocab;
        float best = row[0];
        for (int64_t j = 1; j < vocab; j++) {
            if (row[j] > best) best = row[j];
        }
        double sum = 0.0;
        for (int64_t j = 0; j < vocab; j++) {
            sum += expf(row[j] - best);
        }
        for (int64_t j = 0; j < vocab; j++) {
            float p = (float)(expf(row[j] - best) / sum);
            d[j] = (p - (j == Y[r] ? 1.0f : 0.0f)) * scale;
        }
    }
}

/* The softmax of the rows of a matrix. */
void mlr_softmax(const void *x, void *out, int64_t rows, int64_t cols) {
    const float *X = (const float *)x;
    float *Y = (float *)out;
    for (int64_t r = 0; r < rows; r++) {
        const float *row = X + r * cols;
        float *dst = Y + r * cols;
        float best = row[0];
        for (int64_t j = 1; j < cols; j++) {
            if (row[j] > best) best = row[j];
        }
        double sum = 0.0;
        for (int64_t j = 0; j < cols; j++) {
            float e = expf(row[j] - best);
            dst[j] = e;
            sum += e;
        }
        for (int64_t j = 0; j < cols; j++) {
            dst[j] = (float)(dst[j] / sum);
        }
    }
}

/* A buffer filled with zeros. */
void mlr_zero(void *p, int64_t n) {
    memset(p, 0, (size_t)n * 4);
}

/* The element wise sum of two tensors. */
void mlr_add_elem(const void *a, const void *b, void *out, int64_t n) {
    const float *A = (const float *)a;
    const float *B = (const float *)b;
    float *O = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        O[i] = A[i] + B[i];
    }
}

/* The gradients of y = x @ w: dx = dy @ w^T, dw = x^T @ dy. */
void mlr_matmul_grad(const void *x, const void *w, const void *dy, void *dx, void *dw, int64_t M,
                     int64_t K, int64_t N) {
    const float *X = (const float *)x;
    const float *W = (const float *)w;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    float *DW = (float *)dw;
    /* dx (M x K) = dy (M x N) @ w^T (N x K). */
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < N; k++) {
                sum += DY[i * N + k] * W[j * N + k];
            }
            DX[i * K + j] = sum;
        }
    }
    /* dw (K x N) = x^T (K x M) @ dy (M x N). */
    for (int64_t i = 0; i < K; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < M; k++) {
                sum += X[k * K + i] * DY[k * N + j];
            }
            DW[i * N + j] = sum;
        }
    }
}

/* The gradients of y = m + b: dm = dy, db = the sum of the rows of dy. */
void mlr_add_bias_grad(const void *dy, void *dm, void *db, int64_t rows, int64_t cols) {
    const float *DY = (const float *)dy;
    float *DM = (float *)dm;
    float *DB = (float *)db;
    for (int64_t j = 0; j < cols; j++) {
        DB[j] = 0.0f;
    }
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            float v = DY[i * cols + j];
            DM[i * cols + j] = v;
            DB[j] += v;
        }
    }
}

/* The gradient of relu: dx = dy where x > 0. */
void mlr_relu_grad(const void *x, const void *dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = X[i] > 0.0f ? DY[i] : 0.0f;
    }
}

/* The gradient of gelu: dx = dy * (phi(x) + x * density(x)). */
void mlr_gelu_grad(const void *x, const void *dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    for (int64_t i = 0; i < n; i++) {
        float v = X[i];
        float phi = 0.5f * (1.0f + erff(v * 0.70710678118f));
        float density = expf(-0.5f * v * v) * 0.39894228040f;
        DX[i] = DY[i] * (phi + v * density);
    }
}

/* The sum of the elements of a tensor. */
float mlr_sum(const void *x, int64_t n) {
    const float *X = (const float *)x;
    double total = 0.0;
    for (int64_t i = 0; i < n; i++) {
        total += X[i];
    }
    return (float)total;
}

/* The gradient of the sum: dx is the incoming gradient in every element. */
void mlr_sum_grad(const void *x, float dy, void *dx, int64_t n) {
    float *DX = (float *)dx;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = dy;
    }
    (void)x;
}

/* A copy of a buffer (the gradient of the flatten, whose inverse is the
 * reshape of the same layout). */
void mlr_copy(const void *x, void *out, int64_t n) {
    memcpy(out, x, (size_t)n * 4);
}

/* The gradients of the convolution y[o][y][x] = b[o] +
 * sum(w[o][ky][kx] * x[y+ky-pad][x+kx-pad]).
 *
 * dx[iy][ix] = sum over (o, ky, kx, y, x) of dy[o][y][x] * w[o][ky][kx]
 * dw[o][ky][kx] = sum over (y, x) of dy[o][y][x] * x[y+ky-pad][x+kx-pad]
 * db[o] = sum over (y, x) of dy[o][y][x] */
void mlr_conv2d_grad(const void *x, const void *w, const void *dy, void *dx, void *dw, void *db,
                     int64_t O, int64_t H, int64_t W, int64_t K) {
    const float *X = (const float *)x;
    const float *Wt = (const float *)w;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    float *DW = (float *)dw;
    float *DB = (float *)db;
    int64_t pad = K / 2;
    for (int64_t i = 0; i < H * W; i++) {
        DX[i] = 0.0f;
    }
    for (int64_t o = 0; o < O; o++) {
        DB[o] = 0.0f;
        for (int64_t ky = 0; ky < K; ky++) {
            for (int64_t kx = 0; kx < K; kx++) {
                DW[(o * K + ky) * K + kx] = 0.0f;
            }
        }
    }
    for (int64_t o = 0; o < O; o++) {
        for (int64_t y = 0; y < H; y++) {
            for (int64_t x = 0; x < W; x++) {
                float d = DY[(o * H + y) * W + x];
                DB[o] += d;
                for (int64_t ky = 0; ky < K; ky++) {
                    int64_t iy = y + ky - pad;
                    if (iy < 0 || iy >= H) continue;
                    for (int64_t kx = 0; kx < K; kx++) {
                        int64_t ix = x + kx - pad;
                        if (ix < 0 || ix >= W) continue;
                        DW[(o * K + ky) * K + kx] += d * X[iy * W + ix];
                        DX[iy * W + ix] += d * Wt[o * K * K + ky * K + kx];
                    }
                }
            }
        }
    }
}

/* The gradient of the max pooling: every input position receives the
 * gradient of the window whose maximum it was (the same tie breaking as the
 * forward pass: the first of the greatest). */
void mlr_maxpool2d_grad(const void *x, const void *dy, void *dx, int64_t O, int64_t H, int64_t W,
                        int64_t k) {
    const float *X = (const float *)x;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    int64_t ho = H / k, wo = W / k;
    for (int64_t i = 0; i < O * H * W; i++) {
        DX[i] = 0.0f;
    }
    for (int64_t o = 0; o < O; o++) {
        for (int64_t y = 0; y < ho; y++) {
            for (int64_t x = 0; x < wo; x++) {
                int64_t best_i = -1;
                float best = -3.402823466e+38f;
                for (int64_t ky = 0; ky < k; ky++) {
                    for (int64_t kx = 0; kx < k; kx++) {
                        int64_t i = (o * H + y * k + ky) * W + x * k + kx;
                        if (X[i] > best) {
                            best = X[i];
                            best_i = i;
                        }
                    }
                }
                if (best_i >= 0) {
                    DX[best_i] += DY[(o * ho + y) * wo + x];
                }
            }
        }
    }
}

/* The gradient of the embedding of one token: dtable is zero except for the
 * row of the token, which is the incoming gradient. */
void mlr_embed_grad(int32_t id, const void *dy, void *dtable, int64_t rows, int64_t width) {
    float *T = (float *)dtable;
    const float *DY = (const float *)dy;
    for (int64_t i = 0; i < rows * width; i++) {
        T[i] = 0.0f;
    }
    if (id < 0 || id >= rows) {
        return;
    }
    memcpy(T + (int64_t)id * width, DY, (size_t)width * sizeof(float));
}

/* The gradient of the embedding of a sequence: the gradients of the rows
 * whose token appears more than once are summed. */
void mlr_embed_seq_grad(const void *ids, const void *dy, void *dtable, int64_t rows,
                        int64_t width, int64_t count) {
    const int32_t *I = (const int32_t *)ids;
    const float *DY = (const float *)dy;
    float *T = (float *)dtable;
    for (int64_t i = 0; i < rows * width; i++) {
        T[i] = 0.0f;
    }
    for (int64_t i = 0; i < count; i++) {
        int32_t id = I[i];
        if (id < 0 || id >= rows) {
            continue;
        }
        float *row = T + (int64_t)id * width;
        for (int64_t j = 0; j < width; j++) {
            row[j] += DY[i * width + j];
        }
    }
}

/* The gradient of the layer normalization. The statistics are recomputed
 * from the input:
 *
 *   dx = (g / s) * (dy - mean(dy) - xhat * mean(dy * xhat))
 *   dg = the sum over the rows of dy * xhat
 *   db = the sum over the rows of dy */
void mlr_layernorm_grad(const void *x, const void *gamma, const void *dy, void *dx, void *dg,
                        void *db, int64_t rows, int64_t width) {
    const float *X = (const float *)x;
    const float *G = (const float *)gamma;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    float *DG = (float *)dg;
    float *DB = (float *)db;
    for (int64_t i = 0; i < width; i++) {
        DG[i] = 0.0f;
        DB[i] = 0.0f;
    }
    for (int64_t r = 0; r < rows; r++) {
        const float *row = X + r * width;
        const float *drow = DY + r * width;
        float *dout = DX + r * width;
        double mean = 0.0;
        for (int64_t i = 0; i < width; i++) mean += row[i];
        mean /= (double)width;
        double var = 0.0;
        for (int64_t i = 0; i < width; i++) {
            double d = row[i] - mean;
            var += d * d;
        }
        var /= (double)width;
        float inv = 1.0f / sqrtf((float)(var + 1e-5));
        double sum_dy = 0.0;
        double sum_dy_xhat = 0.0;
        for (int64_t i = 0; i < width; i++) {
            float xhat = (float)(((double)row[i] - mean) * (double)inv);
            // The means are of the gradient scaled by the gamma.
            sum_dy += (double)G[i] * drow[i];
            sum_dy_xhat += (double)G[i] * drow[i] * xhat;
            DG[i] += drow[i] * xhat;
            DB[i] += drow[i];
        }
        float a = (float)(sum_dy / (double)width);
        float b = (float)(sum_dy_xhat / (double)width);
        for (int64_t i = 0; i < width; i++) {
            float xhat = (float)(((double)row[i] - mean) * (double)inv);
            dout[i] = inv * (G[i] * drow[i] - a - xhat * b);
        }
    }
}

/* The gradient of the softmax of the rows: y is the output of the forward
 * pass and dx = y * (dy - sum(dy * y)) for every row. */
void mlr_softmax_grad(const void *y, const void *dy, void *dx, int64_t rows, int64_t cols) {
    const float *Y = (const float *)y;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    for (int64_t r = 0; r < rows; r++) {
        const float *yrow = Y + r * cols;
        const float *drow = DY + r * cols;
        float *out = DX + r * cols;
        double dot = 0.0;
        for (int64_t j = 0; j < cols; j++) {
            dot += (double)drow[j] * yrow[j];
        }
        for (int64_t j = 0; j < cols; j++) {
            out[j] = yrow[j] * (drow[j] - (float)dot);
        }
    }
}

/* The softmax of the rows of the scores with a causal (triangular) mask:
 * the column j of the row r only participates when j <= r. This is the
 * attention of a decoder; the mask lives in the kernel, so that the program
 * has no loops for it. */
void mlr_softmax_tri(const void *scores, void *out, int64_t size) {
    const float *S = (const float *)scores;
    float *Y = (float *)out;
    for (int64_t r = 0; r < size; r++) {
        const float *row = S + r * size;
        float *dst = Y + r * size;
        float best = -3.402823466e+38f;
        for (int64_t j = 0; j <= r; j++) {
            if (row[j] > best) best = row[j];
        }
        double sum = 0.0;
        for (int64_t j = 0; j <= r; j++) {
            float e = expf(row[j] - best);
            dst[j] = e;
            sum += e;
        }
        for (int64_t j = 0; j <= r; j++) {
            dst[j] = (float)(dst[j] / sum);
        }
        for (int64_t j = r + 1; j < size; j++) {
            dst[j] = 0.0f;
        }
    }
}

/* The gradient of the masked softmax: y is the output of the forward pass;
 * the columns beyond the diagonal receive no gradient. */
void mlr_softmax_tri_grad(const void *y, const void *dy, void *dx, int64_t size) {
    const float *Y = (const float *)y;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    for (int64_t r = 0; r < size; r++) {
        const float *yrow = Y + r * size;
        const float *drow = DY + r * size;
        float *out = DX + r * size;
        double dot = 0.0;
        for (int64_t j = 0; j <= r; j++) {
            dot += (double)drow[j] * yrow[j];
        }
        for (int64_t j = 0; j < size; j++) {
            out[j] = j <= r ? yrow[j] * (drow[j] - (float)dot) : 0.0f;
        }
    }
}

/* -------------------------------------------------------------------------
 * Standard library: scalar math.
 * ------------------------------------------------------------------------- */

float mlr_abs(float x) { return fabsf(x); }
float mlr_floor(float x) { return floorf(x); }
float mlr_ceil(float x) { return ceilf(x); }
float mlr_round(float x) { return roundf(x); }
float mlr_sign(float x) { return (x > 0.0f) - (x < 0.0f); }
int32_t mlr_is_nan(float x) { return isnan(x) ? 1 : 0; }
int32_t mlr_is_finite(float x) { return isfinite(x) ? 1 : 0; }
float mlr_pow(float a, float b) { return powf(a, b); }
float mlr_min(float a, float b) { return a < b ? a : b; }
float mlr_max(float a, float b) { return a > b ? a : b; }
float mlr_clamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

/* -------------------------------------------------------------------------
 * Standard library: tensor arithmetic (element wise, shape preserving).
 * ------------------------------------------------------------------------- */

void mlr_tmin(const void *a, const void *b, void *out, int64_t n) {
    const float *A = (const float *)a;
    const float *B = (const float *)b;
    float *O = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        O[i] = A[i] < B[i] ? A[i] : B[i];
    }
}

void mlr_tmax(const void *a, const void *b, void *out, int64_t n) {
    const float *A = (const float *)a;
    const float *B = (const float *)b;
    float *O = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        O[i] = A[i] > B[i] ? A[i] : B[i];
    }
}

void mlr_tclamp(const void *x, float lo, float hi, void *out, int64_t n) {
    const float *X = (const float *)x;
    float *O = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        O[i] = X[i] < lo ? lo : (X[i] > hi ? hi : X[i]);
    }
}

void mlr_tpow(const void *x, float p, void *out, int64_t n) {
    const float *X = (const float *)x;
    float *O = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        O[i] = powf(X[i], p);
    }
}

void mlr_texp(const void *x, void *out, int64_t n) {
    const float *X = (const float *)x;
    float *O = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        O[i] = expf(X[i]);
    }
}

void mlr_tln(const void *x, void *out, int64_t n) {
    const float *X = (const float *)x;
    float *O = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        O[i] = logf(X[i]);
    }
}

/* The gradients of the element wise minimum and maximum: the winner of the
 * comparison receives the incoming gradient (ties go to the first operand,
 * as in PyTorch). */
void mlr_tmin_grad(const void *a, const void *b, const void *dy, void *da, void *db,
                   int64_t n) {
    const float *A = (const float *)a;
    const float *B = (const float *)b;
    const float *DY = (const float *)dy;
    float *DA = (float *)da;
    float *DB = (float *)db;
    for (int64_t i = 0; i < n; i++) {
        if (A[i] <= B[i]) {
            DA[i] = DY[i];
            DB[i] = 0.0f;
        } else {
            DA[i] = 0.0f;
            DB[i] = DY[i];
        }
    }
}

void mlr_tmax_grad(const void *a, const void *b, const void *dy, void *da, void *db,
                   int64_t n) {
    const float *A = (const float *)a;
    const float *B = (const float *)b;
    const float *DY = (const float *)dy;
    float *DA = (float *)da;
    float *DB = (float *)db;
    for (int64_t i = 0; i < n; i++) {
        if (A[i] >= B[i]) {
            DA[i] = DY[i];
            DB[i] = 0.0f;
        } else {
            DA[i] = 0.0f;
            DB[i] = DY[i];
        }
    }
}

void mlr_tclamp_grad(const void *x, float lo, float hi, const void *dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = (X[i] > lo && X[i] < hi) ? DY[i] : 0.0f;
    }
}

void mlr_tpow_grad(const void *x, float p, const void *dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = DY[i] * p * powf(X[i], p - 1.0f);
    }
}

void mlr_texp_grad(const void *x, const void *dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = DY[i] * expf(X[i]);
    }
}

void mlr_tln_grad(const void *x, const void *dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = DY[i] / X[i];
    }
}

/* -------------------------------------------------------------------------
 * Standard library: reductions (tensor -> scalar).
 * ------------------------------------------------------------------------- */

float mlr_mean(const void *x, int64_t n) {
    return mlr_sum(x, n) / (float)(n > 0 ? n : 1);
}

float mlr_tmax_value(const void *x, int64_t n) {
    const float *X = (const float *)x;
    float best = n > 0 ? X[0] : 0.0f;
    for (int64_t i = 1; i < n; i++) {
        if (X[i] > best) best = X[i];
    }
    return best;
}

float mlr_tmin_value(const void *x, int64_t n) {
    const float *X = (const float *)x;
    float best = n > 0 ? X[0] : 0.0f;
    for (int64_t i = 1; i < n; i++) {
        if (X[i] < best) best = X[i];
    }
    return best;
}

int32_t mlr_argmin(const void *x, int64_t n) {
    const float *X = (const float *)x;
    int32_t best = 0;
    for (int64_t i = 1; i < n; i++) {
        if (X[i] < X[best]) best = (int32_t)i;
    }
    return best;
}

float mlr_norm1(const void *x, int64_t n) {
    const float *X = (const float *)x;
    double total = 0.0;
    for (int64_t i = 0; i < n; i++) {
        total += fabsf(X[i]);
    }
    return (float)total;
}

float mlr_norm2(const void *x, int64_t n) {
    const float *X = (const float *)x;
    double total = 0.0;
    for (int64_t i = 0; i < n; i++) {
        total += (double)X[i] * X[i];
    }
    return (float)sqrt(total);
}

float mlr_var(const void *x, int64_t n) {
    const float *X = (const float *)x;
    if (n <= 0) return 0.0f;
    double mean = 0.0;
    for (int64_t i = 0; i < n; i++) mean += X[i];
    mean /= (double)n;
    double total = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = X[i] - mean;
        total += d * d;
    }
    return (float)(total / (double)n);
}

float mlr_std(const void *x, int64_t n) { return sqrtf(mlr_var(x, n)); }

float mlr_dot(const void *a, const void *b, int64_t n) {
    const float *A = (const float *)a;
    const float *B = (const float *)b;
    double total = 0.0;
    for (int64_t i = 0; i < n; i++) {
        total += (double)A[i] * B[i];
    }
    return (float)total;
}

/* The gradient fills the output with the incoming gradient scaled by the
 * other operand (the adjoint of a product reduced by a sum). */
void mlr_scale_fill(const void *other, float dy, void *out, int64_t n) {
    const float *O = (const float *)other;
    float *OUT = (float *)out;
    for (int64_t i = 0; i < n; i++) {
        OUT[i] = dy * O[i];
    }
}

void mlr_tmax_grad_value(const void *x, float dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    float *DX = (float *)dx;
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++) {
        if (X[i] > X[best]) best = i;
    }
    for (int64_t i = 0; i < n; i++) {
        DX[i] = (int64_t)i == best ? dy : 0.0f;
    }
}

void mlr_tmin_grad_value(const void *x, float dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    float *DX = (float *)dx;
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++) {
        if (X[i] < X[best]) best = i;
    }
    for (int64_t i = 0; i < n; i++) {
        DX[i] = (int64_t)i == best ? dy : 0.0f;
    }
}

void mlr_norm1_grad(const void *x, float dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    float *DX = (float *)dx;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = (X[i] > 0.0f ? dy : (X[i] < 0.0f ? -dy : 0.0f));
    }
}

void mlr_norm2_grad(const void *x, float norm, float dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    float *DX = (float *)dx;
    float d = norm > 0.0f ? dy / norm : 0.0f;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = d * X[i];
    }
}

void mlr_var_grad(const void *x, float dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    float *DX = (float *)dx;
    if (n <= 0) return;
    double mean = 0.0;
    for (int64_t i = 0; i < n; i++) mean += X[i];
    mean /= (double)n;
    float c = dy * 2.0f / (float)n;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = c * (X[i] - (float)mean);
    }
}

void mlr_std_grad(const void *x, float dy, void *dx, int64_t n) {
    const float *X = (const float *)x;
    float *DX = (float *)dx;
    if (n <= 0) return;
    double mean = 0.0;
    for (int64_t i = 0; i < n; i++) mean += X[i];
    mean /= (double)n;
    double var = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = X[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    float std = sqrtf((float)var);
    float c = std > 0.0f ? dy / ((float)n * std) : 0.0f;
    for (int64_t i = 0; i < n; i++) {
        DX[i] = c * (X[i] - (float)mean);
    }
}

/* -------------------------------------------------------------------------
 * Standard library: shape operations (copies with a different indexing).
 * ------------------------------------------------------------------------- */

/* The rows [start, start+count) of a tensor whose rows have `rowsize`
 * elements. */
void mlr_rows(const void *x, int64_t start, int64_t count, void *out, int64_t rowsize) {
    const unsigned char *X = (const unsigned char *)x;
    memcpy(out, X + (size_t)start * rowsize * 4, (size_t)count * rowsize * 4);
}

/* The gradient of the rows: the full tensor, zero except for the slice. */
void mlr_rows_grad(const void *dy, int64_t start, int64_t count, void *dx, int64_t rows,
                   int64_t rowsize) {
    unsigned char *DX = (unsigned char *)dx;
    const unsigned char *DY = (const unsigned char *)dy;
    memset(DX, 0, (size_t)rows * rowsize * 4);
    memcpy(DX + (size_t)start * rowsize * 4, DY, (size_t)count * rowsize * 4);
}

/* The concatenation along the first axis: the rows of a, then the rows of b. */
void mlr_concat(const void *a, const void *b, void *out, int64_t rowsa, int64_t rowsb,
                int64_t rowsize) {
    unsigned char *O = (unsigned char *)out;
    memcpy(O, a, (size_t)rowsa * rowsize * 4);
    memcpy(O + (size_t)rowsa * rowsize * 4, b, (size_t)rowsb * rowsize * 4);
}

/* The gradient of the concatenation: each operand receives the rows of its
 * own range. The buffers are filled, not accumulated, because every operand
 * appears exactly once. */
void mlr_concat_grad(const void *dy, int64_t counta, int64_t countb, void *da,
                     void *db, int64_t rowsize) {
    const unsigned char *DY = (const unsigned char *)dy;
    memcpy(da, DY, (size_t)counta * rowsize * 4);
    memcpy(db, DY + (size_t)counta * rowsize * 4, (size_t)countb * rowsize * 4);
}

/* The reverse along the first axis. */
void mlr_reverse(const void *x, void *out, int64_t rows, int64_t rowsize) {
    const unsigned char *X = (const unsigned char *)x;
    unsigned char *O = (unsigned char *)out;
    for (int64_t r = 0; r < rows; r++) {
        memcpy(O + (size_t)(rows - 1 - r) * rowsize * 4, X + (size_t)r * rowsize * 4,
               (size_t)rowsize * 4);
    }
}

/* -------------------------------------------------------------------------
 * Standard library: the shell bridge and the execution environment.
 * ------------------------------------------------------------------------- */

/* Replaces {k} with the k-th argument of `args` (integers in decimal,
 * floats with %.17g); {{ and }} are the escapes of { and }. The caller owns
 * the returned buffer. */
static char *mlr_format_cmd(const char *cmd, int64_t n, const double *args) {
    size_t cap = strlen(cmd) + 64 * (size_t)(n > 0 ? n : 1) + 16;
    char *out = (char *)malloc(cap);
    size_t o = 0;
    for (size_t i = 0; cmd[i]; i++) {
        if (cmd[i] == '{' && cmd[i + 1] == '{') {
            out[o++] = '{';
            i++;
            continue;
        }
        if (cmd[i] == '}' && cmd[i + 1] == '}') {
            out[o++] = '}';
            i++;
            continue;
        }
        if (cmd[i] == '{') {
            int64_t k = 0;
            size_t j = i + 1;
            while (cmd[j] >= '0' && cmd[j] <= '9') {
                k = k * 10 + (cmd[j] - '0');
                j++;
            }
            if (cmd[j] == '}' && k < n) {
                o += (size_t)snprintf(out + o, cap - o, "%.17g", args[k]);
                i = j;
                continue;
            }
        }
        out[o++] = cmd[i];
    }
    out[o] = 0;
    return out;
}

/* Runs the command; kind 0 returns the exit code, kind 1 parses the first
 * integer of the output, kind 2 the first float. The parsed values come
 * back as doubles. */
static double mlr_exec_kind(const char *cmd, int64_t kind, int64_t n, const double *args) {
    char *full = mlr_format_cmd(cmd, n, args);
    if (kind == 0) {
        /* The output of the command is visible on the stderr, so that the
         * pipe stays empty and nothing is lost. */
        size_t len = strlen(full);
        full = (char *)realloc(full, len + 8);
        strcpy(full + len, " 1>&2");
    }
    FILE *f = popen(full, "r");
    if (!f) {
        fprintf(stderr, "mlr: the command cannot be run: %s\n", full);
        free(full);
        exit(1);
    }
    if (kind == 0) {
        int status = pclose(f);
        free(full);
        if (status == -1) {
            fprintf(stderr, "mlr: the command failed\n");
            exit(1);
        }
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
        return 128 + WTERMSIG(status);
    }
    /* The first number of the output. */
    char buffer[4096];
    size_t len = 0;
    int c;
    while ((c = fgetc(f)) != EOF && len + 1 < sizeof(buffer)) {
        buffer[len++] = (char)c;
        /* Enough for a number: stop at the end of the first line. */
        if (c == '\n') break;
    }
    /* The rest of the output is drained, so that the child does not die
     * with SIGPIPE. */
    while (fgetc(f) != EOF) {
    }
    int status = pclose(f);
    free(full);
    buffer[len] = 0;
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        int code = status == -1 ? -1 : (WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        fprintf(stderr, "mlr: the command failed with the exit code %d\n", code);
        exit(1);
    }
    char *end = NULL;
    double value = kind == 1 ? (double)strtol(buffer, &end, 10) : strtod(buffer, &end);
    if (end == buffer) {
        fprintf(stderr, "mlr: the output of the command has no number: %s\n", buffer);
        exit(1);
    }
    return value;
}

int64_t mlr_exec(const char *cmd, int64_t n, const double *args) {
    return (int64_t)mlr_exec_kind(cmd, 0, n, args);
}

int64_t mlr_exec_num(const char *cmd, int64_t kind, int64_t n, const double *args) {
    double v = mlr_exec_kind(cmd, kind, n, args);
    if (kind == 1) {
        /* The integer comes back as its own value. */
        return (int64_t)v;
    }
    /* The float comes back as the bits of the double, which the lowering
     * casts to the result type. */
    union {
        double d;
        int64_t i;
    } u;
    u.d = v;
    return u.i;
}

/* The tensor variant of the exec: the numbers of the output (whitespace
 * separated) fill the buffer; the count must match exactly. */
void mlr_exec_tensor(const char *cmd, int64_t n, const double *args, void *out, int64_t count) {
    char *full = mlr_format_cmd(cmd, n, args);
    FILE *f = popen(full, "r");
    if (!f) {
        fprintf(stderr, "mlr: the command cannot be run: %s\n", full);
        free(full);
        exit(1);
    }
    float *O = (float *)out;
    int64_t i = 0;
    char token[128];
    int t = 0;
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (t > 0) {
                if (i >= count) {
                    fprintf(stderr,
                            "mlr: the output of the command has more than %lld numbers\n",
                            (long long)count);
                    exit(1);
                }
                token[t] = 0;
                O[i++] = strtof(token, NULL);
                t = 0;
            }
            continue;
        }
        if (t + 1 < (int)sizeof(token)) {
            token[t++] = (char)c;
        }
    }
    if (t > 0) {
        if (i >= count) {
            fprintf(stderr, "mlr: the output of the command has more than %lld numbers\n",
                    (long long)count);
            exit(1);
        }
        token[t] = 0;
        O[i++] = strtof(token, NULL);
    }
    int status = pclose(f);
    free(full);
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "mlr: the command failed\n");
        exit(1);
    }
    if (i != count) {
        fprintf(stderr, "mlr: the output of the command has %lld numbers, %lld were expected\n",
                (long long)i, (long long)count);
        exit(1);
    }
}

int64_t mlr_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void mlr_exit(int32_t code) { exit(code); }

void mlr_assert(int32_t cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "mlr: assertion failed: %s\n", msg);
        exit(1);
    }
}

/* -------------------------------------------------------------------------
 * Standard library: creation and randomness (M2). The generator is the same
 * xorshift of `mlr_sample`; the seed is an explicit parameter, so that a
 * program is reproducible.
 * ------------------------------------------------------------------------- */

static uint32_t mlr_rand_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float mlr_rand_unit(uint32_t *state) {
    /* A float in [0, 1). */
    return (float)(mlr_rand_next(state) >> 8) / (float)(1 << 24);
}

/* Fills a tensor with uniform values in [lo, hi). */
void mlr_rand_uniform(void *out, uint32_t seed, float lo, float hi, int64_t n) {
    float *O = (float *)out;
    uint32_t state = seed | 1u;
    for (int64_t i = 0; i < n; i++) {
        O[i] = lo + (hi - lo) * mlr_rand_unit(&state);
    }
}

/* Fills a tensor with normal values (Box–Muller, in pairs). */
void mlr_rand_normal(void *out, uint32_t seed, float mean, float std, int64_t n) {
    float *O = (float *)out;
    uint32_t state = seed | 1u;
    int64_t i = 0;
    while (i < n) {
        float u1 = mlr_rand_unit(&state);
        float u2 = mlr_rand_unit(&state);
        if (u1 <= 0.0f) u1 = 1e-12f;
        float r = sqrtf(-2.0f * logf(u1));
        float angle = 6.28318530718f * u2;
        O[i++] = mean + std * r * cosf(angle);
        if (i < n) {
            O[i++] = mean + std * r * sinf(angle);
        }
    }
}

/* The integers 0..n-1. */
void mlr_iota(void *out, int32_t n) {
    int32_t *O = (int32_t *)out;
    for (int64_t i = 0; i < n; i++) {
        O[i] = (int32_t)i;
    }
}

/* The n values from a to b (both ends included). */
void mlr_linspace(float a, float b, int32_t n, void *out) {
    float *O = (float *)out;
    float step = n > 1 ? (b - a) / (float)(n - 1) : 0.0f;
    for (int64_t i = 0; i < n; i++) {
        O[i] = a + step * (float)i;
    }
}

/* A random permutation of 0..n-1 (Fisher–Yates). */
void mlr_rand_permutation(void *out, uint32_t seed, int32_t n) {
    int32_t *O = (int32_t *)out;
    uint32_t state = seed | 1u;
    for (int64_t i = 0; i < n; i++) {
        O[i] = (int32_t)i;
    }
    for (int64_t i = n - 1; i > 0; i--) {
        int64_t j = (int64_t)(mlr_rand_next(&state) % (uint32_t)(i + 1));
        int32_t t = O[i];
        O[i] = O[j];
        O[j] = t;
    }
}

/* The rows of a table that the index tensor selects (the generalization of
 * the embedding of a sequence to any row count). */
void mlr_gather(const void *t, const void *idx, void *out, int64_t rows, int64_t count,
                int64_t rowsize) {
    const unsigned char *T = (const unsigned char *)t;
    const int32_t *I = (const int32_t *)idx;
    unsigned char *O = (unsigned char *)out;
    for (int64_t i = 0; i < count; i++) {
        int32_t id = I[i];
        if (id < 0 || id >= rows) {
            memset(O + (size_t)i * rowsize * 4, 0, (size_t)rowsize * 4);
            continue;
        }
        memcpy(O + (size_t)i * rowsize * 4, T + (size_t)id * rowsize * 4,
               (size_t)rowsize * 4);
    }
}

/* The gradient of the gather: the values are added to the rows that were
 * selected (the table gradient starts at zero). */
void mlr_gather_grad(const void *idx, const void *dy, void *dtable, int64_t rows,
                     int64_t count, int64_t rowsize) {
    const int32_t *I = (const int32_t *)idx;
    const float *DY = (const float *)dy;
    float *T = (float *)dtable;
    for (int64_t i = 0; i < rows * rowsize; i++) {
        T[i] = 0.0f;
    }
    for (int64_t i = 0; i < count; i++) {
        int32_t id = I[i];
        if (id < 0 || id >= rows) {
            continue;
        }
        float *row = T + (int64_t)id * rowsize;
        for (int64_t j = 0; j < rowsize; j++) {
            row[j] += DY[i * rowsize + j];
        }
    }
}

/* A one-hot matrix: the row i has a one at the column of the token. */
void mlr_onehot(const void *ids, void *out, int32_t count, int32_t vocab) {
    const int32_t *I = (const int32_t *)ids;
    float *O = (float *)out;
    for (int64_t i = 0; i < count * vocab; i++) {
        O[i] = 0.0f;
    }
    for (int64_t i = 0; i < count; i++) {
        int32_t id = I[i];
        if (id >= 0 && id < vocab) {
            O[i * vocab + id] = 1.0f;
        }
    }
}

/* The accuracy of the logits: the share of the rows whose argmax is the
 * label. */
float mlr_accuracy(const void *logits, const void *labels, int64_t rows, int64_t vocab) {
    const float *L = (const float *)logits;
    const int32_t *Y = (const int32_t *)labels;
    int64_t correct = 0;
    for (int64_t r = 0; r < rows; r++) {
        const float *row = L + r * vocab;
        int32_t best = 0;
        for (int64_t j = 1; j < vocab; j++) {
            if (row[j] > row[best]) best = (int32_t)j;
        }
        if (best == Y[r]) correct++;
    }
    return rows > 0 ? (float)correct / (float)rows : 0.0f;
}

/* The dropout of the training: every element is zeroed with the probability
 * `rate` and the survivors are scaled by 1/(1-rate). */
void mlr_dropout(const void *x, uint32_t seed, float rate, void *out, void *mask, int64_t n) {
    const float *X = (const float *)x;
    float *O = (float *)out;
    float *M = (float *)mask;
    uint32_t state = seed | 1u;
    float keep = 1.0f - rate;
    float scale = keep > 0.0f ? 1.0f / keep : 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float u = mlr_rand_unit(&state);
        float keep_it = u >= rate;
        M[i] = keep_it ? scale : 0.0f;
        O[i] = X[i] * M[i];
    }
}

/* The gradient of the dropout: the mask is recomputed with the same seed
 * and rate as the forward pass, and applied to the incoming gradient. */
void mlr_dropout_grad(const void *x, const void *dy, uint32_t seed, float rate, void *dx,
                      int64_t n) {
    const float *X = (const float *)x;
    const float *DY = (const float *)dy;
    float *DX = (float *)dx;
    uint32_t state = seed | 1u;
    float keep = 1.0f - rate;
    float scale = keep > 0.0f ? 1.0f / keep : 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float u = mlr_rand_unit(&state);
        float keep_it = u >= rate;
        DX[i] = keep_it ? DY[i] * scale : 0.0f;
        (void)X;
    }
}
