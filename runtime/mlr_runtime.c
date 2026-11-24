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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mlr_print_i32(int x) { printf("%d\n", x); }
void mlr_print_i64(long long x) { printf("%lld\n", x); }
void mlr_print_f32(float x) { printf("%f\n", (double)x); }
void mlr_print_f64(double x) { printf("%f\n", x); }
void mlr_print_bool(bool x) { printf("%s\n", x ? "true" : "false"); }

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
