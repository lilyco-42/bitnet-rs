// bitnet_ref.cpp — 从 bitnet.cpp 原版提取的参考实现（无 ggml 依赖）
// 对拍用：读 input.bin（f32 权重 + f32 激活），写 output.bin（打包字节+scale+点积结果）
// 函数体与 src/ggml-bitnet-mad.cpp 完全一致（NEON 路径，QK_I2_S=64）

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define QK_I2_S 64

// ---- quantize_i2_s (NEON 分支, nrow=1) ----
static size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row) {
    size_t row_size = (size_t)(n_per_row / 4 + 4);
    int n = (int)(nrow * n_per_row);

    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, (size_t)(n * sizeof(uint8_t) / 4));

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / QK_I2_S; i++) {
        for (int j = 0; j < QK_I2_S; j++) {
            int group_idx = j / 16;
            int group_pos = j % 16;
            uint8_t temp = (uint8_t)(q8[i * QK_I2_S + j] << (6 - 2 * group_idx));
            i2_weight[i * 16 + group_pos] |= temp;
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = (float)i2_scale;

    free(q8);
    return nrow * row_size / 4 + 32;
}

// ---- ggml_vec_dot_i2_i8_s_1x1 (NEON 分支) ----
static void vec_dot_1x1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const uint8x16_t mask = vdupq_n_u8(3);

    for (int row = 0; row < nrc; row++) {
        int32x4_t accu = vdupq_n_s32(0);
        const uint8_t * x_row = x + row * bx / 4;

        for (int i=0; i < group32_num; i++) {
            int16x8_t accu32 = vdupq_n_s16(0);
            for (int j=0; j < 32; j++) {
                uint8x16_t xq8_3 = vld1q_u8(x_row + i * 32 * 16 + j * 16);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                const int8x16_t yq8_0 = vld1q_s8(y + i * 32 * 64 + j * 64 + 0);
                const int8x16_t yq8_1 = vld1q_s8(y + i * 32 * 64 + j * 64 + 16);
                const int8x16_t yq8_2 = vld1q_s8(y + i * 32 * 64 + j * 64 + 32);
                const int8x16_t yq8_3 = vld1q_s8(y + i * 32 * 64 + j * 64 + 48);

                accu32 = vmlal_s8(accu32, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            }
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu32)));
            accu = vaddq_s32(accu, vmovl_high_s16(accu32));
        }

        for (int i = 0; i < groupla_num; i++){
            int16x8_t accula = vdupq_n_s16(0);
            for (int j = 0; j < la_num; j++) {
                uint8x16_t xq8_3 = vld1q_u8(x_row + group32_num * 32 * 16 + j * 16);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                const int8x16_t yq8_0 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 0);
                const int8x16_t yq8_1 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 16);
                const int8x16_t yq8_2 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 32);
                const int8x16_t yq8_3 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 48);

                accula = vmlal_s8(accula, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accula = vmlal_s8(accula, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accula = vmlal_s8(accula, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accula = vmlal_s8(accula, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accula = vmlal_s8(accula, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accula = vmlal_s8(accula, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accula = vmlal_s8(accula, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accula = vmlal_s8(accula, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            }
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accula)));
            accu = vaddq_s32(accu, vmovl_high_s16(accula));
        }
        int sumi = vaddlvq_s32(accu);
        s[row] = (float)sumi;
    }
}

// ---- 入口 ----
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <n_weights> <n_activ> <input.bin> <output.bin>\n", argv[0]); return 1; }
    int n_weights = atoi(argv[1]);
    int n_activ   = atoi(argv[2]);
    FILE* fin = fopen(argv[3], "rb");
    if (!fin) { perror("open input"); return 1; }

    // 布局：n_weights 个 f32 权重，然后 n_activ 个 f32 激活
    float* w = (float*)malloc(sizeof(float) * n_weights);
    float* a = (float*)malloc(sizeof(float) * n_activ);
    fread(w, sizeof(float), n_weights, fin);
    fread(a, sizeof(float), n_activ, fin);
    fclose(fin);

    // 量化（nrow=1）
    size_t nbytes = (size_t)(n_weights / 4 + 4);
    uint8_t* packed = (uint8_t*)calloc(1, nbytes + 32);
    quantize_i2_s(w, packed, 1, n_weights);

    // 激活 f32 -> i8
    int8_t* y = (int8_t*)malloc(n_activ);
    float y_scale = 0;
    for (int i = 0; i < n_activ; i++) y_scale = fmaxf(y_scale, fabsf(a[i]));
    if (y_scale == 0) y_scale = 1;
    float inv = 127.0f / y_scale;
    for (int i = 0; i < n_activ; i++) y[i] = (int8_t)lrintf(a[i] * inv);

    // 点积（1x1，nrc=1，要求 n_weights == n_activ）
    float dot = 0;
    vec_dot_1x1(n_weights, &dot, 4, packed, n_weights / 4, y, n_activ, 1);

    FILE* fout = fopen(argv[4], "wb");
    fwrite(packed, 1, n_weights / 4, fout);       // 打包字节
    float scale = *(float*)(packed + n_weights / 4);
    fwrite(&scale, sizeof(float), 1, fout);        // 量化 scale
    fwrite(&y_scale, sizeof(float), 1, fout);      // 激活 scale
    fwrite(&dot, sizeof(float), 1, fout);          // 整数点积结果
    fclose(fout);

    free(w); free(a); free(packed); free(y);
    return 0;
}
