#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <cstring>

#define A(i, j) A[(i) * N + (j)]
#define B(i, j) B[(i) * K + (j)]
#define C(i, j) C[(i) * K + (j)]

void block_mul(const uint64_t sizeM, const uint64_t sizeN, const uint64_t sizeK, float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K)
{
    int i, j;
    float *ptr_packing_a = A, *ptr_packing_b = B;
    float32x4_t a0, a1, a2, a3, b0, b1;
    for (i = 0; i < sizeM; i += 4)
    {
        ptr_packing_b = B;
        for (j = 0; j < sizeN; j += 8)
        {
            ptr_packing_a = A + i * sizeK;

            float32x4_t c00 = vld1q_f32(&C(i + 0, j + 0));
            float32x4_t c01 = vld1q_f32(&C(i + 0, j + 4));
            float32x4_t c10 = vld1q_f32(&C(i + 1, j + 0));
            float32x4_t c11 = vld1q_f32(&C(i + 1, j + 4));
            float32x4_t c20 = vld1q_f32(&C(i + 2, j + 0));
            float32x4_t c21 = vld1q_f32(&C(i + 2, j + 4));
            float32x4_t c30 = vld1q_f32(&C(i + 3, j + 0));
            float32x4_t c31 = vld1q_f32(&C(i + 3, j + 4));

            for (int k = 0; k < sizeK; k++)
            {

                a0 = vdupq_n_f32(ptr_packing_a[0]);
                a1 = vdupq_n_f32(ptr_packing_a[1]);
                a2 = vdupq_n_f32(ptr_packing_a[2]);
                a3 = vdupq_n_f32(ptr_packing_a[3]);

                b0 = vld1q_f32(ptr_packing_b + 0);
                b1 = vld1q_f32(ptr_packing_b + 4);

                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                c30 = vmlaq_f32(c30, a3, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                c11 = vmlaq_f32(c11, a1, b1);
                c21 = vmlaq_f32(c21, a2, b1);
                c31 = vmlaq_f32(c31, a3, b1);

                ptr_packing_a += 4;
                ptr_packing_b += 8;
            }

            vst1q_f32(&C(i + 0, j + 0), c00);
            vst1q_f32(&C(i + 1, j + 0), c10);
            vst1q_f32(&C(i + 2, j + 0), c20);
            vst1q_f32(&C(i + 3, j + 0), c30);
            vst1q_f32(&C(i + 0, j + 4), c01);
            vst1q_f32(&C(i + 1, j + 4), c11);
            vst1q_f32(&C(i + 2, j + 4), c21);
            vst1q_f32(&C(i + 3, j + 4), c31);
        }
    }
}
#define M_BLOCKING 8
#define N_BLOCKING 16
#define K_BLOCKING 16

void pack_a(float *a, float *a_block, const uint64_t n2)
{
    float *tosrc1, *tosrc2, *tosrc3, *tosrc4, *tosrc5, *todst = a_block;
    int rows, cols;
    for (cols = 0; cols < M_BLOCKING; cols += 4)
    {
        tosrc1 = a + cols * n2;
        tosrc2 = tosrc1 + n2;
        tosrc3 = tosrc2 + n2;
        tosrc4 = tosrc3 + n2;
        tosrc5 = tosrc4 + n2;
        for (rows = 0; rows < K_BLOCKING; rows++)
        {
            *todst = *tosrc1;
            tosrc1++;
            todst++;
            *todst = *tosrc2;
            tosrc2++;
            todst++;
            *todst = *tosrc3;
            tosrc3++;
            todst++;
            *todst = *tosrc4;
            tosrc4++;
            todst++;
        }
    }
}

void pack_b(float *b, float *b_block, const uint64_t n3)
{
    float *tosrc, *todst = b_block;
    for (auto count = 0; count < N_BLOCKING; count += 8)
    {
        tosrc = b + count;
        for (auto count_second = 0; count_second < K_BLOCKING; count_second++)
        {
            vst1_f32_x4(todst, vld1_f32_x4(tosrc));
            vst1_f32_x4(todst + 4, vld1_f32_x4(tosrc + 4));
            tosrc += n3;
            todst += 8;
        }
    }
}


float *a_block, *b_block;

void mul(float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K)
{
    int m_count, n_count, k_count;
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    a_block = (float *)aligned_alloc(1024, M_BLOCKING * K_BLOCKING * sizeof(float));
    b_block = (float *)aligned_alloc(2048, K_BLOCKING * N_BLOCKING * sizeof(float) * 2);
    for (m_count = 0; m_count < M; m_count += M_BLOCKING)
    {
        for (k_count = 0; k_count < N; k_count += K_BLOCKING)
        {
            pack_a(&A(m_count, k_count), a_block, N);
            for (n_count = 0; n_count < K; n_count += N_BLOCKING)
            {
                pack_b(&B(k_count, n_count), b_block, K);
                block_mul(M_BLOCKING, N_BLOCKING, K_BLOCKING, a_block, b_block, &C(m_count, n_count), M, N, K);
            }
        }
    }
    free(a_block);
    free(b_block);
}

int main()
{
    uint64_t n1, n2, n3;
    FILE *fi;

    fi = fopen("conf.data", "rb");
    fread(&n1, 1, 8, fi);
    fread(&n2, 1, 8, fi);
    fread(&n3, 1, 8, fi);

    printf("n1=%d, n2=%d, n3=%d\n", n1, n2, n3);

    float *a = (float *)aligned_alloc(64, n1 * n2 * 8);
    float *b = (float *)aligned_alloc(64, n2 * n3 * 8);
    float *c = (float *)aligned_alloc(64, n1 * n3 * 8);
    fread(a, 1, n1 * n2 * 8, fi);
    fread(b, 1, n2 * n3 * 8, fi);
    fclose(fi);

    memset(c, 0, n1 * n3 * 8);

    auto t1 = std::chrono::steady_clock::now();
    mul(a, b, c, n1, n2, n3);
    auto t2 = std::chrono::steady_clock::now();
    int d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("%dus\n", d1);

    fi = fopen("out.data.tr", "wb");
    fwrite(c, 1, n1 * n3 * 8, fi);
    fclose(fi);

    free(a);
    free(b);
    free(c);

    return 0;
}