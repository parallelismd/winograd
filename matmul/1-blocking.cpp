#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <cstring>

#define A(i, j) A[(i) * N + (j)]
#define B(i, j) B[(i) * K + (j)]
#define C(i, j) C[(i) * K + (j)]

void block_mul(const uint64_t sizeM, const uint64_t sizeN, const uint64_t sizeK, float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K)
{
  int i,j;
  float32x4_t a0, a1, a2, a3, b0, b1;
  for (i = 0; i < sizeM; i += 4)
  {
    for (j = 0; j < sizeN; j += 8)
    {
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
        a0 = vdupq_n_f32(A(i + 0, k));
        a1 = vdupq_n_f32(A(i + 1, k));
        a2 = vdupq_n_f32(A(i + 2, k));
        a3 = vdupq_n_f32(A(i + 3, k));

        b0 = vld1q_f32(&B(k, j + 0));
        b1 = vld1q_f32(&B(k, j + 4));
 
        c00 = vmlaq_f32(c00, a0, b0);
        c10 = vmlaq_f32(c10, a1, b0);
        c20 = vmlaq_f32(c20, a2, b0);
        c30 = vmlaq_f32(c30, a3, b0);
        c01 = vmlaq_f32(c01, a0, b1);
        c11 = vmlaq_f32(c11, a1, b1);
        c21 = vmlaq_f32(c21, a2, b1);
        c31 = vmlaq_f32(c31, a3, b1);

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
#define M_BLOCKING 16
#define N_BLOCKING 16
#define K_BLOCKING 16


void mul(float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K)
{
    int m_count, n_count, k_count;
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    #pragma omp parallel for private(m_count, n_count, k_count)
    for (m_count = 0; m_count < M; m_count += M_BLOCKING)
    {
        for (k_count = 0; k_count < N; k_count += K_BLOCKING)
        {
            for (n_count = 0; n_count < K; n_count += N_BLOCKING)
            {
                block_mul(M_BLOCKING, N_BLOCKING, K_BLOCKING, &A(m_count, k_count), &B(k_count, n_count), &C(m_count, n_count), M, N, K);
            }
        }
    }
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
    int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%dms\n", d1);

    fi = fopen("out.data", "wb");
    fwrite(c, 1, n1 * n3 * 8, fi);
    fclose(fi);

    return 0;
}