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
  float32x4_t a0, a1, a2, a3, b0, b1, b2, b3;
  for (i = 0; i < sizeM; i += 4)
  {
    for (j = 0; j < sizeN; j += 16)
    {
      float32x4_t c00 = vld1q_f32(&C(i + 0, j + 0));
      float32x4_t c01 = vld1q_f32(&C(i + 0, j + 4));
      float32x4_t c10 = vld1q_f32(&C(i + 1, j + 0));
      float32x4_t c11 = vld1q_f32(&C(i + 1, j + 4));
      float32x4_t c20 = vld1q_f32(&C(i + 2, j + 0));
      float32x4_t c21 = vld1q_f32(&C(i + 2, j + 4));
      float32x4_t c30 = vld1q_f32(&C(i + 3, j + 0));
      float32x4_t c31 = vld1q_f32(&C(i + 3, j + 4));
      float32x4_t c02 = vld1q_f32(&C(i + 0, j + 8));
      float32x4_t c03 = vld1q_f32(&C(i + 0, j + 12));
      float32x4_t c12 = vld1q_f32(&C(i + 1, j + 8));
      float32x4_t c13 = vld1q_f32(&C(i + 1, j + 12));
      float32x4_t c22 = vld1q_f32(&C(i + 2, j + 8));
      float32x4_t c23 = vld1q_f32(&C(i + 2, j + 12));
      float32x4_t c32 = vld1q_f32(&C(i + 3, j + 8));
      float32x4_t c33 = vld1q_f32(&C(i + 3, j + 12));


      for (int k = 0; k < sizeK; k++)
      {
        a0 = vdupq_n_f32(A(i + 0, k));
        a1 = vdupq_n_f32(A(i + 1, k));
        a2 = vdupq_n_f32(A(i + 2, k));
        a3 = vdupq_n_f32(A(i + 3, k));
    

        b0 = vld1q_f32(&B(k, j + 0));
        b1 = vld1q_f32(&B(k, j + 4));
        b2 = vld1q_f32(&B(k, j + 8));
        b3 = vld1q_f32(&B(k, j + 12));
 
        c00 = vmlaq_f32(c00, a0, b0);
        c10 = vmlaq_f32(c10, a1, b0);
        c20 = vmlaq_f32(c20, a2, b0);
        c30 = vmlaq_f32(c30, a3, b0);
        c01 = vmlaq_f32(c01, a0, b1);
        c11 = vmlaq_f32(c11, a1, b1);
        c21 = vmlaq_f32(c21, a2, b1);
        c31 = vmlaq_f32(c31, a3, b1);
        
        c02 = vmlaq_f32(c02, a0, b2);
        c12 = vmlaq_f32(c12, a1, b2);
        c22 = vmlaq_f32(c22, a2, b2);
        c32 = vmlaq_f32(c32, a3, b2);
        c03 = vmlaq_f32(c03, a0, b3);
        c13 = vmlaq_f32(c13, a1, b3);
        c23 = vmlaq_f32(c23, a2, b3);
        c33 = vmlaq_f32(c33, a3, b3);

      }

      vst1q_f32(&C(i + 0, j + 0), c00);
      vst1q_f32(&C(i + 1, j + 0), c10);
      vst1q_f32(&C(i + 2, j + 0), c20);
      vst1q_f32(&C(i + 3, j + 0), c30);

      vst1q_f32(&C(i + 0, j + 4), c01);
      vst1q_f32(&C(i + 1, j + 4), c11);
      vst1q_f32(&C(i + 2, j + 4), c21);
      vst1q_f32(&C(i + 3, j + 4), c31);

      vst1q_f32(&C(i + 0, j + 8), c02);
      vst1q_f32(&C(i + 1, j + 8), c12);
      vst1q_f32(&C(i + 2, j + 8), c22);
      vst1q_f32(&C(i + 3, j + 8), c32);
      
      vst1q_f32(&C(i + 0, j + 12), c03);
      vst1q_f32(&C(i + 1, j + 12), c13);
      vst1q_f32(&C(i + 2, j + 12), c23);
      vst1q_f32(&C(i + 3, j + 12), c33);
      
    }
  }
}
#define M_BLOCKING 16
#define N_BLOCKING 64
#define K_BLOCKING 16


void mul(float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K)
{
    int m_count, n_count, k_count;
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    #pragma omp parallel for private(m_count, n_count, k_count) collapse(1)
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
    int d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("%dus\n", d1);

    fi = fopen("out.data", "wb");
    fwrite(c, 1, n1 * n3 * 8, fi);
    fclose(fi);

    return 0;
}