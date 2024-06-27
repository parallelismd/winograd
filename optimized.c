#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// added
#include "kblas.h"
#include <time.h>
#include <mpi.h>
#include <linux/time.h>
#include <arm_neon.h>

#define TIMING

#ifdef TIMING
struct timespec __start_t;

#define SET_TIME \
  clock_gettime(CLOCK_MONOTONIC, &__start_t);

#define PRTT                                                                                                \
  {                                                                                                         \
    struct timespec __end_t;                                                                                \
    clock_gettime(CLOCK_MONOTONIC, &__end_t);                                                               \
    long __time = (__end_t.tv_sec - __start_t.tv_sec) * 1000000000 + (__end_t.tv_nsec - __start_t.tv_nsec); \
    printf("+%.5fms\n", __time / 1000000.0);                                                                \
  }

#define PRTTM(format, message...)                                                                           \
  {                                                                                                         \
    struct timespec __end_t;                                                                                \
    clock_gettime(CLOCK_MONOTONIC, &__end_t);                                                               \
    long __time = (__end_t.tv_sec - __start_t.tv_sec) * 1000000000 + (__end_t.tv_nsec - __start_t.tv_nsec); \
    printf("+%.5fms " format, __time / 1000000.0, message);                                                 \
  }

#else
#define SET_TIME
#define PRTT
#define PRTTM
#endif

const float G[4][3] = {
    {1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0.0, 0.0, 1.0}};
const float G_T[3][4] = {
    {1, 0.5, 0.5, 0.0}, {0.0, 0.5, -0.5, 0.0}, {0.0, 0.5, 0.5, 1.0}};
const float B[4][4] = {
    {1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
const float B_T[4][4] = {
    {1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
const float A[4][2] = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};
const float A_T[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};

// Matrix Multiplication: Out = A x B (A:M*K, B:K*N, out: M*N)
// All arrays should have their memory prepared correctly outside this function
// For rookies: this sgemm is the worst sgemm I've ever written throughout my
// career.
//      If you don't know where to start, optimize this function as a good
//      starting point.
void sgemm(const float *A, const float *B, float *out, const int M, const int K, const int N)
{
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        out[i * N + j] += A[i * K + k] * B[k * N + j];
  // case all the matrix multi using this func is small and relatively const
  // let compiler do the optimization
}
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
#define M_BLOCKING 8
#define N_BLOCKING 64
#define K_BLOCKING 16


void sgemm__(float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K)
{
  int m_count, n_count, k_count;
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

// User API for winograd F(2,3)
// image: [batch * C * inHeight * inWidth]
// filter: [K * C * 3 * 3]
// result: [batch * K * outHeight * outWidth]
void winconv_2x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int C, float *__restrict__ filter,
                 const int K, const int N, float *__restrict__ out,
                 float *__restrict__ U, float *__restrict__ V,
                 float *__restrict__ M)
{
  // m = 2; r = 3; alpha = 4
  const int outHeight = inHeight - 2;
  const int outWidth = inWidth - 2;
  const long sizeI = inHeight * inWidth;
  const int sizeF = 3 * 3;
  const int sizeO = outHeight * outWidth;
  const long P = outHeight / 2 * outWidth / 2 * N;
  printf("sizeI: %ld, sizeF: %d, sizeO: %d, P: %ld, N: %d,C: %d, K: %d, parallel: %d %d\n", sizeI, sizeF, sizeO, P, N, C, K, BlasGetParallel(), BlasGetNumThreads());

  SET_TIME;

  float tmp_u[12]; // 4 * 3
  float u[16];     // 4 * 4;
  // U[:, :, k, c] = G * filters[k, c, :, :] * G.T()
#pragma omp parallel for collapse(2) private(tmp_u, u)
  for (int k = 0; k < K; ++k)
  {
    for (int c = 0; c < C; ++c)
    {
      float *filters_ptr = filter + (k * C + c) * sizeF;
      sgemm(&G[0][0], filters_ptr, tmp_u, 4, 3, 3); // TODO G is a const 4 * 3 matrix. consider not use sgemm
      sgemm(tmp_u, &G_T[0][0], u, 4, 3, 4);         // TODO consider merge these two sgemm because G is a const 4 * 3 matrix
      for (int xi = 0; xi < 4; ++xi)
        for (int nu = 0; nu < 4; ++nu)
          U[((xi * 4 + nu) * K + k) * C + c] = u[xi * 4 + nu]; // TODO the memory access is not continuous
    }
  }

  PRTTM("stage 1 complete\n", "");
  // V[:, :, c, p] = B_T * image[c, b, :, :] * B
  float tmp_v[16];
  float d[16]; // d: [4 * 4];
  float v[16]; // v: [4 * 4];
#pragma omp parallel for collapse(4) private(tmp_v, d, v)
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    {
      for (int y = 0; y < outHeight / 2; ++y)
      {
        for (int x = 0; x < outWidth / 2; ++x)
        {

          // Generate d_cb
          for (int iy = 0; iy < 4; ++iy)
            for (int ix = 0; ix < 4; ++ix)
              d[iy * 4 + ix] = image[(n * C + c) * sizeI +
                                     (y * 2 + iy) * inWidth + (x * 2 + ix)]; // TODO the memory access is not continuous
          sgemm(&B_T[0][0], d, tmp_v, 4, 4, 4);
          sgemm(tmp_v, &B[0][0], v, 4, 4, 4); // TODO consider merge these two sgemm because B is a const 4 * 4 matrix
          int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
          for (int xi = 0; xi < 4; ++xi)
            for (int nu = 0; nu < 4; ++nu)
              V[((long)(xi * 4 + nu) * C + c) * P + b] = v[xi * 4 + nu]; // TODO the memory access is not continuous
        }
      }
    }

  PRTTM("stage 2 complete\n", "");
  // TODO try to completely rewrite the last part
  //  M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
  int m_count, n_count, k_count;
#pragma omp parallel for collapse(3) private(m_count, n_count, k_count)
  for (int xi = 0; xi < 4; ++xi)
  {
    for (int nu = 0; nu < 4; ++nu)
    {
      float *M_ptr = M + (long)(xi * 4 + nu) * K * P;
      float *U_ptr = U + (long)(xi * 4 + nu) * K * C;
      float *V_ptr = V + (long)(xi * 4 + nu) * C * P;

      for (m_count = 0; m_count < K; m_count += M_BLOCKING)
      {
        for (k_count = 0; k_count < C; k_count += K_BLOCKING)
        {
          for (n_count = 0; n_count < P; n_count += N_BLOCKING)
          {
            block_mul(M_BLOCKING, N_BLOCKING, K_BLOCKING, &U_ptr[(m_count)*C + (k_count)], &V_ptr[(k_count)*P + (n_count)], &M_ptr[(m_count)*P + (n_count)], K, C, P);
          }
        }
      }
      // sgemm__(U_ptr, V_ptr, M_ptr, K, C, P); // TODO this is the big gemm M K N
      PRTTM("stage 3 big gemm %d %d %d from %d / %d\n", K, C, P, omp_get_thread_num(), omp_get_num_threads());
    }
  }

  PRTTM("stage 3 big gemm complete %d %d %d %d %d %d\n", M_BLOCKING, K_BLOCKING, N_BLOCKING, K / M_BLOCKING, C / K_BLOCKING, P / N_BLOCKING);
  // Y = A_T * m * A
  float mm[16];      // 4 * 4
  float tmp_m[8];    // 2 * 4
  float temp_out[4]; // 2 * 2

#pragma omp parallel for collapse(4) private(mm, temp_out, tmp_m)
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
    {
      for (int y = 0; y < outHeight / 2; ++y)
      {
        for (int x = 0; x < outWidth / 2; ++x)
        {
          int b = (n * outHeight / 2 + y) * outWidth / 2 + x;
          for (long xi = 0; xi < 4; ++xi)
          {
            for (long nu = 0; nu < 4; ++nu)
            {
              mm[xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
            }
          }
          sgemm(&A_T[0][0], mm, tmp_m, 2, 4, 4);
          sgemm(tmp_m, &A[0][0], temp_out, 2, 4, 2); // TODO consider merge these two sgemm because A is a const 2 * 4 matrix
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
              out[(long)((n * K + k) * outHeight + y * 2 + i) * outWidth + x * 2 +
                  j] = temp_out[i * 2 + j];
        }
      }
    }
  PRTTM("stage 4 complete N=%ld,K=%ld,outHeight=%ld,outWidth=%ld\n", N, K, outHeight, outWidth);
}
