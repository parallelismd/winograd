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

// 2 * 4
void block_mul(const uint64_t sizeM, const uint64_t sizeN, const uint64_t sizeK, float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K)
{
  float32x4_t a0, a1, a2, a3, b0, b1, b2, b3;
  for (int i = 0; i < sizeM; i += 4)
  {
    for (int j = 0; j < sizeN; j += 8)
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
        // b2 = vld1q_f32(B(k, j + 8));
        // b3 = vld1q_f32(B(k, j + 12));

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
      vst1q_f32(&C(i + 0, j + 4), c01);
      vst1q_f32(&C(i + 1, j + 0), c10);
      vst1q_f32(&C(i + 1, j + 4), c11);
      vst1q_f32(&C(i + 2, j + 0), c20);
      vst1q_f32(&C(i + 2, j + 4), c21);
      vst1q_f32(&C(i + 3, j + 0), c30);
      vst1q_f32(&C(i + 3, j + 4), c31);
    }
  }
}

#define M_BLOCKING 32
#define N_BLOCKING 16
#define K_BLOCKING 512

void sgemm__(float *A, float *B, float *C, const uint64_t M, const uint64_t K, const uint64_t N)
{
  int mb_id, nb_id, kb_id;
  for (mb_id = 0; mb_id < M; mb_id += M_BLOCKING)
  {
    for (kb_id = 0; kb_id < K; kb_id += K_BLOCKING)
    {
      for (nb_id = 0; nb_id < N; nb_id += N_BLOCKING)
      {
        block_mul(M_BLOCKING, N_BLOCKING, K_BLOCKING, &A(mb_id, kb_id), &B(kb_id, nb_id), &C(mb_id, nb_id), M, K, N);
      }
    }
  }
}

void sgemm_parallel(const float *A, const float *B, float *out, const int M, const int K, const int N)
{
  for (int i = 0; i < M; i += 2)
  {
    for (int j = 0; j < N; j += 4)
    {
      float32x4_t c00_vec = vdupq_n_f32(0.0);
      float32x4_t c01_vec = vdupq_n_f32(0.0);
      float32x4_t c10_vec = vdupq_n_f32(0.0);
      float32x4_t c11_vec = vdupq_n_f32(0.0);

      for (int k = 0; k < K; k += 4)
      {
        // Load A matrix elements
        float32x4_t a0_vec = vld1q_f32(A + i * K + k);
        float32x4_t a1_vec = vld1q_f32(A + (i + 1) * K + k);

        // Load B matrix elements and process 4 columns
        float32x4_t b_vec = vld1q_f32(B + k * N + j);
        c00_vec = vfmaq_laneq_f32(c00_vec, b_vec, a0_vec, 0);
        c01_vec = vfmaq_laneq_f32(c01_vec, b_vec, a1_vec, 0);
        c10_vec = vfmaq_laneq_f32(c10_vec, b_vec, a0_vec, 0);
        c11_vec = vfmaq_laneq_f32(c11_vec, b_vec, a1_vec, 0);

        b_vec = vld1q_f32(B + k * N + j + N);
        c00_vec = vfmaq_laneq_f32(c00_vec, b_vec, a0_vec, 1);
        c01_vec = vfmaq_laneq_f32(c01_vec, b_vec, a1_vec, 1);
        c10_vec = vfmaq_laneq_f32(c10_vec, b_vec, a0_vec, 1);
        c11_vec = vfmaq_laneq_f32(c11_vec, b_vec, a1_vec, 1);

        b_vec = vld1q_f32(B + k * N + j + 2 * N);
        c00_vec = vfmaq_laneq_f32(c00_vec, b_vec, a0_vec, 2);
        c01_vec = vfmaq_laneq_f32(c01_vec, b_vec, a1_vec, 2);
        c10_vec = vfmaq_laneq_f32(c10_vec, b_vec, a0_vec, 2);
        c11_vec = vfmaq_laneq_f32(c11_vec, b_vec, a1_vec, 2);

        b_vec = vld1q_f32(B + k * N + j + 3 * N);
        c00_vec = vfmaq_laneq_f32(c00_vec, b_vec, a0_vec, 3);
        c01_vec = vfmaq_laneq_f32(c01_vec, b_vec, a1_vec, 3);
        c10_vec = vfmaq_laneq_f32(c10_vec, b_vec, a0_vec, 3);
        c11_vec = vfmaq_laneq_f32(c11_vec, b_vec, a1_vec, 3);
      }

      // Store results back to memory
      vst1q_f32(out + i * N + j, c00_vec);
      vst1q_f32(out + i * N + j + N, c01_vec);
      vst1q_f32(out + (i + 1) * N + j, c10_vec);
      vst1q_f32(out + (i + 1) * N + j + N, c11_vec);
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
#pragma omp parallel for collapse(2)
  for (int xi = 0; xi < 4; ++xi)
  {
    for (int nu = 0; nu < 4; ++nu)
    {
      float *M_ptr = M + (long)(xi * 4 + nu) * K * P;
      float *U_ptr = U + (long)(xi * 4 + nu) * K * C;
      float *V_ptr = V + (long)(xi * 4 + nu) * C * P;
      sgemm__(U_ptr, V_ptr, M_ptr, K, C, P); // TODO this is the big gemm
      PRTTM("stage 3 big gemm %d %d %d from %d\n", K, C, P, omp_get_thread_num());
    }
  }

  PRTTM("stage 3 big gemm complete\n", "");
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
