#include <iostream>
#include <chrono>
#include <arm_sve.h>
#include <cstring>

#define A(i, j) A[(i) * N + (j)]
#define B(i, j) B[(i) * K + (j)]
#define C(i, j) C[(i) * K + (j)]

#define M_BLOCKING 16
#define N_BLOCKING 32
#define K_BLOCKING 16

void block_mul(const uint64_t sizeM, const uint64_t sizeN, const uint64_t sizeK, float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K){
    for(int i=0;i<sizeM;i=i+4){
        for(int j=0;j<sizeN;j=j+16){
            svbool_t pg = svwhilelt_b32(0,7);

            svfloat32_t c00 = svld1(pg,&C(i+0,j+0));
            svfloat32_t c01 = svld1(pg,&C(i+0,j+8));
            svfloat32_t c10 = svld1(pg,&C(i+1,j+0));
            svfloat32_t c11 = svld1(pg,&C(i+1,j+8));
            svfloat32_t c20 = svld1(pg,&C(i+2,j+0));
            svfloat32_t c21 = svld1(pg,&C(i+2,j+8));
            svfloat32_t c30 = svld1(pg,&C(i+3,j+0));
            svfloat32_t c31 = svld1(pg,&C(i+3,j+8));
            for(int k=0;k<sizeK;k++){
                svfloat32_t a0 = svdup_n_f32(A(i+0,k));
                svfloat32_t a1 = svdup_n_f32(A(i+1,k));
                svfloat32_t a2 = svdup_n_f32(A(i+2,k));
                svfloat32_t a3 = svdup_n_f32(A(i+3,k));

                svfloat32_t b0 = svld1(pg,&B(k,j+0));
                svfloat32_t b1 = svld1(pg,&B(k,j+8));

                c00 = svmla_f32_m(pg,c00,a0,b0);
                c01 = svmla_f32_m(pg,c01,a0,b1);
                c10 = svmla_f32_m(pg,c10,a1,b0);
                c11 = svmla_f32_m(pg,c11,a1,b1);
                c20 = svmla_f32_m(pg,c20,a2,b0);
                c21 = svmla_f32_m(pg,c21,a2,b1);
                c30 = svmla_f32_m(pg,c30,a3,b0);
                c31 = svmla_f32_m(pg,c31,a3,b1);
            }
            svst1(pg,&C(i+0,j+0),c00);
            svst1(pg,&C(i+0,j+8),c01);
            svst1(pg,&C(i+1,j+0),c10);
            svst1(pg,&C(i+1,j+8),c11);
            svst1(pg,&C(i+2,j+0),c20);
            svst1(pg,&C(i+2,j+8),c21);
            svst1(pg,&C(i+3,j+0),c30);
            svst1(pg,&C(i+3,j+8),c31);
        }
    }
}

void mul(float *A, float *B, float *C, const uint64_t M, const uint64_t N, const uint64_t K){
    int m_count, n_count, k_count;
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    #pragma omp parallel for collapse(3) private(m_count, n_count, k_count)
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