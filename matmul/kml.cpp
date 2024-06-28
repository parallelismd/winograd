#include <iostream>
#include <chrono>
#include "kblas.h"

void mul(float* a, float* b, float* c, uint64_t n1, uint64_t n2, uint64_t n3) {

  int M = n1, N = n3, K = n2;

  const int lda = K, ldb = N, ldc = N;
  
  float alpha = 1.0, beta = 2.0;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc);
}

int main() {
 uint64_t n1, n2, n3;
 FILE* fi;

 fi = fopen("conf.data", "rb");
 fread(&n1, 1, 8, fi);
 fread(&n2, 1, 8, fi);
 fread(&n3, 1, 8, fi);

 float* a = (float*)malloc(n1 * n2 * 8);
 float* b = (float*)malloc(n2 * n3 * 8);
 float* c = (float*)malloc(n1 * n3 * 8);

 fread(a, 1, n1 * n2 * 8, fi);
 fread(b, 1, n2 * n3 * 8, fi);
 fclose(fi);

 for (uint64_t i = 0; i < n1; i++) {
  for (uint64_t k = 0; k < n3; k++) {
   c[i * n3 + k] = 0;
  }
 }

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

/*

clang++ kml.cpp -O3 -I/shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/include -mcpu=tsv110 -L/shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/locking -lkblas  -o kml

*/