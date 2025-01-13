#include <immintrin.h>  
#include <cstring>     

#define A(i, p) a[(i)*lda + (p)]
#define B(p, j) b[(p)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

template <unsigned regsA, unsigned regsB>
void matmul_dot_inner(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    __m256 csum[regsA][regsB];  
    memset(csum, 0, sizeof(csum));  

    for (int p = 0; p < k; ++p) {
        // Load and compute dot products
        for (int bi = 0; bi < regsB; ++bi) {
            __m256 bb = _mm256_loadu_ps(&B(p, bi * 8));  
            for (int ai = 0; ai < regsA; ++ai) {
                __m256 aa = _mm256_broadcast_ss(&A(ai, p)); 
                csum[ai][bi] = _mm256_fmadd_ps(aa, bb, csum[ai][bi]);  
            }
        }
    }

    // Store results back to matrix C
    for (int ai = 0; ai < regsA; ++ai) {
        for (int bi = 0; bi < regsB; ++bi) {
            __m256 prev_c = _mm256_loadu_ps(&C(ai, bi * 8));  
            __m256 new_c = _mm256_add_ps(prev_c, csum[ai][bi]);  
            _mm256_storeu_ps(&C(ai, bi * 8), new_c);  
        }
    }
}

int main() {
    const int m = 8, n = 8, k = 8;
    float a[m * k], b[k * n], c[m * n];

    // Initialize matrices A, B, and C
    for (int i = 0; i < m * k; ++i) a[i] = 1.0f;
    for (int i = 0; i < k * n; ++i) b[i] = 2.0f;
    memset(c, 0, sizeof(c));

    // Perform matrix multiplication
    matmul_dot_inner<1, 1>(k, a, k, b, n, c, n);

    // Print the resulting matrix C
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.1f ", c[i * n + j]);
        }
        printf("\n");
    }
    return 0;
}
