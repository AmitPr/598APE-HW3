#ifndef VEC_H
#define VEC_H
#include <immintrin.h>
#ifdef AVX512
#define set1(a) _mm512_set1_pd(a)
#define zero() _mm512_setzero_pd()
#define mul(a, b) _mm512_mul_pd(a, b)
#define add(a, b) _mm512_add_pd(a, b)
#define sub(a, b) _mm512_sub_pd(a, b)
#define vsqrt(a) _mm512_sqrt_pd(a)
#define rsqrt(a) _mm512_rsqrt14_pd(a)
#define div(a, b) _mm512_div_pd(a, b)
#define load(a) _mm512_load_pd(a)
#define store(a, b) _mm512_store_pd(a, b)
#define fmadd(a, b, c) _mm512_fmadd_pd(a, b, c)
#define reduce_add(a) _mm512_reduce_add_pd(a)
typedef __m512d vec;
#define WIDTH 8
#else
#define set1(a) _mm256_set1_pd(a)
#define zero() _mm256_setzero_pd()
#define mul(a, b) _mm256_mul_pd(a, b)
#define add(a, b) _mm256_add_pd(a, b)
#define sub(a, b) _mm256_sub_pd(a, b)
#define vsqrt(a) _mm256_sqrt_pd(a)
#define rsqrt(a) _mm256_rsqrt14_pd(a)
#define div(a, b) _mm256_div_pd(a, b)
#define load(a) _mm256_load_pd(a)
#define store(a, b) _mm256_store_pd(a, b)
#define fmadd(a, b, c) _mm256_fmadd_pd(a, b, c)
#define reduce_add(a) _mm256_reduce_add_pd(a)
typedef __m256d vec;
#define WIDTH 4

inline double _mm256_reduce_add_pd(vec v) {
  // v[:128] + v[128:]
  __m128d vlow = _mm256_castpd256_pd128(v);
  __m128d vhigh = _mm256_extractf128_pd(v, 1);
  vlow = _mm_add_pd(vlow, vhigh);

  // vlow[:64] + vlow[64:]
  __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
  return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}
#endif

#endif  // VEC_H