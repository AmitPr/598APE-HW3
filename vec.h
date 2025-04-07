#ifndef VEC_H
#define VEC_H
#include <immintrin.h>

struct alignas(16) Vec2 {
  union {
    __m128d vec;
    struct alignas(16) {
      double x, y;
    };
  };

  explicit Vec2(__m128d v) : vec(v) {}
  Vec2() : vec(_mm_setzero_pd()) {}
  Vec2(double x_, double y_) : vec(_mm_set_pd(y_, x_)) {
    static_assert(sizeof(Vec2) == 16, "Vec2 must be 16 bytes aligned");
  }

  Vec2 operator+(const Vec2& other) const {
    return Vec2{_mm_add_pd(vec, other.vec)};
  }
  void operator+=(const Vec2& other) { vec = _mm_add_pd(vec, other.vec); }
  Vec2 operator-(const Vec2& other) const {
    return Vec2{_mm_sub_pd(vec, other.vec)};
  }
  void operator-=(const Vec2& other) { vec = _mm_sub_pd(vec, other.vec); }
  Vec2 operator*(double scalar) const {
    return Vec2{_mm_mul_pd(vec, _mm_set1_pd(scalar))};
  }
  Vec2 operator/(double scalar) const {
    return Vec2{_mm_div_pd(vec, _mm_set1_pd(scalar))};
  }

  double mag2() const {
    __m128d sq = _mm_mul_pd(vec, vec);
    // https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
    __m128 undef = _mm_undefined_ps();
    __m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(sq));
    __m128d shuf = _mm_castps_pd(shuftmp);
    return _mm_cvtsd_f64(_mm_add_sd(sq, shuf));
  }
};

inline Vec2 operator*(double scalar, const Vec2& v) {
  return Vec2{_mm_mul_pd(v.vec, _mm_set1_pd(scalar))};
}
inline Vec2 operator/(const Vec2& v, double scalar) {
  return Vec2{_mm_div_pd(v.vec, _mm_set1_pd(scalar))};
}
inline Vec2 operator-(const Vec2& v) {
  return Vec2{_mm_sub_pd(_mm_setzero_pd(), v.vec)};
}

#endif