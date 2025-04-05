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

  Vec2 operator+(const Vec2& other) const {
    return Vec2{x + other.x, y + other.y};
  }
  void operator+=(const Vec2& other) {
    x += other.x;
    y += other.y;
  }
  Vec2 operator-(const Vec2& other) const {
    return Vec2{x - other.x, y - other.y};
  }
  void operator-=(const Vec2& other) {
    x -= other.x;
    y -= other.y;
  }
  Vec2 operator*(double scalar) const { return Vec2{x * scalar, y * scalar}; }
  Vec2 operator/(double scalar) const { return Vec2{x / scalar, y / scalar}; }

  double mag2() const { return x * x + y * y; }
};

#endif