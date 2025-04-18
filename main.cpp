#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Use AVX512 instead of AVX2:
// #define AVX512
#define VECTOR_THRESHOLD 152

#include "vec.h"

int nplanets;
int timesteps;
const double dt = 0.001;
const double G = 6.6743;
unsigned long long seed = 100;

float tdiff(struct timeval* start, struct timeval* end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

struct World {
  double* __restrict__ mass;
  double* __restrict__ x;
  double* __restrict__ y;
  double* __restrict__ vx;
  double* __restrict__ vy;

  World(int n) {
    // round n up to the next multiple of 8, for AVX alignment
    n = (n + 7) & ~7;
    int size = sizeof(double) * n;
    int ret = posix_memalign((void**)&mass, 64, size);
    ret |= posix_memalign((void**)&x, 64, size);
    ret |= posix_memalign((void**)&y, 64, size);
    ret |= posix_memalign((void**)&vx, 64, size);
    ret |= posix_memalign((void**)&vy, 64, size);
    if (ret) {
      fprintf(stderr, "Aligned allocation failed\n");
      exit(1);
    }

    memset(mass, 0, size);
    memset(x, 0, size);
    memset(y, 0, size);
    memset(vx, 0, size);
    memset(vy, 0, size);
  }

  ~World() {
    free(mass);
    free(x);
    free(y);
    free(vx);
    free(vy);
  }
};

unsigned long long randomU64() {
  seed ^= (seed << 21);
  seed ^= (seed >> 35);
  seed ^= (seed << 4);
  return seed;
}

double randomDouble() {
  unsigned long long next = randomU64();
  next >>= (64 - 26);
  unsigned long long next2 = randomU64();
  next2 >>= (64 - 26);
  return ((next << 27) + next2) / (double)(1LL << 53);
}

void simulate(World& world) {
  vec dt_vec = set1(dt);
  vec epsilon = set1(0.0001);
#pragma omp parallel for schedule(static) if (nplanets > 32)
  for (int i = 0; i < nplanets; i += 2) {
    const vec x_i = set1(world.x[i]);
    const vec y_i = set1(world.y[i]);
    const vec mass_i = set1(world.mass[i]);
    vec acc_x = zero();
    vec acc_y = zero();

    const vec x_i2 = set1(world.x[i + 1]);
    const vec y_i2 = set1(world.y[i + 1]);
    const vec mass_i2 = set1(world.mass[i + 1]);
    vec acc_x2 = zero();
    vec acc_y2 = zero();

    for (int j = 0; j < nplanets; j += WIDTH) {
      const vec dx = sub(load(world.x + j), x_i);
      const vec dy = sub(load(world.y + j), y_i);
      const vec distSqr = fmadd(dx, dx, fmadd(dy, dy, epsilon));
#ifdef EXACT
      const vec inv = div(mul(mass_i, load(world.mass + j)), vsqrt(distSqr));
#else
      const vec inv = mul(mul(mass_i, load(world.mass + j)), rsqrt(distSqr));
#endif

      const vec dx2 = sub(load(world.x + j), x_i2);
      const vec dy2 = sub(load(world.y + j), y_i2);
      const vec distSqr2 = fmadd(dx2, dx2, fmadd(dy2, dy2, epsilon));
#ifdef EXACT
      const vec inv2 = div(mul(mass_i2, load(world.mass + j)), vsqrt(distSqr2));
#else
      const vec inv2 = mul(mul(mass_i2, load(world.mass + j)), rsqrt(distSqr2));
#endif
      const vec invDist3 = mul(mul(inv, inv), inv);
      // acc = acc + dx*invDist3
      acc_x = fmadd(dx, invDist3, acc_x);
      acc_y = fmadd(dy, invDist3, acc_y);

      const vec invDist32 = mul(mul(inv2, inv2), inv2);
      acc_x2 = fmadd(dx2, invDist32, acc_x2);
      acc_y2 = fmadd(dy2, invDist32, acc_y2);
    }
    world.vx[i] += dt * reduce_add(acc_x);
    world.vy[i] += dt * reduce_add(acc_y);
    world.vx[i + 1] += dt * reduce_add(acc_x2);
    world.vy[i + 1] += dt * reduce_add(acc_y2);
  }

  for (int i = 0; i < nplanets; i += WIDTH) {
    // p = p + v*dt
    const vec x = fmadd(dt_vec, load(world.vx + i), load(world.x + i));
    const vec y = fmadd(dt_vec, load(world.vy + i), load(world.y + i));
    store(world.x + i, x);
    store(world.y + i, y);
  }
}

void simulate_sequential(World& world) {
  vec dt_vec = set1(dt);
  vec epsilon = set1(0.0001);
  for (int i = 0; i < nplanets; i += 2) {
    const vec x_i = set1(world.x[i]);
    const vec y_i = set1(world.y[i]);
    const vec mass_i = set1(world.mass[i]);
    vec acc_x = zero();
    vec acc_y = zero();

    const vec x_i2 = set1(world.x[i + 1]);
    const vec y_i2 = set1(world.y[i + 1]);
    const vec mass_i2 = set1(world.mass[i + 1]);
    vec acc_x2 = zero();
    vec acc_y2 = zero();

    for (int j = 0; j < nplanets; j += WIDTH) {
      const vec dx = sub(load(world.x + j), x_i);
      const vec dy = sub(load(world.y + j), y_i);
      const vec distSqr = fmadd(dx, dx, fmadd(dy, dy, epsilon));
#ifdef EXACT
      const vec inv = div(mul(mass_i, load(world.mass + j)), vsqrt(distSqr));
#else
      const vec inv = mul(mul(mass_i, load(world.mass + j)), rsqrt(distSqr));
#endif

      const vec dx2 = sub(load(world.x + j), x_i2);
      const vec dy2 = sub(load(world.y + j), y_i2);
      const vec distSqr2 = fmadd(dx2, dx2, fmadd(dy2, dy2, epsilon));
#ifdef EXACT
      const vec inv2 = div(mul(mass_i2, load(world.mass + j)), vsqrt(distSqr2));
#else
      const vec inv2 = mul(mul(mass_i2, load(world.mass + j)), rsqrt(distSqr2));
#endif
      const vec invDist3 = mul(mul(inv, inv), inv);
      // acc = acc + dx*invDist3
      acc_x = fmadd(dx, invDist3, acc_x);
      acc_y = fmadd(dy, invDist3, acc_y);

      const vec invDist32 = mul(mul(inv2, inv2), inv2);
      acc_x2 = fmadd(dx2, invDist32, acc_x2);
      acc_y2 = fmadd(dy2, invDist32, acc_y2);
    }
    world.vx[i] += dt * reduce_add(acc_x);
    world.vy[i] += dt * reduce_add(acc_y);
    world.vx[i + 1] += dt * reduce_add(acc_x2);
    world.vy[i + 1] += dt * reduce_add(acc_y2);
  }

  for (int i = 0; i < nplanets; i += WIDTH) {
    // p = p + v*dt
    const vec x = fmadd(dt_vec, load(world.vx + i), load(world.x + i));
    const vec y = fmadd(dt_vec, load(world.vy + i), load(world.y + i));
    store(world.x + i, x);
    store(world.y + i, y);
  }
}

int main(int argc, const char** argv) {
  if (argc < 2) {
    printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
    return 1;
  }
  nplanets = atoi(argv[1]);
  timesteps = atoi(argv[2]);

  World world(nplanets);
  for (int i = 0; i < nplanets; i++) {
    world.mass[i] = randomDouble() * 10 + 0.2;
    world.x[i] = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
    world.y[i] = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
    world.vx[i] = randomDouble() * 5 - 2.5;
    world.vy[i] = randomDouble() * 5 - 2.5;
  }

  struct timeval start, end;
  gettimeofday(&start, NULL);
  if (nplanets <= VECTOR_THRESHOLD) {
    for (int i = 0; i < timesteps; i++) simulate_sequential(world);
  } else {
    for (int i = 0; i < timesteps; i++) simulate(world);
  }
  gettimeofday(&end, NULL);
  printf(
      "Total time to run simulation %0.6f seconds, final location %f "
      "%f\n",
      tdiff(&start, &end), world.x[nplanets - 1], world.y[nplanets - 1]);

  return 0;
}