#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define CACHE_LOOKAHEAD 64
// #define AVX512

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
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nplanets; i++) {
    const vec x_i = set1(world.x[i]);
    const vec y_i = set1(world.y[i]);
    const vec mass_i = set1(world.mass[i]);
    vec acc_x = zero();
    vec acc_y = zero();
    for (int j = 0; j < nplanets; j += WIDTH) {
      const vec dx = sub(load(world.x + j), x_i);
      const vec dy = sub(load(world.y + j), y_i);
      const vec distSqr = fmadd(dx, dx, fmadd(dy, dy, epsilon));
#ifdef EXACT
      const vec inv = div(mul(mass_i, load(world.mass + j)), sqrt(distSqr));
#else
      const vec inv = mul(mul(mass_i, load(world.mass + j)), rsqrt(distSqr));
#endif
      const vec invDist3 = mul(mul(inv, inv), inv);
      // acc = acc + dx*invDist3
      acc_x = fmadd(dx, invDist3, acc_x);
      acc_y = fmadd(dy, invDist3, acc_y);
    }
    world.vx[i] += dt * reduce_add(acc_x);
    world.vy[i] += dt * reduce_add(acc_y);
  }

  for (int i = 0; i < nplanets; i += WIDTH) {
    // p = p + v*dt
    const vec x = fmadd(dt_vec, load(world.vx + i), load(world.x + i));
    const vec y = fmadd(dt_vec, load(world.vy + i), load(world.y + i));
    store(world.x + i, x);
    store(world.y + i, y);
  }
  // for (int i = 0; i < nplanets; i+=16)
  //   printf("%0.16f %0.16f ", world.x[i], world.y[i]);
  // printf("\n");
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
    // if (i%16 ==0) printf("%0.16f %0.16f ", world.x[i], world.y[i]);
  }
  // printf("\n");

  struct timeval start, end;
  gettimeofday(&start, NULL);
  for (int i = 0; i < timesteps; i++) simulate(world);
  gettimeofday(&end, NULL);
  printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
         tdiff(&start, &end), world.x[nplanets - 1], world.y[nplanets - 1]);

  return 0;
}