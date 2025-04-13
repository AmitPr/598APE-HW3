#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "immintrin.h"

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
    if (posix_memalign((void**)&mass, 64, size)) {
      fprintf(stderr, "Aligned allocation failed\n");
      exit(1);
    }

    if (posix_memalign((void**)&x, 64, size)) {
      fprintf(stderr, "Aligned allocation failed\n");
      exit(1);
    }

    if (posix_memalign((void**)&y, 64, size)) {
      fprintf(stderr, "Aligned allocation failed\n");
      exit(1);
    }

    if (posix_memalign((void**)&vx, 64, size)) {
      fprintf(stderr, "Aligned allocation failed\n");
      exit(1);
    }

    if (posix_memalign((void**)&vy, 64, size)) {
      fprintf(stderr, "Aligned allocation failed\n");
      exit(1);
    }
  }

  ~World() { free(mass); }
};

unsigned long long seed = 100;

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

int nplanets;
int timesteps;
double dt;
double G;

void simulate(World& world) {
  __m512d dt_vec = _mm512_set1_pd(dt);
  __m512d epsilon = _mm512_set1_pd(0.0001);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nplanets; i++) {
    __m512d x_i = _mm512_set1_pd(world.x[i]);
    __m512d y_i = _mm512_set1_pd(world.y[i]);
    __m512d mass_i = _mm512_set1_pd(world.mass[i]);
    __m512d acc_x = _mm512_setzero_pd();
    __m512d acc_y = _mm512_setzero_pd();
    for (int j = 0; j < nplanets; j += 8) {
      __m512d dx = _mm512_sub_pd(_mm512_load_pd(world.x + j), x_i);
      __m512d dy = _mm512_sub_pd(_mm512_load_pd(world.y + j), y_i);
      __m512d distSqr =
          _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, epsilon));
      __m512d invDist =
          _mm512_div_pd(_mm512_mul_pd(mass_i, _mm512_load_pd(world.mass + j)),
                        _mm512_sqrt_pd(distSqr));
      __m512d invDist3 =
          _mm512_mul_pd(_mm512_mul_pd(invDist, invDist), invDist);
      // acc = acc + dx*invDist3
      acc_x = _mm512_fmadd_pd(dx, invDist3, acc_x);
      acc_y = _mm512_fmadd_pd(dy, invDist3, acc_y);
    }
    world.vx[i] += dt * _mm512_reduce_add_pd(acc_x);
    world.vy[i] += dt * _mm512_reduce_add_pd(acc_y);
  }
  for (int i = 0; i < nplanets; i += 8) {
    // p = p + v*dt
    __m512d x = _mm512_fmadd_pd(dt_vec, _mm512_load_pd(world.vx + i),
                                _mm512_load_pd(world.x + i));
    __m512d y = _mm512_fmadd_pd(dt_vec, _mm512_load_pd(world.vy + i),
                                _mm512_load_pd(world.y + i));
    _mm512_store_pd(world.x + i, x);
    _mm512_store_pd(world.y + i, y);

    // printf("%0.16f %0.16f ", world.x[i], world.y[i]);
  }
  // printf("\n");
}

int main(int argc, const char** argv) {
  if (argc < 2) {
    printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
    return 1;
  }
  nplanets = atoi(argv[1]);
  timesteps = atoi(argv[2]);
  dt = 0.001;
  G = 6.6743;

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
  for (int i = 0; i < timesteps; i++) simulate(world);
  gettimeofday(&end, NULL);
  printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
         tdiff(&start, &end), world.x[nplanets - 1], world.y[nplanets - 1]);

  return 0;
}