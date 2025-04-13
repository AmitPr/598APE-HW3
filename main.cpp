#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define CACHE_LOOKAHEAD 64

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

    memset(mass, 0, sizeof(double) * nplanets);
    memset(x, 0, sizeof(double) * nplanets);
    memset(y, 0, sizeof(double) * nplanets);
    memset(vx, 0, sizeof(double) * nplanets);
    memset(vy, 0, sizeof(double) * nplanets);
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

#define set1(a) _mm256_set1_pd(a)
#define zero() _mm256_setzero_pd()
#define mul(a, b) _mm256_mul_pd(a, b)
#define add(a, b) _mm256_add_pd(a, b)
#define sub(a, b) _mm256_sub_pd(a, b)
#define sqrt(a) _mm256_sqrt_pd(a)
#define rsqrt(a) _mm256_rsqrt14_pd(a)
#define div(a, b) _mm256_div_pd(a, b)
#define load(a) _mm256_load_pd(a)
#define store(a, b) _mm256_store_pd(a, b)
#define fmadd(a, b, c) _mm256_fmadd_pd(a, b, c)
// #define reduce_add(a) _mm512_reduce_add_pd(a)
#define reduce_add(a) _mm256_reduce_add_pd(a)

inline double _mm256_reduce_add_pd(__m256d v) {
  // v[:128] + v[128:]
  __m128d vlow = _mm256_castpd256_pd128(v);
  __m128d vhigh = _mm256_extractf128_pd(v, 1);
  vlow = _mm_add_pd(vlow, vhigh);

  // vlow[:64] + vlow[64:]
  __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
  return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}

void simulate(World& world) {
  __m256d dt_vec = set1(dt);
  __m256d epsilon = set1(0.0001);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nplanets; i++) {
    const __m256d x_i = set1(world.x[i]);
    const __m256d y_i = set1(world.y[i]);
    const __m256d mass_i = set1(world.mass[i]);
    __m256d acc_x = zero();
    __m256d acc_y = zero();
    for (int j = 0; j < nplanets; j += 4) {
      const __m256d dx = sub(load(world.x + j), x_i);
      const __m256d dy = sub(load(world.y + j), y_i);
      const __m256d distSqr = fmadd(dx, dx, fmadd(dy, dy, epsilon));
#ifdef EXACT
      const __m256d invDist =
          div(mul(mass_i, load(world.mass + j)), sqrt(distSqr));
#else
      const __m256d invDist =
          mul(mul(mass_i, load(world.mass + j)), rsqrt(distSqr));
#endif
      const __m256d invDist3 = mul(mul(invDist, invDist), invDist);
      // acc = acc + dx*invDist3
      acc_x = fmadd(dx, invDist3, acc_x);
      acc_y = fmadd(dy, invDist3, acc_y);
    }
    world.vx[i] += dt * reduce_add(acc_x);
    world.vy[i] += dt * reduce_add(acc_y);
  }

  for (int i = 0; i < nplanets; i += 4) {
    // p = p + v*dt
    const __m256d x = fmadd(dt_vec, load(world.vx + i), load(world.x + i));
    const __m256d y = fmadd(dt_vec, load(world.vy + i), load(world.y + i));
    store(world.x + i, x);
    store(world.y + i, y);

    printf("%0.16f %0.16f ", world.x[i], world.y[i]);
  }
  printf("\n");
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
  for (int i = 0; i < timesteps; i++) simulate(world);
  gettimeofday(&end, NULL);
  printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
         tdiff(&start, &end), world.x[nplanets - 1], world.y[nplanets - 1]);

  return 0;
}