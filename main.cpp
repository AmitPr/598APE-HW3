#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

float tdiff(struct timeval* start, struct timeval* end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

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
const double dt = 0.001;
const double G = 6.6743;

struct alignas(32) Vec2 {
  double x, y;
};

void simulate(const double* __restrict__ mass, Vec2* __restrict__ pos,
              Vec2* __restrict__ vel) {
  for (int t = 0; t < timesteps; t++) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nplanets; i++) {
      for (int j = 0; j < nplanets; j++) {
        const double dx = pos[j].x - pos[i].x;
        const double dy = pos[j].y - pos[i].y;
        const double distSqr = dx * dx + dy * dy + 0.0001;
        const double invDist = mass[i] * mass[j] / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        vel[i].x += dt * dx * invDist3;
        vel[i].y += dt * dy * invDist3;
      }
    }

    // Update positions
    for (int i = 0; i < nplanets; i++) {
      pos[i].x += dt * vel[i].x;
      pos[i].y += dt * vel[i].y;
    }
  }
}

int main(int argc, const char** argv) {
  if (argc < 2) {
    printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
    return 1;
  }
  nplanets = atoi(argv[1]);
  timesteps = atoi(argv[2]);

  double* mass = (double*)malloc(sizeof(double) * nplanets);
  Vec2* pos = (Vec2*)malloc(sizeof(Vec2) * nplanets);
  Vec2* vel = (Vec2*)malloc(sizeof(Vec2) * nplanets);
  for (int i = 0; i < nplanets; i++) {
    mass[i] = randomDouble() * 10 + 0.2;
    pos[i].x = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
    pos[i].y = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
    vel[i].x = randomDouble() * 5 - 2.5;
    vel[i].y = randomDouble() * 5 - 2.5;
  }

  struct timeval start, end;
  gettimeofday(&start, NULL);
  simulate(mass, pos, vel);
  gettimeofday(&end, NULL);
  printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
         tdiff(&start, &end), pos[nplanets - 1].x, pos[nplanets - 1].y);

  return 0;
}