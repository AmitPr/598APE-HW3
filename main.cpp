#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

float tdiff(struct timeval* start, struct timeval* end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

struct World {
  double* mass;
  double* x;
  double* y;
  double* vx;
  double* vy;

  World(int n) {
    mass = (double*)malloc(sizeof(double) * n);
    x = (double*)malloc(sizeof(double) * n);
    y = (double*)malloc(sizeof(double) * n);
    vx = (double*)malloc(sizeof(double) * n);
    vy = (double*)malloc(sizeof(double) * n);
  }
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
  for (int t = 0; t < timesteps; t++) {
    for (int i = 0; i < nplanets; i++) {
      for (int j = 0; j < nplanets; j++) {
        double dx = world.x[j] - world.x[i];
        double dy = world.y[j] - world.y[i];
        double distSqr = dx * dx + dy * dy + 0.0001;
        double invDist = world.mass[i] * world.mass[j] / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        world.vx[i] += dt * dx * invDist3;
        world.vy[i] += dt * dy * invDist3;
      }
      world.x[i] += dt * world.vx[i];
      world.y[i] += dt * world.vy[i];
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
  simulate(world);
  gettimeofday(&end, NULL);
  printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
         tdiff(&start, &end), world.x[nplanets - 1], world.y[nplanets - 1]);

  return 0;
}