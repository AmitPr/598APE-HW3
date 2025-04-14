#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [N-Body Simulator: Optimizations],
  abstract: [
    This paper-report describes the performance analysis and optimization of a n-body simulator, with a focus on optimizing to exploit low-level details of the system architecture. We achieve a 27x speedup over the baseline implementation for scenes with large numbers of bodies, whilst providing a fallback serial simulator for smaller scenes, which remains performant compared to the baseline.

    Artifacts hosted on GitHub: #link("https://github.com/AmitPr/598APE-HW3")
  ],
  authors: (
    (
      name: "Amit Prasad",
      email: "amitp4",
    ),
  ),

  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
In this report, we describe our overall approach to optimizng an N-Body simulator in two regimes: Large and Small body counts. Some duplicate information from previous reports are present regarding tools and approach (`hyperfine`, `valgrind`, and `perf`) along with our methods for defining correctness and verifying that optimized artifacts continue to work as expected. Next, we broadly discuss our optimizations, and then the results we see. We conclude with a discussion of alternate techniques and methods that we did not implement in the final iteration, and future avenues of exploration.

= Methods And Tooling <sec::methods>
In this section, we discuss the high-level methods and tooling applied to optimize the symbolic regression program, whilst maintaining correctness. Note that much of this remains identical to the ray-tracing assignment. The optimizations were developed and tested single-handedly.

== Benchmarking and Profiling
To measure executable runtime, we use hyperfine@hyperfine_2025, a CLI benchmarking tool. To profile executions, we used the `perf` tool, with issues due to our system being a VM. We use a *Large Scene* with 1000 bodies over 5000 timesteps, and a *Small Scene* with 16 bodies over 10 million timesteps.

== Correctness
To prevent memory leaks, we used Valgrind @nethercote2007valgrind (with the `--leak-check=full` flag).

As small perturbations in N-Body simulations can blowup over large timescales, we used the following to preserve correctness:
1. Calculate error as distance at each timestep between baseline and optimized bodies.
3. Graph the resultant error over time, ensuring that the error remains low early in the simulation.

Over larger timescales, this error inevitably blows up, but we want to ensure that the error initially grows slowly as in @fig::errors

#figure(
  image("error.png", width: 90%),
  caption: [
    This graph shows the error over time for a simulation with 1000 bodies. The error starts extremely small, growing over time.
  ],
  placement: auto,
)<fig::errors>

= Implementation <sec::implementation>
In this section, we discuss the implementation of our optimizations, and the speedup achieved by each compared to previous versions, and the baseline. The data is presented in @table::runtimes.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, horizon, horizon, horizon, horizon),
    table.header(
      [*Optimization*], [*Small Scene Runtime*],  [*Speedup (Cumulative)*], [*Large Scene Runtime*], [*Speedup (Cumulative)*],
    ),
    [Baseline],[13.847s],[-],[26.259s],[-],
    [Compiler Flags],[11.922s],[1.16x #text(gray)[(1.16x)]],[22.926s],[1.14x #text(gray)[(1.14x)]],
    [Memory Layout \ & Allocations],[13.944s],[0.85x #text(gray)[(0.99x)]],[27.274s],[0.84x #text(gray)[(0.96x)]],
    [AVX512],[5.800s],[2.40x #text(gray)[(2.39x)]],[11.466s],[2.38x #text(gray)[(2.00x)]],
    [Parallelization],[26.252s],[0.22x #text(gray)[(0.53x)]],[3.179s],[3.61x #text(gray)[(8.26x)]],
    [Optimized Math],[31.938s],[0.82x #text(gray)[(0.43x)]],[1.292s],[2.46x #text(gray)[(20.32x)]],
    [AVX2],[25.638s],[1.25x #text(gray)[(0.54x)]],[1.127s],[1.15x #text(gray)[(23.30x)]],
    [Loop Unrolling],[24.011s],[1.07x #text(gray)[(0.58x)]],[0.961s],[1.17x #text(gray)[(27.33x)]],
    [Serial for Small],[2.561s],[9.38x #text(gray)[(5.41x)]],[0.962s],[ 1.00x #text(gray)[(27.33x)]],
  ),
  caption: [Cumulative Speedups by Optimization.],
  scope: "parent",
  placement: auto,
)<table::runtimes>

== Compiler Flags
We first apply the following optimization flags to allow the compiler to tune the code as much as possible:
#block(
  fill: luma(240),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
```bash
-O3 -mavx -mavx2 -mfma -mavx512vl -mavx512f -mtune=native -march=native
```
)

Several of these flags are made redundant by the native `-march` and `-mtune` flags, but we include them for specificity. This results in an approximately 15% speedup.

== Memory Allocations and Layout

We prepare for future optimizations by first removing extra allocations in the hot loop and working in place, noting that in-place is simply updating all the velocities, and then all the positions:

 #block(
   fill: luma(240),
   inset: 8pt,
   radius: 2pt,
   width: 100%,
   ```cpp
  void next(Planet* planets) {
    for (int i = 0; i < nplanets; i++) {
      for (int j = 0; j < nplanets; j++) {
        // Calculate velocity update
      }
    }
    for (int i = 0; i < nplanets; i++) {
      // Apply velocity to position
    }
  }
   ```
 )

 Then, changing data layout from "array-of-structs" to "struct-of-arrays" to improve cache locality in the future:

 #block(
   fill: luma(240),
   inset: 8pt,
   radius: 2pt,
   width: 100%,
   ```cpp
  struct World {
    double* __restrict__ mass;
    double* __restrict__ x;
    double* __restrict__ y;
    double* __restrict__ vx;
    double* __restrict__ vy;
  };
   ```
 )

These changes resulted in an temporary slowdown, likely due to uncertainty surrounding data dependencies.

== AVX512 Vectorization
With contiguous chunks of data, we can use SIMD. Specifically, we can vectorize the inner loop by calculating $F_(i j)$ for many $j$ at once. Vectorization requires aligned memory, so we employ `posix_memalign` to ensure that the data is aligned to 64-byte boundaries and preventing re-alignment.

We employ AVX512 as theoretically the "most parallel", packing 8 double-precision FP numbers into a single register. Here's a sample of the vectorized code:

#block(
  fill: luma(240),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
  ```cpp
  for (int i = 0; i < nplanets; i++) {
    __m512d acc_x, acc_y = ...; // zero
    for (int j = 0; j < nplanets; j+=8) { // step = 8
      // Load 8 points of data at a time
      __m512d x_js = _mm512_load_pd(world.x + j);
      // ... calculate and apply acceleration
    }
    // Horizontal reduction to get the final acceleration
    world.vx[i] += dt * _mm512_reduce_add_pd(acc_x);
  }
  ```
)

This brings a \~2.4x speedup, which is less than the theoretical \~8x, which we attribute to heavier overheads in vectorized instructions and loads/stores.

== Parallelization
We use OpenMP@openmp_2008 for parallelism. The outer loop is easily parallelizable, as each update is independent. The inner loop may not be as it would require synchronization for updates to a single body. We chose to use static scheduling as each body has an equal amount of work, and the overhead of dynamic scheduling would be unnecessary.

With a small number of bodies, this overhead ruins performance (80% slowdown). We revisit this choice in @sec::small. The large scene sees a \~3.6x speedup with 4 cores.

== Optimized Math
The hot loop's calculations consist of: 2 vector subtractions, 3 vector multiplication, 1 vector division, 4 fused-multiply-adds, and 1 vectorized square root. The two most expensive operations here are the division (\~8-15 cycles depending on architecture), and the square root (\~30 cycles).

We employ the `rsqrt14` reciprocal square root approximation (accurate to $10^(-14)$) to replace both, which considerably speeds up the loop (\~2.5x on the large scene), at the expense of some precision.

== AVX2
Online discussions suggest using AVX2 (256-bit) over AVX512, as AVX512 is optimized for specific FMA operations. We saw a modest speedup despite parallelism loss, attributed to:
- AVX512 instructions optimized for specific operations.
- AVX512 implemented as invocations of AVX2 units.
- AVX512 causing the CPU to downclock.

Whilst much of this is speculation, switching to AVX2 from AVX512 did have a performance increase of around 15%.

== Loop unrolling
The final optimization applied to the vectorized code was unrolling the outer loop to calculate $F_i$ and $F_(i+1)$ at the same time. This allows us to utilize more the 32 vector registers available, and allows for instruction-level parallelism (ILP) to be exploited, especially during more expensive instructions like `rsqrt`.This brings a \~15% speedup, yielding a final speedup of \~27x for the large scene.

== Small Scene Serial Processing <sec::small>

As mentioned earlier, the parallelization ruins performance on scenes with few bodies. Empirically, we find a threshold in @fig::paths, where the parallelized code becomes faster at around $n=152$ bodies. Interestingly, adding a conditional to the OpenMP pragma did not help, as the library overhead continued to dominate. Instead, we duplicated the function, adding the serial version and a simple `if` statement to choose between the two.

#figure(
  image("parallel.png", width: 90%),
  caption: [
    A comparison of the sequntial and parallel implementations, as number of bodies increases. The vectorized path is faster for $n>152$
  ],
)<fig::paths>


= Discussion <sec::discussion>

We implemented several versions and optimizations that resulted in net performance decreases, a short list below:
- 128-bit vectorization along $(x,y)$ pairs instead of individual axis-wise vectorization.
- 512-bit vectorization along $(x,y)$ quartets instead of individual axis-wise vectorization.
- Using a QuadTree approximation (discussed in @sec::future).
- Explicit cache control intrinsics (discussed in @sec::future).
- Precomputing a mass multiplication matrix.

One of the most interesting results from our exploration was the performance superiority of AVX2 over AVX512. As discussed earlier, this is likely due to the specific operations and hardware implementation of AVX512 on our system.

We also found the performance improvements obtained from loop unrolling to be significant. We attribute this to the CPU's ability to perform out-of-order execution and ILP. Not all 32 vector registers are used in a single iteration of the loop, and this unrolling allows the CPU to utilize more of the available registers.

We also see some odd behavior from the vectorized code between $n=80$ and $n=152$, where the vectorized code performance is quite poor and the performance improves significantly once more bodies are added. We suspect this is likely to do with cache thrashing and/or prediction algorithms in the CPU coming into play.

= Future Work <sec::future>
== Algorithmic Improvements
A comparison of a naive and our optimized implementation is below in @fig::extended. As can be seen, both the naive and vectorized implementations are ultimately $O(n^2)$, although with different constant factors. This can be improved to $O(n log n)$ with the Barnes-Hut @barnes1986hierarchical approximation (via a QuadTree), or the Fast Multipole Method @greengard1987fast. We implemented a simple QuadTree, but the construction overhead, approximation imprecision, and code complexity were not worth the (modest) performance gains it brought, and the vectorized implementaiton remained \~3x faster even for large scenes.
#figure(
  image("extended.png", width: 90%),
  caption: [
    An extended view of a naive implementation, and the optimized path, as number of bodies increases.
  ],
)<fig::extended>


As $n$ scales past the thousands, or tens of thousands mark, we expect to see the performance of the vectorized $O(n^2)$ implementation to degrade faster than the $O(n log n)$ implementations, but this was out of scope.

== Cache Control & Prefetching

We note that, due to hardware constraints, the `perf` tool was unable to gather data on cache misses, stalled cycles, and other hardware counters that would have been useful to diagnose remaining bottlenecks. We believe that there are still improvements that can be achieved by correctly pipelining main memory access / prefetching with computation, but are unable to support this hypothesis without the stall and cache data.

== Temporal Approximations
One possible optimization that was ideated, but not ultimately considered was to run the simulation at a high time resolution within spatially local regions, and only apply the results to the rest of the bodies at a lower time resolution. This would capture most of the high-fidelity interactions whilst continuing to properly apply forces at a less precise level to all bodies. However, we decided that this was ultimately out of scope.
