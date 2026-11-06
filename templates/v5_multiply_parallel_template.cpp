// ----------------------------------------------------------------------------
// PRACTICE TEMPLATE: P1 version 5, multiply_parallel (OpenMP)
//
// This file is NOT compiled (no CMakeLists.txt lists it). The finished
// version is in src/matmul.cpp. To practise:
//
//   1. Open src/matmul.cpp and DELETE the body of multiply_parallel
//      (keep the signature line).
//   2. Paste in the body below and fill in every TODO, without peeking.
//   3. Build and run the tests:
//        cmake --build build-native && ctest --test-dir build-native -R Parallel
//   4. Stuck, or done and want to compare? `git diff src/matmul.cpp` shows
//      the difference from the solution; `git checkout src/matmul.cpp` puts
//      the solution back.
//
// Experiments worth trying once it passes (re-run the tests each time):
//   a. Move the #pragma onto the pp loop. Which test catches the data race?
//   b. Remove collapse(2). Compare BM_Parallel/n:1024/threads:20 before and
//      after. How many pieces of work are there now with tile 128?
//   c. Change schedule(dynamic) to schedule(static). Faster or slower with
//      20 threads (P-cores + E-cores)? With `taskset -c 0,2,4,6,8,10` and
//      6 threads (P-cores only)?
// ----------------------------------------------------------------------------

Matrix multiply_parallel(const Matrix& A, const Matrix& B, std::size_t tile, int threads) {
  // ---- 1. Shapes -------------------------------------------------------------
  const std::size_t n = A.rows();
  const std::size_t k = A.cols();
  const std::size_t m = B.cols();

  // ---- 2. Validate the input ---------------------------------------------------
  // Do ALL the checks here, before any threads start: an exception thrown
  // inside a parallel loop can't get out of it (the program terminates).
  check_multiply_shapes(A, B);

  // TODO: throw std::invalid_argument if tile == 0 (the loops would never end).

  // TODO: throw std::invalid_argument if threads < 0.

  // TODO: work out the team size: if threads == 0, use omp_get_max_threads()
  //       (OpenMP's default), otherwise use threads.
  //       Hint: const int team_size = (condition) ? a : b;

  // ---- 3. Create the output (zero-filled) ------------------------------------
  Matrix C(n, m);

  // ---- 4. Share the blocks of C between threads --------------------------------
  // TODO: write the OpenMP line right above the ii loop. It needs:
  //   - parallel for        : split the loop below across threads
  //   - collapse(2)         : split the ii AND jj loops together
  //   - schedule(dynamic)   : hand out blocks one at a time
  //   - num_threads(...)    : use your team size
  //   Shape:  #pragma omp parallel for <clause> <clause> <clause>

  for (std::size_t ii = 0; ii < n; ii += tile) {
    for (std::size_t jj = 0; jj < m; jj += tile) {
      // collapse(2) rule: no code may sit BETWEEN the ii and jj lines, which is
      // why both ends are computed here, inside the jj loop.

      // TODO: i_end = the smaller of ii + tile and n   (std::min)
      // TODO: j_end = the smaller of jj + tile and m

      // TODO: the rest is your version 3 inner part, unchanged:
      //   for pp in steps of tile, with p_end = the smaller of pp + tile and k
      //     for i in [ii, i_end)
      //       for p in [pp, p_end)
      //         a = A(i, p)
      //         for j in [jj, j_end)
      //           C(i, j) += a * B(p, j)
      //
      // Ask yourself: which variables here are SHARED by all threads, and
      // which does each thread get its OWN copy of? (Answer in the comments
      // above multiply_parallel in src/matmul.cpp.)
    }
  }

  // ---- 5. Return the result ----------------------------------------------------
  return C;
}

// ----------------------------------------------------------------------------
// BONUS (optional, after the OpenMP version): the same thing with std::thread,
// no OpenMP. You do by hand what the #pragma did for you:
//
//   #include <thread>
//   #include <vector>
//
//   1. Make a std::vector<std::thread> workers.
//   2. For t = 0 .. team_size-1, start a thread:
//        workers.emplace_back([&, t] { ...work... });
//      Inside, thread t handles block rows ii = t*tile, (t+team_size)*tile, ...
//      (every team_size-th block row: a "static, round-robin" schedule).
//   3. After the loop: for (auto& w : workers) w.join();  // wait for all
//      (join() is the barrier that OpenMP gave you for free.)
//
// Then compare its speed to the OpenMP version. Why might it be slower with
// 20 threads? (Hint: static vs dynamic, P-cores vs E-cores.)
// ----------------------------------------------------------------------------
