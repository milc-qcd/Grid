/*
 * GridMilc/benchmarks/Benchmark_a2a_spin_taste.cc
 *
 * Throughput benchmark for the staggered spin-taste A2A meson-field
 * contraction paths:
 *
 *   Local      -- A2AWorkerLocal   (popcount 0: scalar phase only)
 *   Onelink    -- A2AWorkerOnelink (popcount 1: single covariant hop)
 *   SpinTaste  -- A2AWorkerSpinTaste   (cshift-based general spin-taste)
 *   Stencil    -- A2AWorkerSpinTasteStencil (5D padded-gather spin-taste)
 *
 * Each path is run for one representative gamma (one per popcount 0-4) and
 * timed over NWARM warmup + NREP measured repetitions.  Throughput is
 * reported in GFlop/s using the task's own flop count.
 *
 * Usage:
 *   Benchmark_a2a_spin_taste [--grid W.X.Y.Z] [--mpi a.b.c.d]
 *                            [--nevec N]  (default: 32)
 *                            [--nrep  N]  (default: 5)
 *                            [--legacy]   run Local + Onelink paths
 *                            [--cshift]   run SpinTaste (cshift) path
 *                            [--stencil]  run Stencil (5D padded-gather) path
 *                            [--pc N]     run only the popcount-N gamma batch
 *                                         (0-4; default: all)
 *
 * If none of --legacy/--cshift/--stencil are given, all three run (original
 * behaviour).
 */
#include <Grid/Grid.h>
#include <Grid/qcd/utils/SpaceTimeGrid.h>
#include <GridMilc/GridMilc.h>

#include <iomanip>
#include <sstream>
#include <string>

using namespace Grid;

typedef StaggeredImplD FImpl;
typedef FImpl::FermionField FermionField;
typedef FImpl::ComplexField ComplexField;



// ---------------------------------------------------------------------------
// Timer helper: prints name, GFlop/s, elapsed ms
// ---------------------------------------------------------------------------
struct BenchResult {
  std::string name;
  double elapsed_s;   // wall time for NREP calls
  double gflops;      // GFlop/s
  int    nrep;
};

static void printHeader() {
  std::cout << GridLogMessage
            << std::left << std::setw(30) << "Path"
            << std::right
            << std::setw(12) << "GFlop/s"
            << std::setw(12) << "ms/call"
            << std::setw(10) << "nrep"
            << std::endl;
  std::cout << GridLogMessage
            << std::string(64, '-') << std::endl;
}

static void printResult(const BenchResult &r) {
  double ms_per_call = r.elapsed_s / r.nrep * 1e3;
  std::cout << GridLogMessage
            << std::left << std::setw(30) << r.name
            << std::right
            << std::fixed << std::setprecision(2)
            << std::setw(12) << r.gflops
            << std::setw(12) << ms_per_call
            << std::setw(10) << r.nrep
            << std::endl;
}

// ---------------------------------------------------------------------------
// runBench: warm up then time NREP calls of func(), return GFlop/s
// ---------------------------------------------------------------------------
template <typename Func>
static BenchResult runBench(const std::string &name, double flops_per_call,
                            int nwarm, int nrep, Func &&func) {
  for (int i = 0; i < nwarm; i++) func();

  double t0 = Grid::usecond();
  for (int i = 0; i < nrep; i++) func();
  double elapsed = (usecond() - t0) * 1e-6; // seconds

  double gflops = flops_per_call * nrep / elapsed * 1e-9;
  return {name, elapsed, gflops, nrep};
}

// ---------------------------------------------------------------------------
// Volume-weighted flop estimate for meson-field inner products.
// Both Local and Stencil paths expose task->getFlops() (flops per site per
// (l,r) pair per time-slice); total = flops * volume * nL * nR.
// ---------------------------------------------------------------------------
static double totalFlops(double flops_per_site_lr, double volume,
                         int nL, int nR) {
  return flops_per_site_lr * volume * nL * nR;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // ---- options ----
  std::string nevecStr = GridCmdOptionPayload(argv, argv + argc, "--nevec");
  std::string nrepStr  = GridCmdOptionPayload(argv, argv + argc, "--nrep");
  int nevec = nevecStr.empty() ? 32 : std::stoi(nevecStr);
  int nrep  = nrepStr.empty()  ?  5 : std::stoi(nrepStr);
  int nwarm = 2;

  bool flagLegacy  = GridCmdOptionExists(argv, argv + argc, "--legacy");
  bool flagCshift  = GridCmdOptionExists(argv, argv + argc, "--cshift");
  bool flagStencil = GridCmdOptionExists(argv, argv + argc, "--stencil");
  // if none specified, run everything
  if (!flagLegacy && !flagCshift && !flagStencil)
    flagLegacy = flagCshift = flagStencil = true;

  std::string pcStr = GridCmdOptionPayload(argv, argv + argc, "--pc");
  int pcFilter = pcStr.empty() ? -1 : std::stoi(pcStr); // -1 = all

  // ---- grid ----
  Coordinate fdim  = GridDefaultLatt();
  GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(
      fdim, GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());

  double volume = 1.0;
  for (int mu = 0; mu < Nd; mu++) volume *= fdim[mu];
  int Nt = fdim[Tdir];
  int orthogDir = Tdir;

  std::cout << GridLogMessage
            << "Benchmark_a2a_spin_taste: lattice=" << fdim
            << "  nevec=" << nevec << "  nrep=" << nrep << std::endl;

  // ---- gauge field (random, non-SU(3) — only throughput matters) ----
  LatticeGaugeField U(grid);
  {
    GridParallelRNG rng(grid);
    rng.SeedFixedIntegers({10, 20, 30, 40});
    random(rng, U);
  }

  // ---- fermion vectors ----
  std::vector<FermionField> vecs(nevec, grid);
  {
    GridParallelRNG rng(grid);
    rng.SeedFixedIntegers({1, 2, 3, 4});
    for (auto &v : vecs) { v = Zero(); random(rng, v); }
  }

  // ---- output tensor ----
  std::vector<ComplexField>  emptyMom;

  typedef StagGamma::StagAlgebra A;

  // ---- gamma batches: group 4 same-popcount gammas per worker call ----
  // All gammas in one worker must share a popcount (A2AWorkerSpinTaste asserts
  // uniform popcount). Where the popcount has >=4 channels we pass 4; pc=0 and
  // pc=4 have only 1 channel each. This tests the multi-gamma path (the
  // production scenario: many spin-taste channels contracted in one pass).
  struct GammaBatch {
    std::string label;
    int popcount;
    std::vector<StagGamma::SpinTastePair> gammas;
  };
  std::vector<GammaBatch> allBatches = {
    {"pc=0", 0, {{A::G1,  A::G1},  {A::GX,  A::GX},  {A::GY,  A::GY},  {A::GZ,  A::GZ}}},
    {"pc=1", 1, {{A::GX,  A::G1},  {A::GY,  A::G1},  {A::GZ,  A::G1},  {A::GT,  A::G1}}},
    {"pc=2", 2, {{A::GXY, A::G1},  {A::GZX, A::G1},  {A::GYZ, A::G1},  {A::GXT, A::G1}}},
    {"pc=3", 3, {{A::G5T, A::G1},  {A::G5X, A::G1},  {A::G5Y, A::G1},  {A::G5Z, A::G1}}},
    {"pc=4", 4, {{A::G5,  A::G1},  {A::GX,  A::G5X}, {A::GY,  A::G5Y}, {A::GZ,  A::G5Z}}},
  };
  std::vector<GammaBatch> batches;
  for (auto &b : allBatches)
    if (pcFilter == -1 || pcFilter == b.popcount)
      batches.push_back(b);

  // Helper: format a benchmark label as "Path/pc=N[Mγ]"
  auto blabel = [](const std::string &path, const GammaBatch &b) {
    std::ostringstream s;
    s << path << "/" << b.label << "[" << b.gammas.size() << "γ]";
    return s.str();
  };

  for (auto &b : batches)
    std::cout << GridLogMessage << "batch label: " << b.label << std::endl;

  printHeader();

  // ====================================================================
  // 1. Local + Onelink (--legacy)
  // ====================================================================
  if (flagLegacy) {
    // Local: popcount 0 only
    if (pcFilter == -1 || pcFilter == 0) {
      auto &b = allBatches[0];
      Eigen::Tensor<ComplexD, 5> mat(1, b.gammas.size(), Nt, nevec, nevec);
      A2AWorkerLocal<FImpl> worker(grid, emptyMom, b.gammas, orthogDir);

      double fp = worker._task_e->getFlops();
      double flops = totalFlops(fp, volume, nevec, nevec);

      auto res = runBench(blabel("Local", b), flops, nwarm, nrep, [&] {
        mat.setZero();
        worker.StagMesonField(mat, vecs.data(), vecs.data(),
                              vecs.data(), vecs.data());
      });
      printResult(res);
    }

    // Onelink: popcount 1 only
    if (pcFilter == -1 || pcFilter == 1) {
      auto &b = allBatches[1];
      Eigen::Tensor<ComplexD, 5> mat(1, b.gammas.size(), Nt, nevec, nevec);
      A2AWorkerOnelink<FImpl> worker(grid, emptyMom, b.gammas, &U, orthogDir);

      double fp = worker._task_e->getFlops();
      double flops = totalFlops(fp, volume, nevec, nevec);

      auto res = runBench(blabel("Onelink", b), flops, nwarm, nrep, [&] {
        mat.setZero();
        worker.StagMesonField(mat, vecs.data(), vecs.data(),
                              vecs.data(), vecs.data());
      });
      printResult(res);
    }
  }

  // ====================================================================
  // 3. SpinTaste (cshift, all popcounts)
  // setRight calls applyGamma (Cshift + W_s) on every right vector — that
  // cost scales with 2^popcount and is incurred once per j-block in
  // production.  resetCache() forces it on every timed rep so the benchmark
  // reflects the real per-block cost.
  //
  // FLOP accounting includes both setRight and execute:
  //   setRight: applyGamma runs n! permutations of n oneLink hops per
  //             (gamma, right_vector). Each oneLink = CovShiftForward (66) +
  //             CovShiftBackward (66) + FermVec addition (6) = 138 FLOP/site.
  //             Total per site per (gamma, r) = n!*n*138 + (n!-1)*6 + 18.
  //   execute:  plain innerProduct = 22 FLOP/site per (gamma, l, r).
  // ====================================================================
  // applyGamma FLOP per site per (gamma, right_vector), n = popcount
  auto applyGammaFlopsPerSite = [](int n) -> double {
    int nfact = 1;
    for (int i = 1; i <= n; i++) nfact *= i;
    if (n == 0) return 18.0;  // applyCoeffsAndPhase only
    return nfact * n * 138.0 + (nfact - 1) * 6.0 + 18.0;
  };

  if (flagCshift) {
    for (auto &b : batches) {
      Eigen::Tensor<ComplexD, 5> mat(1, b.gammas.size(), Nt, nevec, nevec);
      A2AWorkerSpinTaste<FImpl> worker(grid, emptyMom, b.gammas, &U, orthogDir);

      double nGamma  = b.gammas.size();
      double flops_setright = nGamma * nevec * volume * applyGammaFlopsPerSite(b.popcount);
      double flops_execute  = totalFlops(worker._task_e->getFlops(), volume, nevec, nevec);
      double flops = flops_setright + flops_execute;

      auto res = runBench(blabel("SpinTaste", b), flops, 0, nrep, [&] {
        worker.resetCache();
        mat.setZero();
        worker.StagMesonField(mat, vecs.data(), vecs.data(),
                              vecs.data(), vecs.data());
      });
      printResult(res);
    }
  }

  // ====================================================================
  // 4. Stencil (5D padded gather, all popcounts)
  // setRight packs _paddedRight5d (local memory reformat + halo exchange).
  // resetCache() ensures that cost is included in every timed rep.
  // ====================================================================
  if (flagStencil) {
    for (auto &b : batches) {
      Eigen::Tensor<ComplexD, 5> mat(1, b.gammas.size(), Nt, nevec, nevec);
      A2AWorkerSpinTasteStencil<FImpl> worker(grid, emptyMom, b.gammas, &U,
                                              orthogDir);

      double fp = worker._stencil_task->getFlops();
      double flops = totalFlops(fp, volume, nevec, nevec);

      auto res = runBench(blabel("Stencil", b), flops, 0, nrep, [&] {
        worker.resetCache();
        mat.setZero();
        worker.StagMesonFieldStencil(mat, vecs.data(), vecs.data(),
                                     nevec, nevec);
      });
      printResult(res);
    }
  }

  std::cout << GridLogMessage << "Benchmark_a2a_spin_taste: done" << std::endl;
  Grid_finalize();
  return 0;
}
