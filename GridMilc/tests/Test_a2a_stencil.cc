/*
 * GridMilc/tests/Test_a2a_stencil.cc
 *
 * Portable validation of the 5D stencil spin-taste A2A path.
 * Works at any lattice size (--grid / --mpi). No hardcoded 4^4 data files.
 *
 * Tests:
 *   (1) testStencilGather5d: 5D padded gather vs brute-force Cshift for all
 *       16 single/double-axis +/-1 endpoints. Self-contained (random field).
 *   (2) testStencilVsCshiftSpinTaste: A2AWorkerSpinTasteStencil (5D stencil
 *       path) vs A2AWorkerSpinTaste (cshift oracle), all popcount 0-4.
 *       Differential test on identical input, so random gauge + random
 *       fermion vectors are sufficient (both paths do the same covariant
 *       shift math).
 *       Alternates the default and explicit ContractType::Full call shapes
 *       (both must take the identical Full path).
 *
 * Gauge field: read from --gauge (ILDG) if provided, else random.
 * Fermion vectors: random.
 *
 * Usage:
 *   Test_a2a_stencil [--grid W.X.Y.Z] [--mpi a.b.c.d] [--gauge <file>]
 *                    [--nevec <N>]
 *
 * This test is self-contained (no external eigenvector or gauge-config files
 * required). It validates that the optimized stencil path reproduces the
 * cshift oracle to machine precision at production-representative lattice
 * sizes and MPI decompositions.
 */
#include <Grid/Grid.h>
#include <Grid/qcd/utils/SpaceTimeGrid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <GridMilc/GridMilc.h>
#include <GridMilc/a2a/StencilGather5d.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

using namespace Grid;

// ---- FImpl: use the Grid staggered double-precision impl directly ----
typedef StaggeredImplD FImpl;
typedef FImpl::FermionField FermionField;
typedef FImpl::ComplexField ComplexField;

// (L2-01) Test-only 5D padded gather via per-site offset. Moved here from
// StencilGather5d.h so the production header ships only production code.
// (Symbols autoView/Lattice/deviceVector/coalescedRead/coalescedWrite/
//  accelerator_for come transitively from <Grid/Grid.h>.)
template <typename vobj>
inline void gatherViaOffset5d(Lattice<vobj> &result,
                              const Lattice<vobj> &padded5d,
                              const deviceVector<int> &offset, int ep,
                              int osites) {
  autoView(result_v, result, AcceleratorWrite);
  autoView(padded_v, padded5d, AcceleratorRead);
  auto offset_p = offset.data();
  int Nsimd = result.Grid()->Nsimd();

  accelerator_for(ss, osites, Nsimd, {
    int paddedSS = offset_p[ep * osites + ss];
    coalescedWrite(result_v[ss], coalescedRead(padded_v[paddedSS]));
  });
}

// ---- Spin-taste pairs spanning popcount 0-4, with ALL six popcount-2 sets ----
static std::vector<StagGamma::SpinTastePair> testGammas() {
  typedef StagGamma::StagAlgebra A;
  return {
      {A(A::G1), A(A::G1)},   // popcount 0
      {A(A::GX), A(A::G1)},   // popcount 1
      // popcount 2: ALL six 2-direction sets (taste=G1) -> spans corr=+1 AND
      // corr=-1 endpoint pairs (oscillateDirs ∩ active != ∅). A scalar-W
      // pairing/phase sign error shows up ONLY on corr=-1 pairs (phase(x) !=
      // phase(x-s) there); these catch it.
      {A(A::GXY), A(A::G1)},
      {A(A::GZX), A(A::G1)},
      {A(A::GYZ), A(A::G1)},
      {A(A::GXT), A(A::G1)},
      {A(A::GYT), A(A::G1)},
      {A(A::GZT), A(A::G1)},
      {A(A::G5T), A(A::G1)},  // popcount 3
      {A(A::G5), A(A::G1)},   // popcount 4
  };
}

// ---- Explicit-eps helpers (verbatim from Test_staggamma.cc:168-186) ----
// parity(x) = (x+y+z+t) % 2 and f *= eps(x): the EXPLICIT-eps side of the
// applyG5 cross-check below.
static void makeParityField(Lattice<iScalar<vInteger>> &parity) {
  GridBase *grid = parity.Grid();
  Lattice<iScalar<vInteger>> coor(grid);
  parity = Zero();
  for (int mu = 0; mu < Nd; mu++) {
    LatticeCoordinate(coor, mu);
    parity = parity + coor;
  }
}

template <class obj>
static void applyEpsilon(Lattice<obj> &f,
                         const Lattice<iScalar<vInteger>> &parity) {
  Lattice<obj> neg(f.Grid());
  neg = -f;
  f = where(mod(parity, 2) == (Integer)1, neg, f);
}

// Reconstruction of the legacy combined CB slots from the raw parity
// partials M0 (even source sites) / M1 (odd) emitted by the stencil worker.
// This table is the exact algebraic image of the sign branches the stencil
// kernel applied before the raw-parity redesign and of the frozen legacy
// oracle's simdSumHalf/simdSumMixed tables (A2ATask.h): slot = sA*M0 +
// sB*M1 with sigma = -1 for odd popcount. sigma enters RightHalf's rc=1
// slot and BothHalf's rc=1 column slots -- (0,1) and (1,1) -- while the
// rc=0 slots and all LeftHalf slots are sigma-free (the legacy
// LeftHalf-vs-RightHalf asymmetry); a reconstruction that wrongly applies
// sigma to LeftHalf passes all even-popcount gammas and fails only the
// popcount-1/3 ones (testGammas spans both).
//   BothHalf  slot (row 2l+lc, col 2r+rc):
//     (0,0): M0+M1          (0,1): sigma*(M0-M1)
//     (1,0): M0-M1          (1,1): sigma*(M0+M1)
//   LeftHalf  slot (row 2l+lc, col r):   lc=0: M0+M1   lc=1: M0-M1
//   RightHalf slot (row l, col 2r+rc):   rc=0: M0+M1   rc=1: sigma*(M0-M1)
// lc/rc select the e/o combination on each CB-flagged side (the legacy
// interleaved layout); sig = -1 for odd popcount, else +1.
static ComplexD reconstructLegacySlot(ContractType ct, int lc, int rc,
                                      ComplexD M0, ComplexD M1,
                                      ComplexD sig) {
  if (ct == ContractType::BothHalf) {
    if (lc == 0 && rc == 0)
      return M0 + M1;
    if (lc == 0 && rc == 1)
      return sig * (M0 - M1);
    if (lc == 1 && rc == 0)
      return M0 - M1;
    return sig * (M0 + M1); // (1,1)
  } else if (ct == ContractType::LeftHalf) {
    return (lc == 0) ? (M0 + M1) : (M0 - M1);
  } else { // RightHalf
    return (rc == 0) ? (M0 + M1) : (sig * (M0 - M1));
  }
}

// =============================================================================
// testStencilGather5d: validates the 5D padded gather against Cshift.
// Creates a 5D grid (simd_layout={1,1,1,1,Nsimd}), builds padded offsets for
// all 16 single/double-axis +/-1 endpoints, gathers via offset, and compares
// with brute-force Cshift. Must match to machine precision.
// =============================================================================
void testStencilGather5d(void) {
  typedef LatticeFermion FermionField;
  typedef FermionField::vector_object vobj;

  std::cout << GridLogMessage
            << "=== testStencilGather5d ===" << std::endl;

  Coordinate dims  = GridDefaultLatt();
  Coordinate procs = GridDefaultMpi();
  const int Nsimd = (int)vobj::Nsimd();

  GridCartesian grid4d(dims, Coordinate({1,1,1,Nsimd}), procs);

  // createGrid5d returns an owning unique_ptr; hold it for scope-exit
  // release and derive a non-owning raw view so the test body's raw-pointer
  // call sites are unchanged.
  auto grid5dOwned = createGrid5d(&grid4d);
  GridCartesian *grid5d = grid5dOwned.get();
  std::cout << GridLogMessage << "  5D grid: simd_layout={1,1,1,1,"
            << Nsimd << "}, Nsimd=" << Nsimd
            << ", procs={" << procs[0] << "," << procs[1] << ","
            << procs[2] << "," << procs[3] << "}" << std::endl;

  // All 16 single/double-axis +/-1 endpoints
  std::vector<Coordinate> endpoints;
  for (int d0 = -1; d0 <= 1; d0 += 2) { Coordinate s(4,0); s[0]=d0; endpoints.push_back(s); }
  for (int d1 = -1; d1 <= 1; d1 += 2) { Coordinate s(4,0); s[1]=d1; endpoints.push_back(s); }
  for (int d2 = -1; d2 <= 1; d2 += 2) { Coordinate s(4,0); s[2]=d2; endpoints.push_back(s); }
  for (int d3 = -1; d3 <= 1; d3 += 2) { Coordinate s(4,0); s[3]=d3; endpoints.push_back(s); }
  for (int d0 = -1; d0 <= 1; d0 += 2) for (int d1 = -1; d1 <= 1; d1 += 2) {
    Coordinate s(4,0); s[0]=d0; s[1]=d1; endpoints.push_back(s); }
  for (int d2 = -1; d2 <= 1; d2 += 2) for (int d3 = -1; d3 <= 1; d3 += 2) {
    Coordinate s(4,0); s[2]=d2; s[3]=d3; endpoints.push_back(s); }

  // Pad the 5D grid
  PaddedCell cell5d(1, grid5d);
  GridCartesian *paddedGrid5d = cell5d.grids.back();

  // Build offset table
  std::vector<int> hostOffset = buildPaddedOffset5d(grid5d, paddedGrid5d, 1, endpoints);
  deviceVector<int> offsetDev(hostOffset.size());
  acceleratorCopyToDevice((void*)hostOffset.data(), (void*)offsetDev.data(),
                          hostOffset.size() * sizeof(int));

  // Random 4D field, promote to 5D, pad
  FermionField field4d(&grid4d);
  field4d = Zero();
  GridParallelRNG rng(&grid4d);
  std::vector<int> seed = {1, 2, 3, 4, 5};
  rng.SeedFixedIntegers(seed);
  random(rng, field4d);

  Lattice<vobj> field5d(grid5d);
  promoteField5d(field5d, field4d, &grid4d, grid5d);
  Lattice<vobj> padded5d = cell5d.Exchange(field5d);

  int osites = grid5d->oSites();
  int nFail = 0;

  for (int ep = 0; ep < (int)endpoints.size(); ep++) {
    Lattice<vobj> gathered(grid5d);
    gatherViaOffset5d(gathered, padded5d, offsetDev, ep, osites);

    // Brute-force: sequential Cshift on 4D, then promote to 5D
    FermionField shifted(&grid4d);
    shifted = field4d;
    for (int d = 0; d < 4; d++) {
      if (endpoints[ep][d] != 0)
        shifted = Cshift(shifted, d, endpoints[ep][d]);
    }
    Lattice<vobj> shifted5d(grid5d);
    promoteField5d(shifted5d, shifted, &grid4d, grid5d);

    Lattice<vobj> diff(grid5d);
    diff = gathered - shifted5d;
    auto diffNorm = norm2(diff);
    auto gatheredNorm = norm2(gathered);
    RealD relErr = (gatheredNorm > 0) ? sqrt(diffNorm / gatheredNorm) : sqrt(diffNorm);

    if (relErr > 1e-13) {
      std::cerr << "    FAIL endpoint " << ep << " shift={"
                << endpoints[ep][0] << "," << endpoints[ep][1] << ","
                << endpoints[ep][2] << "," << endpoints[ep][3]
                << "} relErr=" << relErr << std::endl;
      nFail++;
    } else {
      std::cout << GridLogMessage << "    OK endpoint " << ep
                << " shift={" << endpoints[ep][0] << "," << endpoints[ep][1]
                << "," << endpoints[ep][2] << "," << endpoints[ep][3]
                << "} relErr=" << relErr << std::endl;
    }
  }

  // grid5dOwned (unique_ptr) releases the 5D grid at scope exit.
  if (nFail > 0) {
    std::cerr << "testStencilGather5d: " << nFail << " failures!" << std::endl;
    GridAbort();
  }
  std::cout << GridLogMessage << "testStencilGather5d: ALL PASSED" << std::endl;
}

// =============================================================================
// testZeroCopy5dInputs: validates the shared_ptr 5D input path -- zero-copy
// LHS borrow (setLeft 5D: no Zero/pack5d, direct read-only views on caller
// lattices) and the RHS pack-skip (setRight 5D: Exchange without pack) --
// against the 4D pack path on identical input bytes. A simulated 5D producer
// (pack5d into caller-owned shared_ptr lattices on a createGrid5d grid)
// feeds a worker constructed with that grid ADOPTED; results must match the
// 4D path element-for-element (the kernel and its inputs are bit-identical,
// so any difference is an input-layout bug). Also covers: partial-batch
// sizeL (caller-zeroed lanes -- the documented contract), a CB contract
// mode (LeftHalf), both-sides 5D, worker address-cache cross-invalidation
// on 4D->5D->4D alternation, and grid5dCompatible (positive + negative).
// =============================================================================
static int testZeroCopy5dInputs(GridCartesian *grid, LatticeGaugeField *Up,
                                int nevec) {
  int nFail = 0;
  int Nsimd = grid->Nsimd();
  int Nt = grid->_fdimensions[Tdir];
  int orthogDir = Tdir;

  std::cout << GridLogMessage << "=== testZeroCopy5dInputs ===" << std::endl;

  // ---- grid5dCompatible: positive (independent grids) + negative ----
  {
    std::shared_ptr<GridCartesian> g5a = createGrid5d(grid);
    std::shared_ptr<GridCartesian> g5b = createGrid5d(grid);
    if (!grid5dCompatible(grid, g5a.get()) ||
        !grid5dCompatible(grid, g5b.get())) {
      std::cerr << "    FAIL grid5dCompatible: independent createGrid5d "
                   "grids must be compatible" << std::endl;
      nFail++;
    }
    if (grid5dCompatible(grid, grid)) { // 4D grid: Nd mismatch
      std::cerr << "    FAIL grid5dCompatible: 4D grid must be incompatible"
                << std::endl;
      nFail++;
    }
    Coordinate fdim = grid->_fdimensions;
    Coordinate procs = grid->_processors;
    // Legal 5D grid with the WRONG dim-5 extent (2*Nsimd): constructible,
    // must be rejected by the compatibility check.
    Coordinate gdimBad(std::vector<int>(
        {fdim[0], fdim[1], fdim[2], fdim[3], 2 * Nsimd}));
    Coordinate simdBad(std::vector<int>({1, 1, 1, 1, Nsimd}));
    Coordinate procsBad(std::vector<int>(
        {procs[0], procs[1], procs[2], procs[3], 1}));
    std::shared_ptr<GridCartesian> g5bad =
        std::make_shared<GridCartesian>(gdimBad, simdBad, procsBad);
    if (grid5dCompatible(grid, g5bad.get())) {
      std::cerr << "    FAIL grid5dCompatible: wrong dim-5 extent must be "
                   "incompatible" << std::endl;
      nFail++;
    }
    // Wrong simd_layout with CORRECT fdim: isolates the _simd_layout term
    // of the compatibility check (dim-5 extent stays Nsimd; the lane split
    // deviates to Nsimd/2). Requires Nsimd >= 2; on Nsimd == 1
    // (scalar/GPU builds) the only legal dim-5 simd split is 1, so the
    // deviation is not constructible and the case is vacuous.
    if (Nsimd >= 2) {
      Coordinate gdimSim(std::vector<int>(
          {fdim[0], fdim[1], fdim[2], fdim[3], Nsimd}));
      Coordinate simdSimBad(std::vector<int>({1, 1, 1, 1, Nsimd / 2}));
      std::shared_ptr<GridCartesian> g5simBad =
          std::make_shared<GridCartesian>(gdimSim, simdSimBad, procsBad);
      if (grid5dCompatible(grid, g5simBad.get())) {
        std::cerr << "    FAIL grid5dCompatible: wrong simd_layout must be "
                     "incompatible" << std::endl;
        nFail++;
      }
    }
  }

  // ---- equality loop: two consecutive sizes so at least one is a partial
  //      batch (sizeL % Nsimd != 0) unless Nsimd == 1 (scalar/GPU builds,
  //      where every batch is full and the partial contract is vacuous) ----
  std::vector<StagGamma> oneGamma;
  {
    StagGamma st;
    st.setGaugeField(*Up);
    st.setSpinTaste(testGammas()[1]); // (GX, G1): popcount 1
    oneGamma.push_back(st);
  }
  std::vector<ComplexField> emptyMom;

  auto compare = [&](const Eigen::Tensor<ComplexD, 5> &a,
                     const Eigen::Tensor<ComplexD, 5> &b, const char *what) {
    double err = 0.0;
    for (int t = 0; t < Nt; t++)
      for (int i = 0; i < (int)a.dimension(3); i++)
        for (int j = 0; j < (int)a.dimension(4); j++) {
          ComplexD x = a(0, 0, t, i, j);
          ComplexD y = b(0, 0, t, i, j);
          auto mag = [](ComplexD z) {
            return std::sqrt(z.real() * z.real() + z.imag() * z.imag());
          };
          double denom = std::max(mag(x), mag(y));
          double e = (denom > 0.0) ? mag(x - y) / denom : mag(x - y);
          err = std::max(err, e);
        }
    if (err > 1e-14) {
      std::cerr << "    FAIL " << what << " maxRelErr=" << err << std::endl;
      return 1;
    }
    std::cout << GridLogMessage << "    OK " << what
              << " (maxRelErr=" << err << ")" << std::endl;
    return 0;
  };

  for (int sizeL : {nevec, nevec + 1}) {
    int nB = (sizeL + Nsimd - 1) / Nsimd;

    // Shared random inputs (4D).
    std::vector<FermionField> vecsL(sizeL, grid), vecsR(sizeL, grid);
    {
      GridParallelRNG rng(grid);
      std::vector<int> seed = {7, 8, 9, 10};
      rng.SeedFixedIntegers(seed);
      for (auto &v : vecsL) { v = Zero(); random(rng, v); }
      for (auto &v : vecsR) { v = Zero(); random(rng, v); }
    }

    // Simulated 5D producer: pack ONCE into caller-owned shared_ptr
    // lattices on its own createGrid5d grid. Zero() before pack5d honours
    // the caller-side partial-batch zeroing contract.
    std::shared_ptr<GridCartesian> g5 = createGrid5d(grid);
    std::vector<std::shared_ptr<FermionField>> lhs5(nB), rhs5(nB);
    for (int b = 0; b < nB; b++) {
      int nVec = std::min(Nsimd, sizeL - b * Nsimd);
      lhs5[b] = std::make_shared<FermionField>(g5.get());
      rhs5[b] = std::make_shared<FermionField>(g5.get());
      (*lhs5[b]) = Zero();
      (*rhs5[b]) = Zero();
      pack5d(*lhs5[b], vecsL.data() + b * Nsimd, nVec, grid, g5.get());
      pack5d(*rhs5[b], vecsR.data() + b * Nsimd, nVec, grid, g5.get());
    }

    // Reference: 4D pack path (worker without adopted grid).
    Eigen::Tensor<ComplexD, 5> mf4(1, 1, Nt, sizeL, sizeL);
    mf4.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w4(grid, emptyMom, oneGamma, Up,
                                          orthogDir);
      w4.StagMesonField(mf4, vecsL.data(), vecsR.data(), sizeL, sizeL);
    }

    // Zero-copy LHS (5D lhs + 4D rhs): worker adopts the producer's grid.
    Eigen::Tensor<ComplexD, 5> mf5L(1, 1, Nt, sizeL, sizeL);
    mf5L.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w5(grid, emptyMom, oneGamma, Up,
                                          orthogDir, g5);
      w5.StagMesonField(mf5L, lhs5, vecsR.data(), sizeL, sizeL);
    }
    nFail += compare(mf5L, mf4, "5D lhs zero-copy == 4D pack");

    // Both sides 5D (RHS pack-skip, Exchange kept).
    Eigen::Tensor<ComplexD, 5> mf5LR(1, 1, Nt, sizeL, sizeL);
    mf5LR.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w5b(grid, emptyMom, oneGamma, Up,
                                           orthogDir, g5);
      w5b.StagMesonField(mf5LR, lhs5, rhs5, sizeL, sizeL);
    }
    nFail += compare(mf5LR, mf4, "5D lhs+rhs == 4D pack");

    // CB mode (LeftHalf) through the 5D path vs the 4D path: raw (2L, L)
    // parity-split output, plain full-grid arrays on both sides (the task
    // never probes packing; the parity source-site filter is well-defined
    // on any full-grid data).
    Eigen::Tensor<ComplexD, 5> mf4cb(1, 1, Nt, 2 * sizeL, sizeL);
    mf4cb.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w4c(grid, emptyMom, oneGamma, Up,
                                           orthogDir);
      w4c.StagMesonField(mf4cb, vecsL.data(), vecsR.data(), sizeL, sizeL,
                         ContractType::LeftHalf);
    }
    Eigen::Tensor<ComplexD, 5> mf5cb(1, 1, Nt, 2 * sizeL, sizeL);
    mf5cb.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w5c(grid, emptyMom, oneGamma, Up,
                                           orthogDir, g5);
      w5c.StagMesonField(mf5cb, lhs5, vecsR.data(), sizeL, sizeL,
                         ContractType::LeftHalf);
    }
    nFail += compare(mf5cb, mf4cb, "5D lhs LeftHalf == 4D LeftHalf");

    // Address-cache cross-invalidation (once): ONE adopted-grid worker
    // serves 4D -> 5D -> 4D(same address) with different L data in the 5D
    // call. The 5D set must invalidate the cached 4D address: if it did
    // not, the third call (same vecsL.data() address as the first) would
    // be skipped by the 4D gate and silently return the STALE 5D-borrowed
    // result (vecsL2's data) instead of re-packing vecsL.
    if (sizeL == nevec) {
      std::vector<FermionField> vecsL2(sizeL, grid);
      {
        GridParallelRNG rng(grid);
        std::vector<int> seed = {11, 12, 13, 14};
        rng.SeedFixedIntegers(seed);
        for (auto &v : vecsL2) { v = Zero(); random(rng, v); }
      }
      std::vector<std::shared_ptr<FermionField>> lhs5b(nB);
      for (int b = 0; b < nB; b++) {
        int nVec = std::min(Nsimd, sizeL - b * Nsimd);
        lhs5b[b] = std::make_shared<FermionField>(g5.get());
        (*lhs5b[b]) = Zero();
        pack5d(*lhs5b[b], vecsL2.data() + b * Nsimd, nVec, grid, g5.get());
      }
      Eigen::Tensor<ComplexD, 5> mAlt(1, 1, Nt, sizeL, sizeL);
      mAlt.setZero();
      {
        A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, oneGamma, Up,
                                           orthogDir, g5);
        w.StagMesonField(mAlt, vecsL.data(), vecsR.data(), sizeL, sizeL); // 4D #1
        w.StagMesonField(mAlt, lhs5b, vecsR.data(), sizeL, sizeL);        // 5D
        w.StagMesonField(mAlt, vecsL.data(), vecsR.data(), sizeL, sizeL); // 4D #2, same addr
      }
      // Must equal the vecsL 4D reference (mf4), NOT vecsL2's result.
      nFail += compare(mAlt, mf4, "4D->5D->4D alternation re-sets inputs");
    }
  }

  if (nFail > 0) {
    std::cerr << "testZeroCopy5dInputs: " << nFail << " failures!"
              << std::endl;
    GridAbort();
  }
  std::cout << GridLogMessage << "testZeroCopy5dInputs: ALL PASSED"
            << std::endl;
  return nFail;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // ---- Command-line options ----
  std::string gaugeFile = GridCmdOptionPayload(argv, argv + argc, "--gauge");
  std::string nevecStr  = GridCmdOptionPayload(argv, argv + argc, "--nevec");
  int nevec = nevecStr.empty() ? 8 : std::stoi(nevecStr);

  // ---- Phase 1: 5D stencil gather exhaustive cshift test ----
  testStencilGather5d();

  // ---- Grids (production-scale, via --grid / --mpi) ----
  Coordinate fdim = GridDefaultLatt();
  GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(
      fdim, GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  int Nt = fdim[Tdir];
  int orthogDir = Tdir;

  std::cout << GridLogMessage << "Test_a2a_stencil: lattice " << fdim
            << ", nevec=" << nevec << std::endl;

  // ---- Gauge field: read from --gauge (ILDG) or generate random ----
  LatticeGaugeField U(grid);
  {
    if (!gaugeFile.empty()) {
      FieldMetaData header;
      IldgReader reader;
      reader.open(gaugeFile);
      reader.readConfiguration(U, header);
      reader.close();
      std::cout << GridLogMessage << "Gauge field read from " << gaugeFile
                << std::endl;
    } else {
      // Random gauge field. This is a DIFFERENTIAL test (stencil path vs
      // cshift oracle on the SAME input), so a random (non-SU(3)) field is
      // sufficient: both paths do the identical covariant-shift arithmetic.
      // Use a fixed seed for reproducibility.
      U = Zero();
      GridParallelRNG rng(grid);
      std::vector<int> seed = {10, 20, 30, 40};
      rng.SeedFixedIntegers(seed);
      random(rng, U);
      std::cout << GridLogMessage
                << "Gauge field: random (differential test)" << std::endl;
    }
  }

  // Parity field for the explicit-eps side of the applyG5 cross-check.
  Lattice<iScalar<vInteger>> parity(grid);
  makeParityField(parity);

  // ---- Random fermion vectors (full grid) ----
  std::vector<FermionField> vecs(nevec, grid);
  {
    GridParallelRNG rng(grid);
    std::vector<int> seed = {1, 2, 3, 4};
    rng.SeedFixedIntegers(seed);
    for (auto &v : vecs) {
      v = Zero();
      random(rng, v);
    }
  }
  std::cout << GridLogMessage << "Generated " << nevec
            << " random fermion vectors" << std::endl;

  // ---- Stencil vs Cshift oracle, all popcount 0-4 ----
  auto gammas = testGammas();
  int nStencilFail = 0;
  double stencilMaxErrAll = 0.0;

  for (int gi = 0; gi < (int)gammas.size(); gi++) {
    auto &gamma = gammas[gi];
    StagGamma spinTaste;
    spinTaste.setSpinTaste(gamma);
    int pc = StagGamma::popcountShift(spinTaste._spin, spinTaste._taste);

    std::cout << GridLogMessage
              << "=== gamma " << StagGamma::GetName(gamma)
              << " (popcount " << pc << ") ===" << std::endl;

    // Plain (unfolded) single-gamma operator set: the main/CB equality
    // checks compare like-for-like (both workers on the same objects).
    std::vector<StagGamma> oneGamma;
    {
      StagGamma st;
      st.setGaugeField(U);
      st.setSpinTaste(gamma);
      oneGamma.push_back(st);
    }
    std::vector<ComplexField> emptyMom;

    // ---- Cshift oracle (A2AWorkerSpinTaste) on the full grid ----
    // Passing the same full-grid array for all 4 CB args gives the simple
    // mat(0,0,t,i,j) full-grid layout (no 2x2 CB block).
    Eigen::Tensor<ComplexD, 5> mf_st(1, 1, Nt, nevec, nevec);
    mf_st.setZero();
    {
      A2AWorkerSpinTaste<FImpl> worker(grid, emptyMom, oneGamma, &U, orthogDir);
      worker.StagMesonField(mf_st, vecs.data(), vecs.data(),
                            vecs.data(), vecs.data());
    }

    // ---- Stencil path (A2AWorkerSpinTasteStencil) on the full grid ----
    // Alternate the call shape per gamma: even-index gammas use the default
    // ContractType argument, odd-index gammas pass ContractType::Full
    // explicitly -- both must take the identical Full path (coverage for the
    // canonical entry's contract parameter).
    Eigen::Tensor<ComplexD, 5> mf_stencil(1, 1, Nt, nevec, nevec);
    mf_stencil.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> sw(grid, emptyMom, oneGamma, &U,
                                         orthogDir);
      if (gi % 2 == 0) {
        sw.StagMesonField(mf_stencil, vecs.data(), vecs.data(), nevec, nevec);
      } else {
        sw.StagMesonField(mf_stencil, vecs.data(), vecs.data(), nevec, nevec,
                          ContractType::Full);
      }
    }

    // ---- Direct element-by-element comparison ----
    // Same flat-index layout for both: mat_idx = r + sizeR*(l + sizeL*t),
    // so mf(0,0,t,i,j) maps to the same physics M[l=i,r=j,t] in both.
    double stencilMaxErr = 0.0;
    for (int t = 0; t < Nt; t++)
      for (int i = 0; i < nevec; i++)
        for (int j = 0; j < nevec; j++) {
          ComplexD cv = mf_st(0, 0, t, i, j);
          ComplexD sv = mf_stencil(0, 0, t, i, j);
          auto mag = [](ComplexD z) {
            return std::sqrt(z.real()*z.real() + z.imag()*z.imag());
          };
          double denom = std::max(mag(cv), mag(sv));
          double e = (denom > 0.0) ? mag(cv - sv) / denom : mag(cv - sv);
          stencilMaxErr = std::max(stencilMaxErr, e);
        }

    stencilMaxErrAll = std::max(stencilMaxErrAll, stencilMaxErr);
    if (stencilMaxErr > 1e-10) {
      std::cerr << "    STENCIL vs CSHIFT FAIL gamma "
                << StagGamma::GetName(gamma) << " (popcount " << pc
                << ") maxRelErr=" << stencilMaxErr << std::endl;
      nStencilFail++;
    } else {
      std::cout << GridLogMessage << "    STENCIL vs CSHIFT OK gamma "
                << StagGamma::GetName(gamma) << " (popcount " << pc
                << ") maxRelErr=" << stencilMaxErr << std::endl;
    }
  }

  if (nStencilFail > 0) {
    std::cerr << "testStencilVsCshiftSpinTaste: " << nStencilFail
              << " failures!" << std::endl;
    GridAbort();
  }
  std::cout << GridLogMessage
            << "testStencilVsCshiftSpinTaste: ALL PASSED"
            << " (maxRelErr=" << stencilMaxErrAll << ")" << std::endl;

  // ---- CB half/mixed differential: stencil (packed full objects + ct) vs
  // legacy A2AWorkerSpinTaste (CB-grid arrays, 4-arg entry). The stencil
  // emits RAW parity partials in one uniform block layout -- (2nl, nr)
  // with rows [0,nl) = M0 = <e|.> and rows [nl,2nl) = M1 = <o|.> -- for
  // every CB mode; the legacy combined slots are RECONSTRUCTED test-side
  // via reconstructLegacySlot() and compared value-for-value. Packed
  // objects are built with setCheckerboard (E copy on even sites, O copy on
  // odd) -- the same convention Test_a2a uses for w_full.
  int nCbFail = 0;
  double cbMaxErrAll = 0.0;
  {
    GridRedBlackCartesian *cbGrid =
        SpaceTimeGrid::makeFourDimRedBlackGrid(grid);
    int nl = nevec, nr = nevec;

    std::vector<FermionField> lhsE(nl, cbGrid), lhsO(nl, cbGrid);
    std::vector<FermionField> rhsE(nr, cbGrid), rhsO(nr, cbGrid);
    for (auto &e : lhsE) e.Checkerboard() = Even;
    for (auto &e : lhsO) e.Checkerboard() = Odd;
    for (auto &e : rhsE) e.Checkerboard() = Even;
    for (auto &e : rhsO) e.Checkerboard() = Odd;
    {
      // RNG on the FULL grid (not cbGrid): GridParallelRNG::fill recurses
      // through a temporary on the RNG's own grid for CB lattices
      // (Lattice_rng.h), so a CB-grid RNG would recurse forever. With a
      // full-grid RNG, random(rng, cbField) fills a full-grid temp and
      // pickCheckerboards it into the CB field -- values are well-formed.
      GridParallelRNG rngCb(grid);
      std::vector<int> seed = {5, 6, 7, 8};
      rngCb.SeedFixedIntegers(seed);
      for (auto &e : lhsE) { e = Zero(); random(rngCb, e); }
      for (auto &e : lhsO) { e = Zero(); random(rngCb, e); }
      for (auto &e : rhsE) { e = Zero(); random(rngCb, e); }
      for (auto &e : rhsO) { e = Zero(); random(rngCb, e); }
    }

    // Packed full objects: E copy on even sites, O copy on odd sites.
    std::vector<FermionField> lhsPack(nl, grid), rhsPack(nr, grid);
    for (int k = 0; k < nl; k++) {
      lhsPack[k] = Zero();
      setCheckerboard(lhsPack[k], lhsE[k]);
      setCheckerboard(lhsPack[k], lhsO[k]);
    }
    for (int k = 0; k < nr; k++) {
      rhsPack[k] = Zero();
      setCheckerboard(rhsPack[k], rhsE[k]);
      setCheckerboard(rhsPack[k], rhsO[k]);
    }

    std::vector<ComplexField> emptyMom;

    struct CbMode {
      const char *name;
      ContractType ct;
      int legL, legR; // LEGACY-oracle output multipliers on (lhs, rhs);
                      // the stencil output is always (2*nl, nr)
    } modes[3] = {
        {"BothHalf", ContractType::BothHalf, 2, 2},
        {"LeftHalf",  ContractType::LeftHalf,  2, 1},
        {"RightHalf", ContractType::RightHalf, 1, 2},
    };

    for (int gi = 0; gi < (int)gammas.size(); gi++) {
      auto &gamma = gammas[gi];
      StagGamma spinTaste;
      spinTaste.setSpinTaste(gamma);
      int pc = StagGamma::popcountShift(spinTaste._spin, spinTaste._taste);
      // Plain (unfolded) single-gamma operator set (like-for-like).
      std::vector<StagGamma> oneGamma;
      {
        StagGamma st;
        st.setGaugeField(U);
        st.setSpinTaste(gamma);
        oneGamma.push_back(st);
      }

      // Stencil worker: one per gamma (kept alive across modes).
      A2AWorkerSpinTasteStencil<FImpl> sw(grid, emptyMom, oneGamma, &U,
                                          orthogDir);

      for (int m = 0; m < 3; m++) {
        CbMode &mode = modes[m];
        std::cout << GridLogMessage << "=== CB " << mode.name << " gamma "
                  << StagGamma::GetName(gamma) << " (popcount " << pc
                  << ") ===" << std::endl;

        // Legacy oracle (4-arg E/O entry). Mixed modes pass the full-grid
        // array twice for the non-CB side (only the E array is read there).
        // Created per mode to avoid _transformed vector reuse issues in
        // the legacy task's setRight.
        // RowMajor: the workers memcpy flat row-major device buffers
        // into mat.data(), so operator() maps 1:1 onto the worker
        // slots (default ColMajor would permute (t,l,r) across shapes).
        Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> mf_leg(
            1, 1, Nt, mode.legL * nl, mode.legR * nr);
        mf_leg.setZero();
        {
          A2AWorkerSpinTaste<FImpl> w(grid, emptyMom, oneGamma, &U, orthogDir);
          if (mode.ct == ContractType::BothHalf) {
            w.StagMesonField(mf_leg, lhsE.data(), lhsO.data(), rhsE.data(),
                             rhsO.data());
          } else if (mode.ct == ContractType::LeftHalf) {
            w.StagMesonField(mf_leg, lhsE.data(), lhsO.data(), vecs.data(),
                             vecs.data());
          } else {
            w.StagMesonField(mf_leg, vecs.data(), vecs.data(), rhsE.data(),
                             rhsO.data());
          }
        }

        // Stencil path (packed full objects + explicit ct): uniform raw
        // block layout (2*nl, nr) for every CB mode. RowMajor so
        // operator() matches the worker's flat row-major write (see the
        // mf_leg note above).
        Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> mf_stn(1, 1, Nt, 2 * nl, nr);
        mf_stn.setZero();
        if (mode.ct == ContractType::BothHalf) {
          sw.StagMesonField(mf_stn, lhsPack.data(), rhsPack.data(), nl, nr,
                            ContractType::BothHalf);
        } else if (mode.ct == ContractType::LeftHalf) {
          sw.StagMesonField(mf_stn, lhsPack.data(), vecs.data(), nl, nr,
                            ContractType::LeftHalf);
        } else {
          sw.StagMesonField(mf_stn, vecs.data(), rhsPack.data(), nl, nr,
                            ContractType::RightHalf);
        }

        // Reconstruct-then-diff: rebuild each legacy slot from the raw
        // partials and compare against the frozen oracle. Each tensor is
        // indexed with its OWN dims (legacy interleaved vs stencil block).
        double cbErr = 0.0;
        ComplexD sig = (pc & 1) ? ComplexD(-1.0) : ComplexD(1.0);
        for (int t = 0; t < Nt; t++)
          for (int l = 0; l < nl; l++)
            for (int r = 0; r < nr; r++) {
              ComplexD M0 = mf_stn(0, 0, t, l, r);
              ComplexD M1 = mf_stn(0, 0, t, nl + l, r);
              for (int lc = 0; lc < mode.legL; lc++)
                for (int rc = 0; rc < mode.legR; rc++) {
                  int iLeg = (mode.legL == 2) ? 2 * l + lc : l;
                  int jLeg = (mode.legR == 2) ? 2 * r + rc : r;
                  ComplexD lv = mf_leg(0, 0, t, iLeg, jLeg);
                  ComplexD sv =
                      reconstructLegacySlot(mode.ct, lc, rc, M0, M1, sig);
                  auto mag = [](ComplexD z) {
                    return std::sqrt(z.real() * z.real() +
                                     z.imag() * z.imag());
                  };
                  double denom = std::max(mag(lv), mag(sv));
                  double e = (denom > 0.0) ? mag(lv - sv) / denom
                                           : mag(lv - sv);
                  cbErr = std::max(cbErr, e);
                }
            }
        cbMaxErrAll = std::max(cbMaxErrAll, cbErr);
        if (cbErr > 1e-10) {
          std::cerr << "    CB " << mode.name << " FAIL gamma "
                    << StagGamma::GetName(gamma) << " (popcount " << pc
                    << ") maxRelErr=" << cbErr << std::endl;
          nCbFail++;
        } else {
          std::cout << GridLogMessage << "    CB " << mode.name
                    << " OK gamma " << StagGamma::GetName(gamma)
                    << " (popcount " << pc << ") maxRelErr=" << cbErr
                    << std::endl;
        }

        // Physics consistency (oracle-independent): the raw parity partials
        // of ANY CB mode must sum to the Full contraction of that mode's
        // inputs -- M0 + M1 covers even+odd source sites, i.e. all sites.
        // BothHalf: Full(lhsPack, rhsPack); LeftHalf: Full(lhsPack, vecs);
        // RightHalf: Full(vecs, rhsPack). This is the independent brake on
        // the reconstruction: if the kernel output and reconstructLegacySlot
        // shared one wrongly-derived sign table, the differential above
        // would self-confirm; this check cannot. Reuses the per-gamma
        // worker (address cache: same input arrays as the CB call above;
        // grow-only mat cache: Full (nl,nr) fits in the (2nl,nr) slot).
        {
          // RowMajor: matches mf_stn/mf_leg — operator() reads land on
          // the worker's flat row-major slots.
          Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> mf_full(1, 1, Nt, nl, nr);
          mf_full.setZero();
          if (mode.ct == ContractType::BothHalf) {
            sw.StagMesonField(mf_full, lhsPack.data(), rhsPack.data(), nl, nr);
          } else if (mode.ct == ContractType::LeftHalf) {
            sw.StagMesonField(mf_full, lhsPack.data(), vecs.data(), nl, nr);
          } else {
            sw.StagMesonField(mf_full, vecs.data(), rhsPack.data(), nl, nr);
          }
          double physErr = 0.0;
          for (int t = 0; t < Nt; t++)
            for (int l = 0; l < nl; l++)
              for (int r = 0; r < nr; r++) {
                ComplexD a = mf_full(0, 0, t, l, r);
                ComplexD b = mf_stn(0, 0, t, l, r) +
                             mf_stn(0, 0, t, nl + l, r); // M0 + M1
                auto mag = [](ComplexD z) {
                  return std::sqrt(z.real() * z.real() + z.imag() * z.imag());
                };
                double denom = std::max(mag(a), mag(b));
                double e = (denom > 0.0) ? mag(a - b) / denom : mag(a - b);
                physErr = std::max(physErr, e);
              }
          if (physErr > 1e-10) {
            std::cerr << "    CB " << mode.name << " physics FAIL gamma "
                      << StagGamma::GetName(gamma)
                      << " maxRelErr=" << physErr << std::endl;
            nCbFail++;
          } else {
            std::cout << GridLogMessage << "    CB " << mode.name
                      << " physics OK gamma " << StagGamma::GetName(gamma)
                      << " maxRelErr=" << physErr << std::endl;
          }
        }
      }
    }

    delete cbGrid;
  }
  if (nCbFail > 0) {
    std::cerr << "testStencilCbVsLegacy: " << nCbFail << " failures!"
              << std::endl;
    GridAbort();
  }
  std::cout << GridLogMessage << "testStencilCbVsLegacy: ALL PASSED"
            << " (maxRelErr=" << cbMaxErrAll << ")" << std::endl;

  // ---- applyG5 cross-check: stencil worker on eps-folded objects vs
  // legacy worker on plain objects with EXPLICIT eps on the LEFT vectors.
  // eps is real-diagonal: <eps*L, Gamma*R> == <L, (eps∘Gamma)*R>, so the
  // two sides derive eps from different constructions (folded _negated vs
  // explicit where() multiply) and cannot self-confirm (handoff §3; the
  // identity is exact only on the LEFT side -- Γ∘eps != eps∘Γ under
  // displacement).
  int nEpsFail = 0;
  double epsMaxErrAll = 0.0;
  {
    // Explicit-eps left vectors (full grid).
    std::vector<FermionField> vecsEps(nevec, grid);
    for (int k = 0; k < nevec; k++) {
      vecsEps[k] = vecs[k];
      applyEpsilon(vecsEps[k], parity);
    }

    for (int gi = 0; gi < (int)gammas.size(); gi++) {
      auto &gamma = gammas[gi];
      StagGamma spinTaste;
      spinTaste.setSpinTaste(gamma);
      int pc = StagGamma::popcountShift(spinTaste._spin, spinTaste._taste);

      std::cout << GridLogMessage << "=== applyG5 cross-check gamma "
                << StagGamma::GetName(gamma) << " (popcount " << pc
                << ") ===" << std::endl;

      // eps-folded stencil worker (the production applyG5 path).
      std::vector<StagGamma> oneGammaFolded;
      {
        StagGamma st;
        st.setGaugeField(U);
        st.setSpinTaste(gamma);
        st.applyG5Left();
        oneGammaFolded.push_back(st);
      }
      Eigen::Tensor<ComplexD, 5> mf_eps(1, 1, Nt, nevec, nevec);
      mf_eps.setZero();
      {
        std::vector<ComplexField> emptyMom;
        A2AWorkerSpinTasteStencil<FImpl> sw(grid, emptyMom, oneGammaFolded,
                                            &U, orthogDir);
        sw.StagMesonField(mf_eps, vecs.data(), vecs.data(), nevec, nevec);
      }

      // Explicit-eps oracle: legacy worker, plain Gamma, eps-multiplied left.
      Eigen::Tensor<ComplexD, 5> mf_ref(1, 1, Nt, nevec, nevec);
      mf_ref.setZero();
      {
        std::vector<StagGamma> oneGammaPlain;
        {
          StagGamma st;
          st.setGaugeField(U);
          st.setSpinTaste(gamma);
          oneGammaPlain.push_back(st);
        }
        std::vector<ComplexField> emptyMom;
        A2AWorkerSpinTaste<FImpl> w(grid, emptyMom, oneGammaPlain, &U,
                                    orthogDir);
        w.StagMesonField(mf_ref, vecsEps.data(), vecsEps.data(),
                         vecs.data(), vecs.data());
      }

      double epsErr = 0.0;
      for (int t = 0; t < Nt; t++)
        for (int i = 0; i < nevec; i++)
          for (int j = 0; j < nevec; j++) {
            ComplexD ev = mf_eps(0, 0, t, i, j);
            ComplexD rv = mf_ref(0, 0, t, i, j);
            auto mag = [](ComplexD z) {
              return std::sqrt(z.real() * z.real() + z.imag() * z.imag());
            };
            double denom = std::max(mag(ev), mag(rv));
            double e = (denom > 0.0) ? mag(ev - rv) / denom : mag(ev - rv);
            epsErr = std::max(epsErr, e);
          }
      epsMaxErrAll = std::max(epsMaxErrAll, epsErr);
      if (epsErr > 1e-10) {
        std::cerr << "    applyG5 cross-check FAIL gamma "
                  << StagGamma::GetName(gamma) << " (popcount " << pc
                  << ") maxRelErr=" << epsErr << std::endl;
        nEpsFail++;
      } else {
        std::cout << GridLogMessage << "    applyG5 cross-check OK gamma "
                  << StagGamma::GetName(gamma) << " (popcount " << pc
                  << ") maxRelErr=" << epsErr << std::endl;
      }
    }
  }
  if (nEpsFail > 0) {
    std::cerr << "testApplyG5CrossCheck: " << nEpsFail << " failures!"
              << std::endl;
    GridAbort();
  }
  std::cout << GridLogMessage << "testApplyG5CrossCheck: ALL PASSED"
            << " (maxRelErr=" << epsMaxErrAll << ")" << std::endl;

  // ---- Zero-copy 5D input path (shared_ptr borrow + grid adoption) ----
  int nZcFail = testZeroCopy5dInputs(grid, &U, nevec);

  // ---- Result ----
  std::cout << GridLogMessage << "=== Test_a2a_stencil complete ===" << std::endl;
  bool ok = (nStencilFail == 0 && nCbFail == 0 && nEpsFail == 0 &&
             nZcFail == 0);
  if (ok) {
    std::cout << GridLogMessage << "PASS" << std::endl;
  } else {
    std::cerr << "FAIL" << std::endl;
  }

  Grid_finalize();
  return ok ? 0 : 1;
}
