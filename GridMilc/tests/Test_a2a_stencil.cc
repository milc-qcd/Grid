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

    std::vector<StagGamma::SpinTastePair> oneGamma = {gamma};
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
      std::vector<StagGamma::SpinTastePair> oneGamma = {gamma};

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

  // ---- Result ----
  std::cout << GridLogMessage << "=== Test_a2a_stencil complete ===" << std::endl;
  bool ok = (nStencilFail == 0 && nCbFail == 0);
  if (ok) {
    std::cout << GridLogMessage << "PASS" << std::endl;
  } else {
    std::cerr << "FAIL" << std::endl;
  }

  Grid_finalize();
  return ok ? 0 : 1;
}
