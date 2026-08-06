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

  GridCartesian *grid5d = createGrid5d(&grid4d);
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

  delete grid5d;
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
    Eigen::Tensor<ComplexD, 5> mf_stencil(1, 1, Nt, nevec, nevec);
    mf_stencil.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> sw(grid, emptyMom, oneGamma, &U,
                                         orthogDir);
      sw.StagMesonFieldStencil(mf_stencil, vecs.data(), vecs.data(),
                               nevec, nevec);
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

  // ---- Result ----
  std::cout << GridLogMessage << "=== Test_a2a_stencil complete ===" << std::endl;
  bool ok = (nStencilFail == 0);
  if (ok) {
    std::cout << GridLogMessage << "PASS" << std::endl;
  } else {
    std::cerr << "FAIL" << std::endl;
  }

  Grid_finalize();
  return ok ? 0 : 1;
}
