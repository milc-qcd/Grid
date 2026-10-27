/*
 * GridMilc/tests/Test_a2a_stencil_ja_mom.cc
 *
 * Oracle tests for the momentum-projection and J.A (local + one-link)
 * interpolators of the 5D stencil A2A path.
 *
 * All oracles compare against the EXISTING stencil A2A worker with
 * contrived inputs (developer decision: the legacy ASlash code is not
 * assumed correct; no legacy worker appears in this file), plus explicit
 * lattice-expression references:
 *
 *   (1) conj(ph)·L fold: every stencil site phase multiplies the
 *       source-site bra contraction (innerProduct(l,r) = conjugate(l)*r,
 *       Grid_scalar_types.h:522), so mf(g,p) == momEmptyWorker(
 *       conjugate(ph[p])·L, R)(g) exactly (to rounding) — checked at
 *       popcounts 0-4, at blocking-seam-crossing gamma/momentum counts,
 *       and J.A local / one-link + momentum.
 *   (2) Explicit lattice-expression reference (pc0): the
 *       site weight w(x) = chan_a(x)·ph_p(x) is placed on the RIGHT:
 *       sliceInnerProductVector(L, w·R) = sum_x w(x)·<L(x)|R_r(x)> —
 *       exact because w commutes through conj(L).
 *   (3) Blocking-seam stress: 20 gammas x 2 momenta (3 g-blocks) and
 *       1 gamma x 20 momenta (2 p-blocks).
 *   (4) Zero-momentum equivalence: mom={ones} == mom={} EXACTLY (the x1.0
 *       multiply is IEEE-exact).
 *   (5) J.A local contrived: Amu[a] = applyCoeffsAndPhase(pc0 gammas,
 *       ones) == existing stencil pc0 path exactly; random Amu vs (2);
 *       J.A local + momentum via (1).
 *   (6) J.A one-link contrived: random A_k fields; the J.A one-link
 *       worker == existing stencil pc1 path on a MANUALLY A-dressed gauge
 *       field (exact); independent Cshift reference for the X component;
 *       zero T-component => identically-zero T channel (3-vector
 *       convention); + momentum via (1).
 *   (7) Physics: with momentum, CB LeftHalf raw partials satisfy
 *       M0+M1 == Full (oracle-independent).
 *
 * Usage: Test_a2a_stencil_ja_mom [--grid W.X.Y.Z] [--mpi a.b.c.d]
 *                                [--nevec N]  (default: 4)
 */
#include <Grid/Grid.h>
#include <Grid/qcd/utils/SpaceTimeGrid.h>
#include <GridMilc/GridMilc.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

using namespace Grid;

typedef StaggeredImplD FImpl;
typedef FImpl::FermionField FermionField;
typedef FImpl::ComplexField ComplexField;

// ---- momentum phases: ph = exp(2 pi i p·x / L), spatial momenta only
//      (the MesonField.hpp:618-631 pattern). ----
static std::vector<ComplexField>
makeMomPhases(GridCartesian *grid, const std::vector<Coordinate> &ps) {
  std::vector<ComplexField> ph;
  for (const auto &p : ps) {
    ph.emplace_back(grid);
    ComplexField &f = ph.back();
    Complex i(0.0, 1.0);
    ComplexField coor(grid);
    f = Zero();
    for (int mu = 0; mu < Nd - 1; mu++) {
      LatticeCoordinate(coor, mu);
      f = f + ((Real)p[mu] / (Real)grid->GlobalDimensions()[mu]) * coor;
    }
    f = exp((Real)(2 * M_PI) * i * f);
  }
  return ph;
}

// ---- gamma helpers (mirrors the Test_a2a_stencil patterns) ----
static std::vector<StagGamma::SpinTastePair> momTestGammas() {
  typedef StagGamma::StagAlgebra A;
  return {
      {A(A::G1), A(A::G1)},  // pc0
      {A(A::GX), A(A::G1)},  // pc1
      {A(A::GXY), A(A::G1)}, // pc2
      {A(A::G5T), A(A::G1)}, // pc3
      {A(A::G5), A(A::G1)},  // pc4
  };
}

static std::vector<StagGamma>
makeOps(const std::vector<StagGamma::SpinTastePair> &pairs,
        LatticeGaugeField &U) {
  std::vector<StagGamma> ops;
  ops.reserve(pairs.size());
  for (const auto &pr : pairs) {
    StagGamma st;
    st.setGaugeField(U);
    st.setSpinTaste(pr);
    ops.push_back(st);
  }
  return ops;
}

// FOLD: L'(x) = conj(c(x))·L(x) so <L'|X> = c·<L|X> (site-diagonal c,
// conj-on-left inner product).
static void foldConj(const ComplexField &c, const FermionField *L,
                     FermionField *Lout, int n) {
  ComplexField cc = conjugate(c);
  for (int i = 0; i < n; i++)
    Lout[i] = cc * L[i];
}

// Run the mom-empty stencil oracle once per folded momentum and return the
// per-p (nSpec, 1, ...) tensors (dims 3/4 sized per ct: Full (L, R), CB
// interleaved (2L, R) / (2L, 2R)).
static std::vector<Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>>
runFoldOracles(GridCartesian *grid, const std::vector<ComplexField> &phIn,
               const std::vector<StagGamma> &gammas, LatticeGaugeField *U,
               const std::vector<FermionField> &vecsL,
               const std::vector<FermionField> &vecsR, int nevec, int Nt,
               int orthogDir,
               ContractType ct = ContractType::Full) {
  int rowMult = (ct == ContractType::Full) ? 1 : 2;
  int colMult = (ct == ContractType::BothHalf ||
                 ct == ContractType::RightHalf)
                    ? 2
                    : 1;
  std::vector<ComplexField> emptyMom;
  std::vector<Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>> orc(phIn.size());
  std::vector<FermionField> foldL(nevec, grid);
  for (int p = 0; p < (int)phIn.size(); p++) {
    orc[p] = Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>(
        (int)gammas.size(), 1, Nt, rowMult * nevec, colMult * nevec);
    orc[p].setZero();
    foldConj(phIn[p], vecsL.data(), foldL.data(), nevec);
    A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, gammas, U, orthogDir);
    w.StagMesonField(orc[p], foldL.data(), vecsR.data(), nevec, nevec, ct);
  }
  return orc;
}

// max rel-err elementwise over two same-shape (nSpec, nMom, ...) tensors.
static double
compareSame(const Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> &a,
            const Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> &b) {
  double err = 0.0;
  for (int g = 0; g < (int)a.dimension(0); g++)
    for (int p = 0; p < (int)a.dimension(1); p++)
      for (int t = 0; t < (int)a.dimension(2); t++)
        for (int i = 0; i < (int)a.dimension(3); i++)
          for (int j = 0; j < (int)a.dimension(4); j++) {
            ComplexD x = a(g, p, t, i, j), y = b(g, p, t, i, j);
            double den = std::max(std::abs(x), std::abs(y));
            double e = (den > 0.0) ? std::abs(x - y) / den : std::abs(x - y);
            err = std::max(err, e);
          }
  return err;
}

// max rel-err of a full (nSpec, nMom) result vs per-p (nSpec, 1) oracles.
static double
compareMomentumVsOracles(
    const Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> &full,
    const std::vector<Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>> &orc) {
  double err = 0.0;
  for (int p = 0; p < (int)orc.size(); p++)
    for (int g = 0; g < (int)full.dimension(0); g++)
      for (int t = 0; t < (int)full.dimension(2); t++)
        for (int i = 0; i < (int)full.dimension(3); i++)
          for (int j = 0; j < (int)full.dimension(4); j++) {
            ComplexD x = full(g, p, t, i, j);
            ComplexD y = orc[p](g, 0, t, i, j);
            double den = std::max(std::abs(x), std::abs(y));
            double e = (den > 0.0) ? std::abs(x - y) / den : std::abs(x - y);
            err = std::max(err, e);
          }
  return err;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::string nevecStr = GridCmdOptionPayload(argv, argv + argc, "--nevec");
  int nevec = nevecStr.empty() ? 4 : std::stoi(nevecStr);

  Coordinate fdim = GridDefaultLatt();
  GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(
      fdim, GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  int Nt = fdim[Tdir];
  int orthogDir = Tdir;

  std::cout << GridLogMessage << "Test_a2a_stencil_ja_mom: lattice " << fdim
            << ", nevec=" << nevec << std::endl;

  // Random gauge + vectors (contrived-field differential tests: random
  // non-SU(3) gauge is fine — every comparison is same-input).
  LatticeGaugeField U(grid);
  {
    GridParallelRNG rng(grid);
    std::vector<int> sd = {10, 20, 30, 40};
    rng.SeedFixedIntegers(sd);
    U = Zero();
    random(rng, U);
  }
  std::vector<FermionField> vecsL(nevec, grid), vecsR(nevec, grid);
  {
    GridParallelRNG rng(grid);
    std::vector<int> sd = {1, 2, 3, 4};
    rng.SeedFixedIntegers(sd);
    for (auto &v : vecsL) { v = Zero(); random(rng, v); }
    for (auto &v : vecsR) { v = Zero(); random(rng, v); }
  }
  std::vector<ComplexField> emptyMom;

  // 3 non-trivial spatial momenta.
  std::vector<Coordinate> ps;
  ps.push_back(Coordinate(std::vector<int>{1, 0, 0, 0}));
  ps.push_back(Coordinate(std::vector<int>{0, 1, 0, 0}));
  ps.push_back(Coordinate(std::vector<int>{0, 0, 2, 0}));
  std::vector<ComplexField> ph = makeMomPhases(grid, ps);
  int nMom = (int)ph.size();

  int nFail = 0;
  auto check = [&](const char *what, double err, double tol) {
    if (err > tol) {
      std::cerr << "    FAIL " << what << " maxRelErr=" << err << std::endl;
      nFail++;
    } else {
      std::cout << GridLogMessage << "    OK " << what
                << " (maxRelErr=" << err << ")" << std::endl;
    }
  };

  // (1) conj(ph)·L fold, popcounts 0-4 ----------------------------------
  {
    auto gammas = makeOps(momTestGammas(), U);
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> mf(
        (int)gammas.size(), nMom, Nt, nevec, nevec);
    mf.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, ph, gammas, &U, orthogDir);
      w.StagMesonField(mf, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    auto orc = runFoldOracles(grid, ph, gammas, &U, vecsL, vecsR, nevec, Nt,
                              orthogDir);
    check("momentum conj-fold vs existing stencil (pc0-4)",
          compareMomentumVsOracles(mf, orc), 1e-13);

    // (1b) the same fold comparison under ContractType::LeftHalf: both
    // sides emit the legacy interleaved (2L, R) slots post-merge, so the
    // comparison is element-for-element (no M0+M1 reconstruction).
    // Restored 2026-10-13 — the suspected CB deferred-write defect that
    // deferred this check is gone after 62e9fec5 (probe: 3.6e-14).
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> mfl(
        (int)gammas.size(), nMom, Nt, 2 * nevec, nevec);
    mfl.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, ph, gammas, &U, orthogDir);
      w.StagMesonField(mfl, vecsL.data(), vecsR.data(), nevec, nevec,
                       ContractType::LeftHalf);
    }
    auto orcl = runFoldOracles(grid, ph, gammas, &U, vecsL, vecsR, nevec, Nt,
                               orthogDir, ContractType::LeftHalf);
    // 1e-12 (not 1e-13): the odd (partial-batch) nevec run reaches
    // 1.7e-13 — same rounding-scale adaptation as the Cshift reference.
    check("momentum conj-fold under LeftHalf (1b)",
          compareMomentumVsOracles(mfl, orcl), 1e-12);
  }

  // (2) explicit lattice-expression reference (pc0) ---------------------
  {
    typedef StagGamma::StagAlgebra A;
    std::vector<StagGamma::SpinTastePair> pc0pairs = {
        {A(A::G1), A(A::G1)}, {A(A::GX), A(A::GX)}};
    auto gammas = makeOps(pc0pairs, U);
    int nG = (int)gammas.size();

    ComplexField ones(grid);
    ones = 1.0;
    std::vector<ComplexField> chan(nG, grid);
    for (int a = 0; a < nG; a++)
      gammas[a].applyCoeffsAndPhase(chan[a], ones);

    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> mf(nG, nMom, Nt, nevec, nevec);
    mf.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, ph, gammas, &U, orthogDir);
      w.StagMesonField(mf, vecsL.data(), vecsR.data(), nevec, nevec);
    }

    // w on the RIGHT: sliceInnerProductVector(L, w·R) = sum_x w·<L|R>
    // (innerProduct conjugates the LEFT argument; w commutes through).
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> ref(nG, nMom, Nt, nevec, nevec);
    ref.setZero();
    for (int a = 0; a < nG; a++)
      for (int p = 0; p < nMom; p++) {
        ComplexField w4 = chan[a] * ph[p];
        for (int r = 0; r < nevec; r++) {
          FermionField O = w4 * vecsR[r];
          for (int l = 0; l < nevec; l++) {
            std::vector<ComplexD> m;
            sliceInnerProductVector(m, vecsL[l], O, orthogDir);
            for (int t = 0; t < Nt; t++)
              ref(a, p, t, l, r) = m[t];
          }
        }
      }
    check("momentum explicit reference (pc0)", compareSame(mf, ref), 1e-13);
  }

  // (3) blocking-seam stress --------------------------------------------
  {
    // (3a) 20 gammas x 2 momenta: nPBlock=2, nGBlockMax=8, 3 g-blocks.
    std::vector<StagGamma::SpinTastePair> many = momTestGammas();
    for (int rep = 0; rep < 3; rep++)
      for (auto &pr : momTestGammas()) many.push_back(pr);
    auto gammas = makeOps(many, U);
    std::vector<ComplexField> ph2 = {ph[0], ph[2]};
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> mf(
        (int)gammas.size(), 2, Nt, nevec, nevec);
    mf.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, ph2, gammas, &U, orthogDir);
      w.StagMesonField(mf, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    auto orc = runFoldOracles(grid, ph2, gammas, &U, vecsL, vecsR, nevec, Nt,
                              orthogDir);
    check("momentum 20-gamma multi-g-block",
          compareMomentumVsOracles(mf, orc), 1e-13);

    // (3b) 1 gamma x 20 momenta: nPBlock=16, 2 p-blocks.
    std::vector<Coordinate> ps20;
    for (int k = 0; k < 20; k++) {
      Coordinate pk(std::vector<int>{k % 3, (k / 3) % 3, (k / 9) % 3, 0});
      ps20.push_back(pk);
    }
    std::vector<ComplexField> ph20 = makeMomPhases(grid, ps20);
    auto g1 = makeOps({momTestGammas()[1]}, U); // pc1
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> m20(1, 20, Nt, nevec, nevec);
    m20.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, ph20, g1, &U, orthogDir);
      w.StagMesonField(m20, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    auto orc20 = runFoldOracles(grid, ph20, g1, &U, vecsL, vecsR, nevec, Nt,
                                orthogDir);
    check("momentum 20-momentum multi-p-block",
          compareMomentumVsOracles(m20, orc20), 1e-12);
  }

  // (4) zero-momentum equivalence: mom={ones} == mom={} EXACTLY ---------
  {
    auto gammas = makeOps(momTestGammas(), U);
    ComplexField oneF(grid);
    oneF = 1.0;
    std::vector<ComplexField> oneMom = {oneF};
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>
        a((int)gammas.size(), 1, Nt, nevec, nevec),
        b((int)gammas.size(), 1, Nt, nevec, nevec);
    a.setZero();
    b.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, oneMom, gammas, &U, orthogDir);
      w.StagMesonField(a, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, gammas, &U, orthogDir);
      w.StagMesonField(b, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    check("mom={ones} == mom={} exactly", compareSame(a, b), 0.0);
  }

  // (5) J.A local -------------------------------------------------------
  {
    typedef StagGamma::StagAlgebra A;
    std::vector<StagGamma::SpinTastePair> pc0pairs = {
        {A(A::G1), A(A::G1)}, {A(A::GX), A(A::GX)},
        {A(A::GY), A(A::GY)}, {A(A::GZ), A(A::GZ)}};
    auto pc0ops = makeOps(pc0pairs, U);
    ComplexField ones(grid);
    ones = 1.0;
    std::vector<ComplexField> contrived(pc0ops.size(), grid);
    for (int a = 0; a < (int)pc0ops.size(); a++)
      pc0ops[a].applyCoeffsAndPhase(contrived[a], ones);

    // (5a) contrived, mom empty: J.A local worker == stencil pc0 worker.
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>
        ja((int)contrived.size(), 1, Nt, nevec, nevec),
        sg((int)pc0ops.size(), 1, Nt, nevec, nevec);
    ja.setZero();
    sg.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, contrived, orthogDir);
      w.StagMesonField(ja, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    {
      // null U legal: every pc0 spec is zero-displacement (identity chain)
      A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, pc0ops, nullptr,
                                         orthogDir);
      w.StagMesonField(sg, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    check("J.A local contrived == stencil pc0", compareSame(ja, sg), 1e-13);

    // (5b) random Amu (3 channels) vs the explicit reference.
    std::vector<ComplexField> rand3(3, grid);
    {
      GridParallelRNG rng(grid);
      std::vector<int> sd = {21, 22, 23};
      rng.SeedFixedIntegers(sd);
      for (auto &f : rand3) { f = Zero(); random(rng, f); }
    }
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> jr(3, 1, Nt, nevec, nevec),
        jref(3, 1, Nt, nevec, nevec);
    jr.setZero();
    jref.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, rand3, orthogDir);
      w.StagMesonField(jr, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    for (int a = 0; a < 3; a++)
      for (int r = 0; r < nevec; r++) {
        FermionField O = rand3[a] * vecsR[r];
        for (int l = 0; l < nevec; l++) {
          std::vector<ComplexD> m;
          sliceInnerProductVector(m, vecsL[l], O, orthogDir);
          for (int t = 0; t < Nt; t++)
            jref(a, 0, t, l, r) = m[t];
        }
      }
    check("J.A local random vs explicit reference", compareSame(jr, jref),
          1e-13);

    // (5c) J.A local + momentum via the fold.
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> jp(
        (int)contrived.size(), nMom, Nt, nevec, nevec);
    jp.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, ph, contrived, orthogDir);
      w.StagMesonField(jp, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    std::vector<ComplexField> emptyMom2;
    std::vector<Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>> jorc(nMom);
    std::vector<FermionField> foldL(nevec, grid);
    for (int p = 0; p < nMom; p++) {
      jorc[p] = Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>(
          (int)contrived.size(), 1, Nt, nevec, nevec);
      jorc[p].setZero();
      foldConj(ph[p], vecsL.data(), foldL.data(), nevec);
      A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom2, contrived, orthogDir);
      w.StagMesonField(jorc[p], foldL.data(), vecsR.data(), nevec, nevec);
    }
    check("J.A local + momentum fold",
          compareMomentumVsOracles(jp, jorc), 1e-13);
  }

  // (6) J.A one-link ----------------------------------------------------
  {
    // Random photon fields, one per direction.
    std::vector<ComplexField> A4(Nd, grid);
    {
      GridParallelRNG rng(grid);
      std::vector<int> sd = {31, 32, 33, 34};
      rng.SeedFixedIntegers(sd);
      for (auto &f : A4) { f = Zero(); random(rng, f); }
    }

    // Manual dressed gauge field (TEST-side, independent of the ctor).
    LatticeGaugeField Ud(grid);
    Ud = U;
    for (int k = 0; k < Nd; k++) {
      LatticeColourMatrix link = PeekIndex<LorentzIndex>(U, k);
      link = A4[k] * link;
      PokeIndex<LorentzIndex>(Ud, link, k);
    }

    // Stencil pc1 specs = the J.A one-link specs by construction (the
    // spec objects' bound U is inert on the stencil path: the task passes
    // its OWN U argument to the chain builder).
    typedef StagGamma::StagAlgebra A;
    auto pc1ops = makeOps({{A(A::GX), A(A::G1)},
                           {A(A::GY), A(A::G1)},
                           {A(A::GZ), A(A::G1)},
                           {A(A::GT), A(A::G1)}},
                          U);
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> one(4, 1, Nt, nevec, nevec),
        orc(4, 1, Nt, nevec, nevec);
    one.setZero();
    orc.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, A4, &U, orthogDir);
      w.StagMesonField(one, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, pc1ops, &Ud,
                                         orthogDir);
      w.StagMesonField(orc, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    check("J.A one-link == stencil pc1 on dressed U", compareSame(one, orc),
          1e-13);

    // Independent Cshift reference for the X component (k=0):
    //   O(x) = ph0(x)·( W̃(x)·ψ(x+ê_x) + adj(W̃(x-ê_x))·ψ(x-ê_x) )
    // with W̃ = A_0·U_x, adj = conjugate-transpose, ph0 = the (GX,G1)
    // applyCoeffsAndPhase bake; ref = sliceInnerProductVector(L, O).
    {
      ComplexField ones(grid);
      ones = 1.0;
      ComplexField ph0(grid);
      pc1ops[0].applyCoeffsAndPhase(ph0, ones);
      LatticeColourMatrix Wd = A4[0] * PeekIndex<LorentzIndex>(U, 0);
      LatticeColourMatrix Wadj = transpose(conjugate(Wd)); // adj(W̃)
      double errX = 0.0;
      for (int r = 0; r < nevec; r++) {
        // Cshift(f, d, n)(x) = f(x + n·ê_d) (the gather convention of
        // Test_a2a_stencil.cc:202-206).
        FermionField F = Wd * Cshift(vecsR[r], 0, +1);
        FermionField B = Cshift(Wadj, 0, -1) * Cshift(vecsR[r], 0, -1);
        FermionField O = ph0 * (F + B);
        for (int l = 0; l < nevec; l++) {
          std::vector<ComplexD> m;
          sliceInnerProductVector(m, vecsL[l], O, orthogDir);
          for (int t = 0; t < Nt; t++) {
            double den = std::max(std::abs(m[t]),
                                  std::abs(one(0, 0, t, l, r)));
            errX = std::max(errX,
                            den > 0 ? std::abs(m[t] - one(0, 0, t, l, r)) / den
                                    : std::abs(m[t] - one(0, 0, t, l, r)));
          }
        }
      }
      check("J.A one-link X-component vs Cshift reference", errX, 1e-12);
    }

    // Zero-T (3-vector convention): zero T photon => T channel == 0.
    {
      std::vector<ComplexField> A3;
      A3.push_back(A4[0]);
      A3.push_back(A4[1]);
      A3.push_back(A4[2]);
      A3.emplace_back(grid);
      A3[3] = Zero();
      Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> m3(4, 1, Nt, nevec, nevec);
      m3.setZero();
      {
        A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, A3, &U, orthogDir);
        w.StagMesonField(m3, vecsL.data(), vecsR.data(), nevec, nevec);
      }
      double maxT = 0.0;
      for (int t = 0; t < Nt; t++)
        for (int i = 0; i < nevec; i++)
          for (int j = 0; j < nevec; j++)
            maxT = std::max(maxT, std::abs(m3(3, 0, t, i, j)));
      check("J.A one-link zero-T channel vanishes", maxT, 0.0);
    }

    // J.A one-link + momentum via the fold.
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> op(4, nMom, Nt, nevec, nevec);
    op.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, ph, A4, &U, orthogDir);
      w.StagMesonField(op, vecsL.data(), vecsR.data(), nevec, nevec);
    }
    std::vector<Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>> oorc(nMom);
    std::vector<FermionField> foldL(nevec, grid);
    for (int p = 0; p < nMom; p++) {
      oorc[p] = Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>(4, 1, Nt, nevec,
                                                            nevec);
      oorc[p].setZero();
      foldConj(ph[p], vecsL.data(), foldL.data(), nevec);
      A2AWorkerSpinTasteStencil<FImpl> w(grid, emptyMom, A4, &U, orthogDir);
      w.StagMesonField(oorc[p], foldL.data(), vecsR.data(), nevec, nevec);
    }
    check("J.A one-link + momentum fold",
          compareMomentumVsOracles(op, oorc), 1e-12);
  }

  // (7) physics: with momentum, CB interleaved slots vs Full --------------
  // (Post-merge contract: the CB modes emit the LEGACY INTERLEAVED (M,
  // M-dagger) layout -- raw (2L,R) parity partials from the task,
  // sigma-combined in the worker's finishMatOutput host pass. The
  // (even row, even col) slot is M0+M1 = the Full contraction; the
  // sigma-carrying slots are sig*(M0-M1) / sig*(M0+M1) with sig = -1 for
  // odd popcount(spin^taste), replicated per momentum through the flat
  // g-major sigma table -- the odd-odd BothHalf slot exercises exactly
  // that table under momentum.)
  {
    auto gammas = makeOps(momTestGammas(), U);
    int nG = (int)gammas.size();
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor>
        full(nG, nMom, Nt, nevec, nevec),
        cb(nG, nMom, Nt, 2 * nevec, nevec),
        bb(nG, nMom, Nt, 2 * nevec, 2 * nevec);
    full.setZero();
    cb.setZero();
    bb.setZero();
    {
      A2AWorkerSpinTasteStencil<FImpl> w(grid, ph, gammas, &U, orthogDir);
      w.StagMesonField(full, vecsL.data(), vecsR.data(), nevec, nevec);
      w.StagMesonField(cb, vecsL.data(), vecsR.data(), nevec, nevec,
                       ContractType::LeftHalf);
      w.StagMesonField(bb, vecsL.data(), vecsR.data(), nevec, nevec,
                       ContractType::BothHalf);
    }
    double err = 0.0;
    for (int g = 0; g < nG; g++) {
      RealD sig = (StagGamma::popcountShift(gammas[g]._spin,
                                            gammas[g]._taste) & 1)
                      ? -1.0
                      : 1.0;
      for (int p = 0; p < nMom; p++)
        for (int t = 0; t < Nt; t++)
          for (int i = 0; i < nevec; i++)
            for (int j = 0; j < nevec; j++) {
              ComplexD f = full(g, p, t, i, j);
              // LeftHalf (2l, r) = M0+M1 = Full.
              ComplexD y = cb(g, p, t, 2 * i, j);
              double den = std::max(std::abs(y), std::abs(f));
              double d = std::abs(y - f);
              err = std::max(err, den > 0 ? d / den : d);
              // BothHalf (2l, 2r) = M0+M1 = Full; (2l+1, 2r+1) =
              // sig*(M0+M1) = sig*Full (flat sigma table under momentum).
              y = bb(g, p, t, 2 * i, 2 * j);
              den = std::max(std::abs(y), std::abs(f));
              d = std::abs(y - f);
              err = std::max(err, den > 0 ? d / den : d);
              ComplexD ys = sig * f;
              y = bb(g, p, t, 2 * i + 1, 2 * j + 1);
              den = std::max(std::abs(y), std::abs(ys));
              d = std::abs(y - ys);
              err = std::max(err, den > 0 ? d / den : d);
            }
    }
    check("momentum CB interleaved slots == Full (+sigma table)", err, 1e-10);
  }

  bool ok = (nFail == 0);
  std::cout << GridLogMessage
            << "=== Test_a2a_stencil_ja_mom complete ===" << std::endl;
  if (ok)
    std::cout << GridLogMessage << "PASS" << std::endl;
  else
    std::cerr << "FAIL" << std::endl;
  Grid_finalize();
  return ok ? 0 : 1;
}
