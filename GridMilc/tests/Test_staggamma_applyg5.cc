/*
 * GridMilc/tests/Test_staggamma_applyg5.cc
 *
 * 256-pair regression test for applyG5 (eps∘Γ) and the composition-fixed
 * operator*.
 *
 * For EVERY (spin, taste) pair, on each configured gauge (cold/unit +
 * random; optional --gauge <file> with APBC in T), with a fixed-seed
 * random source field:
 *
 *   (a) applyG5Left:    fold.applyGamma(src)  ==  eps * gPlain.applyGamma(src)
 *   (b) operator* Γ*ε:  prod.applyGamma(src)  ==  eps * gPlain.applyGamma(src)
 *                       (prod = gPlain * epsObj; identical folded state)
 *   (c) operator* ε*Γ:  prod2.applyGamma(src) ==  gPlain.applyGamma(eps * src)
 *                       (prod2 = epsObj * gPlain; eps applied FIRST)
 *   (d) operator* disjoint composition (unit gauge only):
 *                       prodD.applyGamma(src) ==  g2.applyGamma(
 *                           gPlain.applyGamma(src))
 *                       for every ordered (gPlain, g2) with disjoint shift
 *                       directions (plan-review extension: re-locks the
 *                       design-time disjoint/local sweep at runtime)
 *
 * The two sides of every check derive eps from DIFFERENT constructions
 * (folded _negated sign vs explicit where() multiply), so a sign bug
 * cannot self-confirm. On sign mismatch norm2(out1-out2) ≈ 4*norm2(out)
 * (relErr ≈ 2) -- binary all-or-nothing per pair, so the 1e-13 threshold
 * cleanly separates.
 *
 * Usage:
 *   Test_staggamma_applyg5 [--grid W.X.Y.Z] [--mpi a.b.c.d] [--gauge <file>]
 *
 * Standalone: no external files required (cold + random gauges). --gauge
 * reads a thin-link config (ILDG if LIME, else NERSC) and applies APBC in
 * T, matching Test_staggamma's thin-link treatment (manual lat.sample
 * runs).
 */
#include <Grid/Grid.h>
#include <GridMilc/spin/StagGamma.h>
#include <Grid/qcd/utils/SpaceTimeGrid.h>
#ifdef HAVE_LIME
#include <Grid/parallelIO/IldgIO.h>
#else
#include <Grid/parallelIO/NerscIO.h>
#endif

#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace Grid;

typedef LatticeStaggeredFermion FermionField;

static const StagGamma::StagAlgebra allAlgebra[16] = {
    StagGamma::StagAlgebra::G1,  StagGamma::StagAlgebra::GZ,
    StagGamma::StagAlgebra::GY,  StagGamma::StagAlgebra::GYZ,
    StagGamma::StagAlgebra::GX,  StagGamma::StagAlgebra::GZX,
    StagGamma::StagAlgebra::GXY, StagGamma::StagAlgebra::G5T,
    StagGamma::StagAlgebra::GT,  StagGamma::StagAlgebra::GZT,
    StagGamma::StagAlgebra::GYT, StagGamma::StagAlgebra::G5X,
    StagGamma::StagAlgebra::GXT, StagGamma::StagAlgebra::G5Y,
    StagGamma::StagAlgebra::G5Z, StagGamma::StagAlgebra::G5};

// parity(x) = (x+y+z+t) % 2  (Test_staggamma.cc:168-178)
static void makeParityField(Lattice<iScalar<vInteger>> &parity) {
  GridBase *grid = parity.Grid();
  Lattice<iScalar<vInteger>> coor(grid);
  parity = Zero();
  for (int mu = 0; mu < Nd; mu++) {
    LatticeCoordinate(coor, mu);
    parity = parity + coor;
  }
}

// f *= eps(x) = (-1)^parity  (Test_staggamma.cc:180-186)
template <class obj>
static void applyEpsilon(Lattice<obj> &f,
                         const Lattice<iScalar<vInteger>> &parity) {
  Lattice<obj> neg(f.Grid());
  neg = -f;
  f = where(mod(parity, 2) == (Integer)1, neg, f);
}

// APBC in the time direction (Test_staggamma.cc:121-136)
static void applyAPBC(LatticeGaugeField &U) {
  Coordinate latt = U.Grid()->GlobalDimensions();
  std::vector<int> boundary(Nd, 1);
  boundary[Nd - 1] = -1;
  for (int mu = 0; mu < Nd; mu++) {
    if (boundary[mu] == 1)
      continue;
    Lattice<iScalar<vInteger>> coord(U.Grid());
    LatticeCoordinate(coord, mu);
    LatticeColourMatrix link(U.Grid());
    link = PeekIndex<LorentzIndex>(U, mu);
    int dimSize = latt[mu] - 1;
    link = where(coord == dimSize, static_cast<double>(boundary[mu]) * link,
                 link);
    PokeIndex<LorentzIndex>(U, link, mu);
  }
}

// Relative error ||a-b|| / ||b|| (Test_a2a_stencil.cc fail-fast pattern)
static RealD relError(const FermionField &a, const FermionField &b) {
  FermionField diff(a.Grid());
  diff = a - b;
  auto dn = norm2(diff);
  auto rn = norm2(b);
  return (rn > 0.0) ? sqrt(dn / rn) : sqrt(dn);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::string gaugeFile = GridCmdOptionPayload(argv, argv + argc, "--gauge");

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);

  std::cout << GridLogMessage << "Test_staggamma_applyg5: lattice " << latt
            << std::endl;

  Lattice<iScalar<vInteger>> parity(&Grid);
  makeParityField(parity);

  // Fixed-seed random source (differential test: both sides use the same
  // field; randomness only avoids measure-zero inputs).
  FermionField src(&Grid);
  {
    GridParallelRNG rng(&Grid);
    rng.SeedFixedIntegers({11, 22, 33, 44});
    src = Zero();
    random(rng, src);
  }

  // Gauge configurations to sweep: cold/unit + random always; --gauge file
  // (with APBC in T) additionally when provided.
  std::vector<std::pair<std::string, LatticeGaugeField>> configs;
  {
    LatticeGaugeField Ucold(&Grid);
    GridParallelRNG rng(&Grid);
    rng.SeedFixedIntegers({1, 2, 3, 4});
    SU<Nc>::ColdConfiguration(rng, Ucold);
    configs.emplace_back("unit-gauge", std::move(Ucold));

    LatticeGaugeField Urand(&Grid);
    Urand = Zero();
    GridParallelRNG rng2(&Grid);
    rng2.SeedFixedIntegers({10, 20, 30, 40});
    random(rng2, Urand);
    configs.emplace_back("random-gauge", std::move(Urand));

    if (!gaugeFile.empty()) {
      LatticeGaugeField Ufile(&Grid);
      FieldMetaData header;
#ifdef HAVE_LIME
      {
        IldgReader IR;
        IR.open(gaugeFile);
        IR.readConfiguration(Ufile, header);
        IR.close();
      }
#else
      NerscIO::readConfiguration(Ufile, header, gaugeFile);
#endif
      applyAPBC(Ufile);
      configs.emplace_back("file:" + gaugeFile, std::move(Ufile));
    }
  }

  FermionField out1(&Grid), out2(&Grid), tmp(&Grid);
  const RealD tol = 1e-13;
  int nFail = 0;
  int nCheck = 0;
  RealD maxErr = 0.0;

  auto check = [&](const char *family, const std::string &name,
                   const std::string &gname) {
    nCheck++;
    RealD relErr = relError(out1, out2);
    maxErr = std::max(maxErr, relErr);
    if (relErr > tol) {
      nFail++;
      std::cerr << "FAIL " << family << " " << name << " [" << gname
                << "] relErr=" << relErr << std::endl;
    }
  };

  for (auto &cfg : configs) {
    LatticeGaugeField &U = cfg.second;

    // eps object: local (popcount 0) so U is never dereferenced by it, but
    // bound so operator* propagates the shared pointer to the products.
    StagGamma epsObj(StagGamma::StagAlgebra::G5, StagGamma::StagAlgebra::G5);
    epsObj.setGaugeField(U);

    for (int s = 0; s < 16; s++) {
      for (int t = 0; t < 16; t++) {
        StagGamma gPlain(allAlgebra[s], allAlgebra[t]);
        gPlain.setGaugeField(U);
        std::string name = StagGamma::GetName(allAlgebra[s], allAlgebra[t]);

        // Reference for (a)/(b): eps ∘ Γ (explicit multiply AFTER Γ)
        gPlain.applyGamma(out2, src);
        applyEpsilon(out2, parity);

        // (a) applyG5Left fold
        StagGamma gFold = gPlain;
        gFold.applyG5Left();
        gFold.applyGamma(out1, src);
        check("applyG5Left eps∘Γ", name, cfg.first);

        // (b) operator* Γ*ε (apply Γ first, then ε)
        StagGamma prod = gPlain * epsObj;
        prod.applyGamma(out1, src);
        check("operator* Γ*ε", name, cfg.first);

        // (c) operator* ε*Γ (apply ε first, then Γ); reference is Γ∘ε
        tmp = src;
        applyEpsilon(tmp, parity);
        gPlain.applyGamma(out2, tmp);
        StagGamma prod2 = epsObj * gPlain;
        prod2.applyGamma(out1, src);
        check("operator* ε*Γ", name, cfg.first);

        // (d) operator* disjoint composition (UNIT GAUGE ONLY): g1*g2
        // means apply g1 FIRST, then g2. On the free field a disjoint
        // product is exactly the sequential composition; on dynamical
        // gauges the covariant transport ordering differs (design D2), so
        // this family runs only on the cold/unit configuration. Re-locks
        // the design-time disjoint/local sweep (local∘X, X∘local,
        // one-link×one-link, multi-link×multi-link) on the real lattice
        // algebra: full ordered (g1,g2) matrix, disjoint-shift filter.
        if (cfg.first == "unit-gauge") {
          int shift1 = gPlain._spin ^ gPlain._taste;
          gPlain.applyGamma(tmp, src); // g1 applied FIRST (hoisted)
          for (int s2 = 0; s2 < 16; s2++) {
            for (int t2 = 0; t2 < 16; t2++) {
              int shift2 = allAlgebra[s2] ^ allAlgebra[t2];
              if (shift1 & shift2)
                continue; // overlapping: inexpressible (guarded domain)
              StagGamma g2(allAlgebra[s2], allAlgebra[t2]);
              g2.setGaugeField(U);
              g2.applyGamma(out2, tmp); // then g2
              StagGamma prodD = gPlain * g2;
              prodD.applyGamma(out1, src);
              check("operator* disjoint g1∘g2",
                    name + ";" + StagGamma::GetName(allAlgebra[s2],
                                                    allAlgebra[t2]),
                    cfg.first);
            }
          }
        }
      }
    }
    std::cout << GridLogMessage << "gauge '" << cfg.first
              << "': swept 256 pairs" << std::endl;
  }

  std::cout << GridLogMessage << "Test_staggamma_applyg5: " << nCheck
            << " checks, maxRelErr=" << maxErr << " (tol " << tol << ")"
            << std::endl;
  if (nFail > 0) {
    std::cerr << "Test_staggamma_applyg5: " << nFail << " failures!"
              << std::endl;
    GridAbort();
  }
  std::cout << GridLogMessage << "Test_staggamma_applyg5: ALL PASSED"
            << std::endl;

  Grid_finalize();
  return 0;
}
