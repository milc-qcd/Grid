/*
 * GridMilc/tests/Test_staggamma.cc — spin-taste meson correlator generator
 *
 * Generates connected two-point correlators C(t) for all 16 x 16 = 256
 * spin-taste operator pairs, for comparison with MILC ks_spectrum output.
 *
 * The Dirac operator is ImprovedStaggeredFermion (HISQ: fat7 + Naik) using
 * pre-computed fat and long links that already carry the KS phases and APBC.
 * The thin links (lat.sample, NO KS phases) are used only for StagGamma's
 * covariant shifts; APBC is applied to them so the temporal boundary
 * condition matches the fat/long links baked into the Dirac operator.
 *
 * For each (spin, taste) pair and each source colour k:
 *   phi_0k = M^{-1}(e_k)                [base propagator]
 *   src_gk = eps? * applyGamma(g, e_k)  [modified source]
 *   phi_gk = M^{-1}(src_gk)             [operator propagator]
 *   snk_gk = eps? * applyGamma(g, phi_0k)
 *   C_k(t) = sum_{x@t} conj(snk_gk) . phi_gk
 *   C(t)   = sum_k C_k(t)
 *
 * Per-colour solves are required because gauge links mix colours; solving
 * with an all-colour source would introduce cross-colour terms absent in
 * MILC.
 *
 * Usage:
 *   Test_staggamma --correlator <out.xml> --gauge <thin> [options]
 *   Test_staggamma --correlator <out.xml> --unit-gauge      [options]
 *
 * Options:
 *   --correlator <xml>    output C(t) per (spin,taste) pair  [required]
 *   --gauge <file>        thin gauge config (NERSC or ILDG).  Fat/long links
 *                         are auto-derived from this path:
 *                           <dir>/lat.sample.l4444.ildg.20  (thin)
 *                           <dir>/fatKSAPBC.l4444.ildg.20   (fat, derived)
 *                           <dir>/lngKSAPBC.l4444.ildg.20   (long, derived)
 *   --fat-links <file>    fat7 links (override auto-derivation)
 *   --long-links <file>   Naik/long links (override auto-derivation)
 *   --unit-gauge          use the identity (cold) gauge field for all links
 *   --epsilon             apply (-1)^(x+y+z+t) at source+sink (MILC's
 *                         antiquark sign flip, which Grid's StagGamma omits)
 *   --only <NAME-NAME>    compute just one spin-taste pair (e.g. G5-G5)
 *   --mass <val>          quark mass in MILC convention (default 0.1;
 *                         the factor-of-2 KS mass convention is applied
 *                         internally via 2.0*mass)
 *
 * The output XML uses the schema shared with build_spin_taste_reference.py:
 *   <staggamma_correlator>
 *     <header lattice=".." mass=".." epsilon=".." generator="Grid"/>
 *     <pair spin="G1" taste="G1">
 *       <ct t="0" re=".." im=".."/>
 *       ...
 *     </pair>
 *     ...
 *   </staggamma_correlator>
 */
#include <Grid/Grid.h>
#include <GridMilc/spin/StagGamma.h>
#include <Grid/qcd/action/fermion/ImprovedStaggeredFermion.h>
#include <Grid/algorithms/iterative/ConjugateGradient.h>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace Grid;

// string -> StagAlgebra (for --only NAME-NAME)
static StagGamma::StagAlgebra algebraFromString(const std::string &s) {
  for (int i = 0; i < StagGamma::nGamma; i++)
    if (s == StagGamma::name[i]) return static_cast<StagGamma::StagAlgebra>(i);
  std::cerr << "ERROR: unknown StagAlgebra '" << s << "'" << std::endl;
  exit(1);
}

// Gauge config reader (NERSC always available; ILDG if built with LIME)
static void readConfig(LatticeGaugeField &U, const std::string &file) {
  FieldMetaData header;
#ifdef HAVE_LIME
  {
    IldgReader IR;
    IR.open(file);
    IR.readConfiguration(U, header);
    IR.close();
  }
#else
  NerscIO::readConfiguration(U, header, file);
#endif
}

// Derive a fat/long link path from the thin-gauge path by replacing the
// filename prefix before the lattice-size tag ".l<digits>" with the given
// prefix (e.g. "fatKSAPBC" or "lngKSAPBC").
//   configs/lat.sample.l4444.ildg.20  ->  configs/fatKSAPBC.l4444.ildg.20
// Returns "" if the lattice-size tag cannot be found.
static std::string deriveLinkPath(const std::string &gaugePath,
                                  const std::string &prefix) {
  size_t slash = gaugePath.find_last_of('/');
  std::string dir =
      (slash == std::string::npos) ? "" : gaugePath.substr(0, slash + 1);
  std::string fname =
      (slash == std::string::npos) ? gaugePath : gaugePath.substr(slash + 1);
  size_t pos = std::string::npos;
  for (size_t i = 0; i + 2 <= fname.size(); i++) {
    if (fname[i] == '.' && fname[i + 1] == 'l' &&
        std::isdigit(static_cast<unsigned char>(fname[i + 2]))) {
      pos = i;
      break;
    }
  }
  if (pos == std::string::npos) return "";
  return dir + prefix + fname.substr(pos);
}

// Apply antiperiodic boundary conditions to a gauge field by flipping the
// sign of links at the last slice in directions where boundary[mu] == -1.
// Matches HadronsMILC's APBCGauge module.  The fat/long link files
// (fatKSAPBC/lngKSAPBC) already have APBC baked in; this applies the same
// transformation to the thin links used by StagGamma's covariant shifts.
static void applyAPBC(LatticeGaugeField &U, const std::vector<int> &boundary) {
  Coordinate latt = U.Grid()->GlobalDimensions();
  for (int mu = 0; mu < Nd; mu++) {
    if (boundary[mu] == 1) continue;
    Lattice<iScalar<vInteger>> coord(U.Grid());
    LatticeCoordinate(coord, mu);
    LatticeColourMatrix link(U.Grid());
    link = PeekIndex<LorentzIndex>(U, mu);
    int dimSize = latt[mu] - 1;
    link = where(coord == dimSize,
                 static_cast<double>(boundary[mu]) * link, link);
    PokeIndex<LorentzIndex>(U, link, mu);
  }
}

// Point source at the global origin. Uses LATTICE-typed where branches
// (int-literal branches do not compile: the trinary evaluator needs scalar_object).
static void makePointSource(LatticeStaggeredFermion &src, int color = -1) {
  GridBase *grid = src.Grid();
  src = Zero();
  Lattice<iScalar<vInteger>> coor(grid), dist(grid), zeroI(grid), oneI(grid);
  zeroI = Zero();
  oneI = 1;
  dist = Zero();
  for (int mu = 0; mu < Nd; mu++) {
    LatticeCoordinate(coor, mu);
    dist = dist + where(coor == 0, zeroI, oneI); // dist==0 iff every coord is 0
  }
  if (color < 0) {
    LatticeStaggeredFermion unit(grid);
    unit = 1.0;
    src = where(dist == 0, unit, src);
  } else {
    // Single-color source: e_color at origin
    vColourVector e_c;
    e_c = Zero();
    e_c()()(color) = 1.0;
    LatticeStaggeredFermion unit(grid);
    unit = e_c;
    src = where(dist == 0, unit, src);
  }
}

// Build the checkerboard parity mask parity(x) = (x+y+z+t) % 2.
// MILC's spin_taste_op folds the "antiquark sign flip" (-1)^(x+y+z+t) into
// every operator; Grid's StagGamma omits it.  When --epsilon is set we flip
// the sign on odd-parity sites at source and sink to match MILC's convention.
static void makeParityField(Lattice<iScalar<vInteger>> &parity) {
  GridBase *grid = parity.Grid();
  Lattice<iScalar<vInteger>> coor(grid);
  parity = Zero();
  for (int mu = 0; mu < Nd; mu++) {
    LatticeCoordinate(coor, mu);
    parity = parity + coor;
  }
}

// Multiply a fermion field by eps(x) = (-1)^parity in place, via where().
template <class obj>
static void applyEpsilon(Lattice<obj> &f,
                         const Lattice<iScalar<vInteger>> &parity) {
  Lattice<obj> neg(f.Grid());
  neg = -f;
  f = where(mod(parity, 2) == (Integer)1, neg, f);
}

static const StagGamma::StagAlgebra allAlgebra[16] = {
    StagGamma::StagAlgebra::G1,  StagGamma::StagAlgebra::GZ,
    StagGamma::StagAlgebra::GY,  StagGamma::StagAlgebra::GYZ,
    StagGamma::StagAlgebra::GX,  StagGamma::StagAlgebra::GZX,
    StagGamma::StagAlgebra::GXY, StagGamma::StagAlgebra::G5T,
    StagGamma::StagAlgebra::GT,  StagGamma::StagAlgebra::GZT,
    StagGamma::StagAlgebra::GYT, StagGamma::StagAlgebra::G5X,
    StagGamma::StagAlgebra::GXT, StagGamma::StagAlgebra::G5Y,
    StagGamma::StagAlgebra::G5Z, StagGamma::StagAlgebra::G5};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // --- command-line options ---
  std::string corrFile =
      GridCmdOptionPayload(argv, argv + argc, "--correlator");
  std::string gaugeFile = GridCmdOptionPayload(argv, argv + argc, "--gauge");
  std::string fatFile =
      GridCmdOptionPayload(argv, argv + argc, "--fat-links");
  std::string longFile =
      GridCmdOptionPayload(argv, argv + argc, "--long-links");
  std::string onlyPair = GridCmdOptionPayload(argv, argv + argc, "--only");
  bool unitGauge = GridCmdOptionExists(argv, argv + argc, "--unit-gauge");
  bool useEpsilon = GridCmdOptionExists(argv, argv + argc, "--epsilon");
  std::string massStr = GridCmdOptionPayload(argv, argv + argc, "--mass");
  RealD mass = massStr.empty() ? 0.1 : std::stod(massStr);

  if (corrFile.empty()) {
    std::cerr << "ERROR: --correlator <xml> is required\n"
              << "Usage: Test_staggamma --correlator <out.xml> "
                 "--gauge <config> [--unit-gauge] [...]\n";
    exit(1);
  }
  if (gaugeFile.empty() && !unitGauge) {
    std::cerr << "ERROR: --correlator requires --gauge <file> or --unit-gauge\n";
    exit(1);
  }

  // --- grid setup ---
  Coordinate latt_size = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout = GridDefaultMpi();
  GridCartesian Grid(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid(&Grid);

  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers(seeds);

  typedef LatticeStaggeredFermion FermionField;

  // --- gauge configurations ---
  // Thin links: lat.sample (no KS phases).  APBC is applied below so the
  // temporal boundary condition matches the fat/long links.
  // Fat/long links: fatKSAPBC / lngKSAPBC (KS phases + APBC pre-baked).
  LatticeGaugeField Uthin(&Grid), Ufat(&Grid), Ulong(&Grid);
  if (unitGauge) {
    SU<Nc>::ColdConfiguration(pRNG, Uthin);
    // For unit gauge fat7(identity) = identity, naik(identity) = identity
    Ufat = Uthin;
    Ulong = Uthin;
  } else {
    readConfig(Uthin, gaugeFile);

    // Derive fat/long paths if not explicitly provided
    if (fatFile.empty()) {
      fatFile = deriveLinkPath(gaugeFile, "fatKSAPBC");
      if (fatFile.empty()) {
        std::cerr << "ERROR: cannot derive fat-links path from '" << gaugeFile
                  << "'; use --fat-links <file>\n";
        exit(1);
      }
    }
    if (longFile.empty()) {
      longFile = deriveLinkPath(gaugeFile, "lngKSAPBC");
      if (longFile.empty()) {
        std::cerr << "ERROR: cannot derive long-links path from '" << gaugeFile
                  << "'; use --long-links <file>\n";
        exit(1);
      }
    }
    std::cout << GridLogMessage << "thin links:  " << gaugeFile << std::endl;
    std::cout << GridLogMessage << "fat links:   " << fatFile << std::endl;
    std::cout << GridLogMessage << "long links:  " << longFile << std::endl;
    readConfig(Ufat, fatFile);
    readConfig(Ulong, longFile);
  }

  // Apply APBC to the thin links (time direction) so StagGamma's covariant
  // shifts use the same temporal boundary condition as the fat/long links.
  // (The fat/long files already carry APBC; do NOT re-apply it to them.)
  {
    std::vector<int> boundary(Nd, 1);
    boundary[Nd - 1] = -1; // antiperiodic in time
    applyAPBC(Uthin, boundary);
    std::cout << GridLogMessage
              << "applied APBC (time) to thin links for StagGamma" << std::endl;
  }

  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Uthin);
  std::cout << GridLogMessage << "thin-link plaquette = " << plaq << std::endl;

  // --- Dirac operator + CG solver ---
  // ImprovedStaggeredFermion (HISQ) with pre-computed fat/long links.
  // ImportGaugeSimple assumes phases + fattening are pre-applied (matching
  // the fatKSAPBC/lngKSAPBC files).  Mass uses the MILC factor-of-2 KS
  // convention: MILC mass m -> Grid constructor 2*m.
  ImprovedStaggeredFermionD Ds(Grid, RBGrid, 2.0 * mass);
  Ds.ImportGaugeSimple(Ulong, Ufat);

  StagGamma gamma;
  gamma.setGaugeField(Uthin); // thin links with APBC for covariant shifts

  MdagMLinearOperator<ImprovedStaggeredFermionD, FermionField> HermOp(Ds);
  ConjugateGradient<FermionField> CG(1.0e-12, 100000);

  Lattice<iScalar<vInteger>> parity(&Grid);
  if (useEpsilon) makeParityField(parity);

  // --- base propagators: phi_0k = M^{-1}(e_k) for each source colour k ---
  std::vector<FermionField> phi0;
  phi0.reserve(Nc);
  for (int k = 0; k < Nc; k++) phi0.emplace_back(&Grid);
  FermionField psrc(&Grid), msrc(&Grid);
  for (int k = 0; k < Nc; k++) {
    phi0[k] = FermionField(&Grid);
    makePointSource(psrc, k);
    Ds.Mdag(psrc, msrc);
    CG(HermOp, msrc, phi0[k]);
  }
  std::cout << GridLogMessage << "correlator: solved " << Nc
            << " base propagators (mass=" << mass << ")" << std::endl;

  // --- decide which pairs to compute ---
  std::vector<std::pair<int, int>> pairs;
  if (!onlyPair.empty()) {
    auto dash = onlyPair.find('-');
    if (dash == std::string::npos) {
      std::cerr << "ERROR: --only expects NAME-NAME e.g. G5-G5\n";
      exit(1);
    }
    pairs.push_back({(int)algebraFromString(onlyPair.substr(0, dash)),
                     (int)algebraFromString(onlyPair.substr(dash + 1))});
  } else {
    for (int s = 0; s < 16; s++)
      for (int t = 0; t < 16; t++)
        pairs.push_back({s, t});
  }

  // --- correlator output ---
  std::ofstream of(corrFile);
  of << "<?xml version=\"1.0\"?>\n<staggamma_correlator>\n";
  of << "  <header lattice=\"" << latt_size[0];
  for (int mu = 1; mu < Nd; mu++) of << " " << latt_size[mu];
  of << "\" mass=\"" << mass << "\" epsilon=\"" << (useEpsilon ? 1 : 0)
     << "\" generator=\"Grid\"/>\n";

  FermionField srcg(&Grid), phig(&Grid), snkg(&Grid);
  for (auto &pr : pairs) {
    int s = pr.first, t = pr.second;
    gamma.setSpinTaste(allAlgebra[s], allAlgebra[t]);

    std::vector<ComplexD> Ct(latt_size[Nd - 1], ComplexD(0.0, 0.0));
    for (int k = 0; k < Nc; k++) {
      makePointSource(psrc, k);

      // modified source
      gamma.applyGamma(srcg, psrc);
      if (useEpsilon) applyEpsilon(srcg, parity);

      // propagator
      Ds.Mdag(srcg, msrc);
      CG(HermOp, msrc, phig);

      // sink op on base propagator for this colour
      gamma.applyGamma(snkg, phi0[k]);
      if (useEpsilon) applyEpsilon(snkg, parity);

      // C_k(t) = sum_{x@t} conj(snkg) . phig
      std::vector<ComplexD> Ck;
      sliceInnerProductVector(Ck, snkg, phig, Nd - 1);
      for (int tt = 0; tt < (int)Ct.size(); tt++) Ct[tt] += Ck[tt];
    }

    of << "  <pair spin=\"" << StagGamma::name[s] << "\" taste=\""
       << StagGamma::name[t] << "\">\n";
    for (int tt = 0; tt < (int)Ct.size(); tt++)
      of << "    <ct t=\"" << tt << "\" re=\"" << std::setprecision(15)
         << Ct[tt].real() << "\" im=\"" << Ct[tt].imag() << "\"/>\n";
    of << "  </pair>\n";
    std::cout << GridLogMessage << "corr "
              << StagGamma::GetName(allAlgebra[s], allAlgebra[t])
              << " C(0)=" << Ct[0] << std::endl;
  }
  of << "</staggamma_correlator>\n";

  std::cout << GridLogMessage << "wrote correlator XML to " << corrFile
            << std::endl;
  Grid_finalize();
  return 0;
}
