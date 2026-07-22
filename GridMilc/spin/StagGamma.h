/*
 * GridMilc/spin/StagGamma.h — part of GridMilc
 * (https://github.com/paboyle/Grid)
 *
 * Staggered spin-taste gamma algebra (Follana 2007). Header-only; lifted
 * from HadronsMILC. Self-contained via Grid's QCD core umbrella.
 *
 * GridMilc is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 2 (or, at your option,
 * any later version). See COPYING/LICENSE in the top-level distribution.
 */
#pragma once

#include <algorithm> // std::next_permutation (symmetric path-ordering sum)
#include <array>
#include <iostream>

#include <Grid/GridQCDcore.h>

NAMESPACE_BEGIN(Grid)

// Spin taste parameters for modules that require StagGamma objects.
struct SpinTasteParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(SpinTasteParams, std::string, gammas,
                                  std::string, gauge, bool, applyG5);
  SpinTasteParams(void) : gammas(""), gauge(""), applyG5(false) {}
};

class StagGamma {
public:
  // XYZT convention
  /*    GRID_SERIALIZABLE_ENUM(StagAlgebra, undef,
                             G1     , 0,
                             GT     , 1,
                             GZ     , 2,
                             GZT    , 3,
                             GY     , 4,
                             GYT    , 5,
                             GYZ    , 6,
                             G5X    , 7,
                             GX     , 8,
                             GXT    , 9,
                             GZX    , 10,
                             G5Y    , 11,
                             GXY    , 12,
                             G5Z    , 13,
                             G5T    , 14,
                             G5     , 15);*/

  // TXYZ convention
  // clang-format off
  GRID_SERIALIZABLE_ENUM(StagAlgebra, undef, 
                         G1,  0b0000,
                         GZ,  0b0001,
                         GY,  0b0010,
                         GYZ, 0b0011,
                         GX,  0b0100,
                         GZX, 0b0101,
                         GXY, 0b0110,
                         G5T, 0b0111,
                         GT,  0b1000,
                         GZT, 0b1001,
                         GYT, 0b1010,
                         G5X, 0b1011,
                         GXT, 0b1100,
                         G5Y, 0b1101,
                         G5Z, 0b1110,
                         G5,  0b1111);
  // clang-format on

  typedef std::pair<StagAlgebra, StagAlgebra> SpinTastePair;

public:
  StagGamma() : _spin(0), _taste(0) {}

  StagGamma(StagAlgebra spin, StagAlgebra taste) {
    _spin = spin;
    _taste = taste;
    calculatePhase();
  }

  StagGamma(SpinTastePair initg) { StagGamma(initg.first, initg.second); }

  void setGaugeField(LatticeGaugeField &U_) { U = &U_; }

  inline void setSpin(StagAlgebra g) {
    _spin = g;
    calculatePhase();
  }

  inline void setTaste(StagAlgebra g) {
    _taste = g;
    calculatePhase();
  }

  inline void setSpinTaste(StagAlgebra spin, StagAlgebra taste) {
    _spin = spin;
    _taste = taste;
    calculatePhase();
  }

  inline void setSpinTaste(SpinTastePair g) { setSpinTaste(g.first, g.second); }

  static std::vector<StagGamma::SpinTastePair>
  ParseSpinTasteString(std::string str, bool applyG5 = false) {
    auto gammas = strToVec<StagGamma::SpinTastePair>(str);

    if (applyG5) {
      StagGamma st;
      StagGamma g5(StagGamma::StagAlgebra::G5, StagGamma::StagAlgebra::G5);

      for (auto &g : gammas) {
        st.setSpinTaste(g);
        st = st * g5;
        g.first = st._spin;
        g.second = st._taste;
      }
    }

    return gammas;
  }
  static std::string GetName(StagAlgebra spin, StagAlgebra taste) {

    std::string name = StagGamma::name[spin];
    name = (name + "_") + StagGamma::name[taste];

    return name;
  }

  static std::string GetName(SpinTastePair g) {
    return StagGamma::GetName(g.first, g.second);
  }

  std::string getName() const { return StagGamma::GetName(_spin, _taste); }

  template <typename obj>
  void applyGamma(Lattice<obj> &lhs, const Lattice<obj> &rhs) const;

  template <typename obj>
  void applyCoeffsAndPhase(Lattice<obj> &lhs, const Lattice<obj> &rhs) const;

  template <typename obj>
  void oneLink(Lattice<obj> &lhs, const Lattice<obj> &rhs, int shift_dir) const;

  template <typename obj>
  inline void operator()(Lattice<obj> &lhs, const Lattice<obj> &rhs) const {
    applyGamma(lhs, rhs);
  }

private:
  // Calculate the < and > operations as defined in Follana (2007) eqns A5 and
  // A7.
  static inline StagAlgebra LessThan(StagAlgebra g);
  static inline StagAlgebra GreaterThan(StagAlgebra g);

  // Assign negative orientations to StagAlgebra gammas according to txyz (or
  // xyzt?) oriented euclidean space.
  inline int getOrientation(StagAlgebra g);

  // Implements eqn. E3 of Follana (2007)
  inline void calculatePhase();

  // Implements (-1)^(x[mu] * ( _taste^< + _spin^> ) ( see eqn. E3 of Follana
  // (2007) )
  inline void calculateOscillation();

  // Implements (-1)^(_spin * (_spin + _taste)^<) ( see eqn. E3 of Follana
  // (2007) )
  inline void calculateNegation();

  inline void toggleNegation() { _negated = !_negated; }

public:
  static constexpr unsigned int nGamma = 16;
  // TXYZ convention
  static inline const std::array<const char *, nGamma> name = {
      {"G1", "GZ", "GY", "GYZ", "GX", "GZX", "GXY", "G5T", "GT", "GZT", "GYT",
       "G5X", "GXT", "G5Y", "G5Z", "G5"}};
  static inline const std::array<const StagAlgebra, 4> gmu = {{
      StagGamma::StagAlgebra::GX,
      StagGamma::StagAlgebra::GY,
      StagGamma::StagAlgebra::GZ,
      StagGamma::StagAlgebra::GT,
  }};
  // Number of active covariant-shift directions for a spin-taste pair: the
  // popcount of (spin ^ taste), scanned in gmu[] (X,Y,Z,T) order. Mirrors the
  // local computation in calculatePhase. Used by A2A task popcount
  // validation/gating (A2ATaskOnelink hardening, A2ATaskSpinTaste uniformity).
  static int popcountShift(StagAlgebra spin, StagAlgebra taste) {
    int shift = spin ^ taste;
    int n = 0;
    for (const auto &dir : gmu) {
      if (static_cast<int>(dir) & shift) {
        n++;
      }
    }
    return n;
  }

  friend inline StagGamma operator*(const StagGamma &g1, const StagGamma &g2);

public:
  StagAlgebra _spin, _taste;
  LatticeGaugeField *U = nullptr;

private:
  StagAlgebra _oscillateDirs = 0;
  bool _negated = false;
  RealD _scaling;
};

inline StagGamma::StagAlgebra StagGamma::LessThan(StagAlgebra g) {
  uint8_t ret = 0;
  uint8_t mask = g;

  for (int i = 0; i < Nd - 1; i++) {
    mask = mask >> 1;
    ret = ret ^ mask; // each bit will toggle for each 1 that passes over it
  }
  return ret;
}

inline StagGamma::StagAlgebra StagGamma::GreaterThan(StagAlgebra g) {
  uint8_t ret = 0;
  uint8_t mask = g;

  for (int i = 0; i < Nd - 1; i++) {
    mask = mask << 1;
    ret = ret ^ mask; // each bit will toggle for each 1 that passes over it
  }
  return (0x0F & ret);
}

template <class obj>
void StagGamma::applyGamma(Lattice<obj> &lhs, const Lattice<obj> &rhs) const {
  // shift = _spin ^ _taste encodes the directions needing a covariant hop.
  // The spin-taste operator is the taste gamma xi_B (Follana 2007), realized
  // on the lattice as a product of the Golterman-Smit hop operators
  // Xi_mu = zeta_mu S_mu: a symmetric covariant shift S_mu dressed by the
  // staggered phase zeta_mu. The Xi_mu anticommute by construction of zeta
  // ({Xi_mu,Xi_nu} ~ 0), so they generate a Clifford algebra and the
  // antisymmetrized Xi-product IS the ordered product -- the lattice taste
  // matrix, sign-free. Gauge transport is order-dependent
  // (U_Y(x)U_X(x+Y) != U_X(x)U_Y(x+X)), so all n! orderings contribute; a
  // single ordered chain is wrong for popcount >= 2.
  //
  // Grid does not use Xi_mu directly: it shifts with plain S_mu and applies the
  // zeta-product ONCE at the end (applyCoeffsAndPhase). Because zeta_mu is
  // position-dependent and S_mu zeta_nu = (-1)^[mu<nu] zeta_nu S_mu, this
  // deferred-phase representation converts the antisymmetric Xi-product into a
  // SYMMETRIC S-sum (every ordering accumulated with +1), at the cost of a
  // residual overall (-1)^C(n,2) folded into _negated (see calculateNegation).
  // We collect the active directions in gmu[] (X,Y,Z,T) order, build each
  // permutation's chain through two ping-pong scratch buffers (oneLink is not
  // alias-safe), then apply the full spin-taste coefficient -- Follana phase
  // (±1) times the _scaling amplitude -- in one step via applyCoeffsAndPhase,
  // which is linear so phase-at-end == sum of per-chain phases.
  int shift = _spin ^ _taste;

  if (shift != 0) {
    assert(U != nullptr);
  }

  if (shift == 0) {
    applyCoeffsAndPhase(lhs, rhs); // _scaling == 1.0 for popcount 0
    return;
  }

  // Active directions in gmu[] (X,Y,Z,T) reference order.
  std::array<int, 4> dirs;
  int n = 0;
  for (int j = 0; j < static_cast<int>(gmu.size()); j++) {
    if (static_cast<int>(gmu[j]) & shift) {
      dirs[n++] = j;
    }
  }

  // Symmetrized sum over all n! orderings of the plain symmetric shifts.
  //
  // The natural operator is the antisymmetrized Xi-product, which (because the
  // Xi_mu anticommute) equals the ordered product -- the Clifford wedge
  // product, i.e. the taste matrix xi_B. In Grid's deferred-phase basis we move
  // every zeta left past the S's that follow it: each move costs
  // (-1)^[mu<nu] via S_mu zeta_nu = (-1)^[mu<nu] zeta_nu S_mu, giving a total
  // (-1)^inversions per ordering that cancels the Levi-Civita weight. The
  // result is a SYMMETRIC S-sum -- every ordering accumulates with +1 -- times
  // a residual (-1)^C(n,2) carried by _negated (see calculateNegation).
  // (An antisymmetric S-sum would
  // be wrong: {S_a,S_b} adds, while [S_a,S_b] cancels for near-abelian gauge.)
  //
  // Cost: n! permutations x 2n eager Cshift halo exchanges per call
  // (n=4 -> 192). A future fused-stencil optimization must reproduce this
  // symmetrization, not replace it.
  Lattice<obj> buf0(rhs.Grid()), buf1(rhs.Grid()), acc(rhs.Grid());
  acc = Zero();

  std::array<int, 4> perm;
  for (int i = 0; i < n; i++)
    perm[i] = i; // ascending -> covers all perms
  bool first = true;
  do {
    // Build the ordered chain: perm[0] innermost (== MILC c[0] first),
    // ping-pong buf0/buf1 so oneLink never aliases its source and destination.
    const Lattice<obj> *in = &rhs;
    Lattice<obj> *out = nullptr;
    for (int i = 0; i < n; i++) {
      out = (i & 1) ? &buf1 : &buf0;
      oneLink(*out, *in, dirs[perm[i]]);
      in = out;
    }
    // *in now holds this permutation's chain result. Every ordering accumulates
    // with +1 (not Levi-Civita): the antisymmetrized Xi-product and the
    // symmetric S-sum differ only by the shift-zeta commutation signs, which
    // cancel the Levi-Civita weight (see the symmetrization comment above), so
    // in Grid's deferred-phase basis all orderings add. The overall phase
    // (zeta-product, spin sign) is applied once at the end by applyCoeffsAndPhase.
    // Seed acc's checkerboard from the chain output's ACTUAL CB before the
    // first accumulation: on a full grid Cshift leaves CB unchanged, while on
    // a red-black grid a hop flips CB. Every chain shares the same direction
    // multiset, so all outputs share one landing CB.
    if (first) {
      acc.Checkerboard() = in->Checkerboard();
      first = false;
    }
    acc += *in;
  } while (std::next_permutation(perm.begin(), perm.begin() + n));

  // Full spin-taste coefficient: Follana phase (±1, incl. the deferred-phase
  // (-1)^C(n,2) and spin sign via _negated) times the _scaling amplitude
  // (1/2)^n / n!. applyCoeffsAndPhase sets lhs's checkerboard.
  applyCoeffsAndPhase(lhs, acc);
}

template <class obj>
void StagGamma::oneLink(Lattice<obj> &lhs, const Lattice<obj> &rhs,
                        int shift_dir) const {
  // Symmetric +/-1 covariant hop in one direction:
  //   oneLink = CovShiftBackward(U,dir,rhs) + CovShiftForward(U,dir,rhs)
  // A symmetric shift flips checkerboard parity, so the output lands on
  // (1 - rhs.Checkerboard()). For the CB bridge the backward-term gauge is
  // picked onto rhs.CB and the forward-term gauge onto the flipped CB
  // (the original code read lhs.Checkerboard() before Cshift set it).

  if (rhs.Grid()->_isCheckerBoarded) {
    LatticeColourMatrix Umu_full(U->Grid());
    Umu_full = PeekIndex<LorentzIndex>(*U, shift_dir);

    LatticeColourMatrix UmuBwd(rhs.Grid()); // gauge for backward hop
    LatticeColourMatrix UmuFwd(rhs.Grid()); // gauge for forward hop
    pickCheckerboard(rhs.Checkerboard(), UmuBwd, Umu_full);
    pickCheckerboard(1 - rhs.Checkerboard(), UmuFwd, Umu_full);

    lhs = PeriodicBC::CovShiftBackward(UmuBwd, shift_dir, rhs) +
          PeriodicBC::CovShiftForward(UmuFwd, shift_dir, rhs);
  } else {
    LatticeColourMatrix Umu(rhs.Grid());
    Umu = PeekIndex<LorentzIndex>(*U, shift_dir);
    lhs = PeriodicBC::CovShiftBackward(Umu, shift_dir, rhs) +
          PeriodicBC::CovShiftForward(Umu, shift_dir, rhs);
  }
}

inline int StagGamma::getOrientation(StagAlgebra g) {
  switch (g) {
    // XYZT convention
  case StagAlgebra::GZX:
  // case StagAlgebra::G5X:
  // case StagAlgebra::G5Z:
  case StagAlgebra::G5Y:
  case StagAlgebra::G5T:
  case StagAlgebra::G5:

    // TXYZ convention
    // case StagAlgebra::GZX:
    // case StagAlgebra::GXT:
    // case StagAlgebra::GYT:
    // case StagAlgebra::GZT:
    // case StagAlgebra::G5Y:
    // case StagAlgebra::G5T:
    return -1;
    break;
  }
  return 1;
}

inline void StagGamma::calculateNegation() {
  // _negated is the overall GLOBAL sign of the Follana phase for this
  // spin-taste pair: a position-independent +/-1 multiplier applied by
  // applyCoeffsAndPhase. It is the product of two independent contributions:

  // (1) Spin sign (Follana 2007): a parity over the active spin directions
  //     of the staggered conjugation phase.
  StagAlgebra result = _spin & LessThan(_spin ^ _taste);
  for (auto &dir : StagGamma::gmu) {
    if (dir & result) {
      toggleNegation();
    }
  }

  // (2) Deferred-phase representation sign (-1)^C(n,2), n = hop count. The
  //     taste operator is naturally the antisymmetrized Xi-product (Xi_mu =
  //     zeta_mu S_mu, the Golterman-Smit Clifford-algebra generator used by
  //     Follana 2007), which -- because the Xi_mu anticommute -- equals its
  //     own ordered product and is sign-free. Grid shifts with plain S_mu and
  //     applies zeta once at the end (applyCoeffsAndPhase). Pulling the interleaved
  //     zeta's out drags each zeta left past the S's that follow it; the
  //     shift-zeta commutation S_mu zeta_nu = (-1)^[mu<nu] zeta_nu S_mu
  //     contributes a uniform (-1)^C(n,2) = (-1)^(n(n-1)/2) -- one sign-flip
  //     per ordered zeta-S cross-pair, independent of ordering. This is a
  //     representation-conversion cost, NOT physics: it vanishes in the Xi
  //     basis and appears only because Grid defers the phase.
  //     n=2 -> -1, n=3 -> -1, n=4 -> +1  (n=0,1 -> +1, so this sign is inert
  //     for local and one-link operators).
  int n = popcountShift(_spin, _taste);
  int cross = n * (n - 1) / 2;
  if (cross & 1) toggleNegation();
}

inline void StagGamma::calculateOscillation() {

  _oscillateDirs = LessThan(_taste) ^ GreaterThan(_spin);
}

inline void StagGamma::calculatePhase() {
  // Scale down by (1/2) for each direction in the symmetric shift
  // (_spin ^ _taste): one symmetric hop per set bit. popcount 0->1.0,
  // 1->0.5, 2->0.25, 3->0.125, 4->0.0625.
  int nShift = popcountShift(_spin, _taste);
  _scaling = 1.0;
  for (int i = 0; i < nShift; i++) {
    _scaling *= 0.5; // (1/2) per symmetric hop direction
  }
  // _scaling is the pure shift AMPLITUDE (1/2)^n / n! (one 1/2 per hop
  // direction to average the +/- symmetric transport, plus the 1/n!
  // symmetrization normalization). It carries NO sign: the overall +/-1 phase
  // -- spin sign and the deferred-phase representation sign -- lives in
  // _negated, set by calculateNegation. applyCoeffsAndPhase applies BOTH the
  // phase (_negated, oscillation) and this amplitude in one step; for local
  // operators (n=0) _scaling == 1.0 so it reduces to a pure phase, while for
  // one-link (n=1) it supplies the symmetric-shift 1/2 that averages the
  // forward+backward sum.
  RealD nfact = 1.0;
  for (int i = 2; i <= nShift; i++) {
    nfact *= i;
  }
  _scaling /= nfact; // _scaling = (1/2)^n / n!  (pure amplitude, no sign)

  _negated = false;
  calculateOscillation();
  calculateNegation();

  // Include sign flip for consistent orientation of gammas ( see eqn. A4 of
  // Follana (2007) ) if (getOrientation(_spin) != getOrientation(_taste)) {
  // toggleNegation();
  // }
}

template <class obj>
void StagGamma::applyCoeffsAndPhase(Lattice<obj> &lhs, const Lattice<obj> &rhs) const {

  GridBase *grid = lhs.Grid();

  Lattice<obj> temp(grid);
  Lattice<iScalar<vInteger>> coor(grid), stag_dirs(grid);
  iScalar<vInteger> one = 1;

  // Propagate rhs's checkerboard to the internal temporaries so the where()
  // expression below is CB-conformant on red-black grids. Fresh lattices
  // default to checkerboard=Even, which mismatches an Odd rhs and trips the
  // expression-template CB-conformance assert (Lattice_ET.h). The assignments
  // below (stag_dirs = Zero()/one and the += coor loop) reset the checkerboard,
  // so we (re)apply cb immediately before every expression that uses these
  // temporaries. No-op on full grids (cb stays Even, the where() CB check does
  // not fire). Required for the A2A CB-grid pre-transform path (Strategy C):
  // applyGamma is called on both Even and Odd right vectors.
  int cb = Even;
  if (grid->_isCheckerBoarded) {
    cb = rhs.Checkerboard();
  }

  if (_negated) {
    stag_dirs = one;
  } else {
    stag_dirs = Zero();
  }
  if (grid->_isCheckerBoarded) {
    stag_dirs.Checkerboard() = cb;
  }

  for (int dir = 0; dir < gmu.size(); dir++) {
    if (gmu[dir] & _oscillateDirs) { // gmu[dir] maps Grid XYZT convention to
                                     // binary flags
      LatticeCoordinate(coor, dir);
      if (grid->_isCheckerBoarded) {
        coor.Checkerboard() = cb;
      }
      stag_dirs += coor;
    }
  }

  // Full spin-taste coefficient: Follana phase (±1 sign incl. the deferred-phase
  // (-1)^C(n,2) and spin sign via _negated, plus the zeta-product oscillation)
  // times the _scaling amplitude (1/2)^n / n!. Applied to a symmetric-shift
  // chain (here) or a raw forward+backward link sum (A2ATaskOnelink), it yields
  // the correctly normalized operator -- the 1/2 averaging the F+B sum.
  if (grid->_isCheckerBoarded) {
    stag_dirs.Checkerboard() = cb; // += coor above may have reset it
  }
  temp = where(mod(stag_dirs, 2) == 0, _scaling * rhs, -_scaling * rhs);

  lhs = std::move(temp);
}

inline StagGamma operator*(const StagGamma &g1, const StagGamma &g2) {

  StagGamma ret(g1._spin ^ g2._spin, g1._taste ^ g2._taste);

  if (g1.U != nullptr) {
    assert(g2.U == g1.U);
    ret.setGaugeField(*(g1.U));
  }

  if (g1._negated != g2._negated) {
    ret.toggleNegation();
  }

  // Following eqn. A4 of Follana (2007)
  uint8_t negate = ((g1._spin & StagGamma::LessThan(g2._spin)) ^
                    (g1._taste & StagGamma::LessThan(g2._taste)));

  for (auto &dir : StagGamma::gmu) {
    if (dir & negate) {
      ret.toggleNegation();
    }
  }

  return ret;
}

template <class obj>
inline Lattice<obj> operator*(const StagGamma &g1, const Lattice<obj> &lat) {

  Lattice<obj> temp(lat.Grid());
  g1.applyGamma(temp, lat);
  return temp;
}

template <class obj>
inline Lattice<obj> operator*(const Lattice<obj> &lat, const StagGamma &g1) {
  return g1 * lat;
}

NAMESPACE_END(Grid)
