/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/utils/HighlyImprovedStaggeredFermionImpl.h
Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

Copyright (C) 2023 

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */

/**
 * @file HighlyImprovedStaggeredFermionImpl.h
 * @brief Interface for implementation of highly improved staggered fermions (HISQ)
 * @author Curtis Taylor Peterson
 * @details
 * This header file is meant to act as an interface for both Grid and the 
 * MILC codebase to utilize the "highly improved staggered quark" action
 * within Grid. The interface itself is composed of a single core component class
 * 
 * * HighlyImprovedStaggeredFermionImpl: Class providing support for highly
 *   improved staggered fermions in Grid. EXtoses methods for fat7/asqtad 
 *   smearing and its derivative with optional inclusion of Lepage term needed 
 *   for second level of HISQ smearing. 
 * 
 * References:
 * * Quantum EXtressions (QEX): https://github.com/jcosborn/qex
 * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
 * * SIMULATEeQCD: https://github.com/LatticeQCD/SIMULATeQCD
 * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
 * * MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501
 * 
 * Acknowledgements:
 *   Curtis Taylor Peterson would like to thank James Osborn for developing/testing
 *   the implementations of HISQ in QEX and QOPQDP, from which the "fast" option for 
 *   the fat7/asqtad smearing and derivativative are based and have been tested against.
 *   
 *   This material is based upon work supported by the U.S. Department of Energy, 
 *   Office of Science, Office of Advanced Scientific Computing Research, Scientific 
 *   Discovery through Advanced Computing (SciDAC) program.
*/

#pragma once 

#ifndef QCD_UTILS_HISQ_IMPL_H
#define QCD_UTILS_HISQ_IMPL_H 

#include <Grid/qcd/utils/Transporters.h>
#include <Grid/qcd/utils/UnitaryProjection.h>
#include <Grid/qcd/action/ActionParams.h>

#include <Grid/perfmon/Tracing.h>

NAMESPACE_BEGIN(Grid);

//
// macros
//

// loop excluding no indices
#define HISQLOOP0(exec) for (int mu = 0; mu < Nd; ++mu) { exec; } \

// loop excluding one index
#define HISQLOOP1(exec) for (int nu = 0; nu < Nd; ++nu)   \
  { if (nu == mu) {continue;} else exec; }                \

// loop excluding two indices
#define HISQLOOP2(exec) for (int i = 0; i < Nd; ++i)     \
  { if ((i == mu) or (i == nu)) {continue;} else exec; } \

// loop excluding three indices
#define HISQLOOP3(exec) for (int j = 0; j < Nd; ++j)                 \
  { if ((j == mu) or (j == nu) or (j == i)) {continue;} else exec; } \

// encapsulates conditional execution of Lepage calculations for cleanliness
#define HISQLEPAGE(exec) if (ctx.lepage != 0.0) { exec; } \

// encapsulates conditional execution of Naik calculations for cleanliness
#define HISQNAIK(exec) if (ctx.naik != 0.0) { HISQLOOP0(exec) } \

// better notation for ternary eXtression
#define when(cond, valTrue, valFalse) ((cond) ? (valTrue) : (valFalse)) \

//
// useful consts
//

// halo depth
const int DEPTH = 1;

// usual 4D HISQ coefficients: defined for convenience
const double // Lepage & Naik
  LEPAGE = -1.0/8.0,
  NAIK = -1.0/24.0;
const double // fat-7
  F7L1 = -LEPAGE,
  F7L3 = -0.5*F7L1,
  F7L5 = -0.25*F7L3,
  F7L7 = 0.0625*NAIK;
const double // asqtad
  ASQL1 = -8.0*LEPAGE,
  ASQL3 = F7L3,
  ASQL5 = F7L5,
  ASQL7 = F7L7; 
const double // Naik epsilon heavy quark mass coefficients
  NAIKEPS1 = -27.0/40.0,
  NAIKEPS2 = 327.0/1120.0,
  NAIKEPS3 = -15607.0/268800.0,
  NAIKEPS4 = -73697.0/3942400.0;
const bool BACKUPSVD = true; // use backup SVD in unitary projection when applicable
const double // unitary projection parameters
  REUNITCUTOFF = 1e-20,         // base-level cutoff on eigenvalues
  REUNITDERIVCUTOFF = 5e-5,     // base-level cutoff on eigenvalues for derivative
  RELBACKUPSVDTOLERANCE = 1e-8, // relative tolerance for triggering backup SVD
  ABSBACKUPSVDTOLERANCE = 1e-8; // absolute tolerance for triggering backup SVD

//
// useful data structures
//

struct HISFContext {
  /**
   * @brief Context for highly improved staggered fermion smearing and projection
   * @author Curtis Taylor Peterson
   * @details
   * Context structure for passing parameters to HISF smearing and projection methods.
   */
  // fat7 smearing parameters
  RealD c0, c1, c2, c3;

  // asqtad parameters
  RealD lepage;
  RealD naik;

  // unitary projection parameters
  bool backupSVD;
  bool svdOnly;
  RealD relSVDTolerance;
  RealD absSVDTolerance;
  RealD eigenCutoff;

  HISFContext() { };

  HISFContext(
    RealD c0, 
    RealD c1, 
    RealD c2, 
    RealD c3,
    RealD lepage, 
    RealD naik,
    bool svdOnly,
    bool backupSVD,
    RealD relSVDTolerance,
    RealD absSVDTolerance,
    RealD eigenCutoff
  ):c0(c0), 
    c1(c1), 
    c2(c2), 
    c3(c3),
    lepage(lepage), 
    naik(naik),
    svdOnly(svdOnly),
    backupSVD(backupSVD),
    relSVDTolerance(relSVDTolerance),
    absSVDTolerance(absSVDTolerance),
    eigenCutoff(eigenCutoff) { };
  
  HISFContext(RealD c0, RealD c1, RealD c2, RealD c3, RealD lepage, RealD naik):
    c0(c0), c1(c1), c2(c2), c3(c3), lepage(lepage), naik(naik) { };
  
  HISFContext(RealD c0, RealD c1, RealD c2, RealD c3): 
    c0(c0), c1(c1), c2(c2), c3(c3), lepage(0.0), naik(0.0) { };
  
  HISFContext(
    bool svdOnly,
    bool backupSVD, 
    RealD relSVDTolerance, 
    RealD absSVDTolerance, 
    RealD eigenCutoff
  ):svdOnly(svdOnly),
    backupSVD(backupSVD), 
    relSVDTolerance(relSVDTolerance), 
    absSVDTolerance(absSVDTolerance), 
    eigenCutoff(eigenCutoff) { };
};

struct MILCContext {
  HISFContext fat7;
  HISFContext asqtad;
  std::vector<Real> naikFactors;
  std::vector<Real> naikEpsilons;
  std::vector<int> naikOrders;

  MILCContext(
    HISFContext fat7,
    HISFContext asqtad,
    std::vector<Real> naikFactors,
    std::vector<Real> naikEpsilons,
    std::vector<int> naikOrders
  ):
    fat7(fat7),
    asqtad(asqtad),
    naikFactors(naikFactors),
    naikEpsilons(naikEpsilons),
    naikOrders(naikOrders) { };
  
  int numNaiks() const { return naikEpsilons.size(); }

  int order(int species) const { return naikOrders[species]; }

  Real epsilon(int species) const { return naikEpsilons[species]; }

  Real factor(int naik) const { return naikFactors[naik]; }
};

//
// helper procedures
//

//RealD naikEpsilon(RealD am) {
//  /**
//   * @brief Calculates Naik epsilon according to MILC prescription
//   * @author Curtis Taylor Peterson
//   * @brief
//   * According to [MILC Collaboration (2010)], the Naik epsilon correction, Naik 
//   * epsilon calculated from heavy quark mass am by combining Eqn 26 and Eqn 27
//   * from [Follana, E. et al.].
//   * References:
//   * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
//   * * MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501
//   */
//  RealD result = am*am;
//  result = NAIKEPS1*result + \
//           NAIKEPS3*result*result*result + \
//           NAIKEPS4*result*result*result*result;
//  return result;
//}

//
// highly improved staggered fermion implementation
//

template <class Gimpl>
class HighlyImprovedStaggeredFermionImpl: Gimpl {
/**
 * @brief Highly improved staggered fermion implementation in Grid
 * @author Curtis Taylor Peterson
 * @details
 * Highly improved staggered fermion (HISF/HISQ) implementation in Grid. EXtoses 
 * methods for FNAL/MILC and non-FNAL/MILC implementations of Highly Improved Staggered
 * Fermions. This class boasts an implementation of HISQ that is low in communication 
 * overhead whilst retaining a low memory footprint by Grid's PaddedCell and 
 * GeneralLocalStencil through the PeriodicBC::Transporters class.
 */
  
public: INHERIT_GIMPL_TYPES(Gimpl);

private:
  typedef typename Simd::scalar_type GridScalar;
  typedef typename std::vector<GaugeLinkField> GaugeLorentzField;
  typedef typename std::vector<ComplexField> StaggeredPhases;

public:
  GridCartesian* ugrid;

  StaggeredImplParams params;
  StaggeredPhases stagPhases;

private:
  int _depth = -1;

private:
  void init(GridCartesian* grid, bool calculateStaggeredPhases) {
    assert(Nc == 3 && "HISQ only suppored for SU(3)");
    assert(Nd == 4 && "HISQ only supported for 4 dimensions");
    this->ugrid = grid;
    if (calculateStaggeredPhases) calcStagPhases(stagPhases);
  }

public:
  HighlyImprovedStaggeredFermionImpl(
    GridCartesian* grid, 
    const StaggeredImplParams params,
    bool calculateStaggeredPhases = true
  ): stagPhases(Nd, grid), params(params)
  { init(grid, calculateStaggeredPhases); }

  HighlyImprovedStaggeredFermionImpl(
    GridCartesian* grid,
    bool calculateStaggeredPhases = true
  ): stagPhases(Nd, grid), params(StaggeredImplParams(std::vector<Complex>({Complex(1),Complex(1),Complex(1),Complex(-1)})))
  { init(grid, calculateStaggeredPhases); }

private:
  void calcStagPhases(StaggeredPhases& phases) {
    /**
     * @brief HISQ gauge configuration constructor
     * @author Curtis Taylor Peterson, Peter Boyle
     * @details
     * Staggered phi follow the "MILC convention", which treats the fourth 
     * direction as the "time" coordinate:
     * (2a) eta_0 = (-1)^{x3}       <---| 
     * (2b) eta_1 = (-1)^{x3+x0}        | convention in
     * (2c) eta_2 = (-1)^{x3+x0+x1}     | this code
     * (2d) eta_3 = 1               <---|
     * Though awkard, this convention follows that of most texts on
     * relativity. This is opposed to the convention that one often finds in 
     * lattice gauge theory textbooks, where the "time" direction is x0:
     * (3a) eta_0 = 1
     * (3b) eta_1 = (-1)^{x0}
     * (3c) eta_2 = (-1)^{x0+x1}
     * (3d) eta_3 = (-1)^{x0+x1+x2}
     * Boundary conditions are imposed by "rephasing" the links 
     * on the boundary, with "1" for periodic and "-1" for anti-periodic.
     */
    GridBase *grid = phases[0].Grid();
    Lattice<iScalar<vInteger>> x(grid), y(grid), t(grid);
    Lattice<iScalar<vInteger>> tx(grid), txy(grid), xyzt(grid), coor(grid);
    ComplexField phi(grid);

    LatticeCoordinate(x, 0);
    LatticeCoordinate(y, 1);
    LatticeCoordinate(t, 3);
    tx = t + x;
    txy = tx + y;

    for (int mu = 0; mu < Nd; mu++) { 
      // for boundary phi
      int N = grid->GlobalDimensions()[mu] - 1;
      auto bpha = params.boundary_phases[mu];
      GridScalar dirichlet(real(bpha), imag(bpha));

      // staggered phases x boundary phases
      LatticeCoordinate(coor, mu);
      phi = 1.0;
      if (mu == 0) { phi = where(mod(t, 2) == (Integer)0,   phi, -phi); }
      if (mu == 1) { phi = where(mod(tx, 2) == (Integer)0,  phi, -phi); }
      if (mu == 2) { phi = where(mod(txy, 2) == (Integer)0, phi, -phi); }
      phases[mu] = where(coor == (Integer)N, dirichlet*phi, phi);
    }
  }

private:
  // wrap PokeIndex (insert gauge link field into gauge field layout)
  inline void intoGauge(GaugeField& Uout, const GaugeLinkField& U, int mu)
  { PokeIndex<LorentzIndex>(Uout, U, mu); }

  // wrap PeekIndex (extract gauge link field form gauge field layout)
  inline const GaugeLinkField toLink(const GaugeField& U, int mu)
  { return PeekIndex<LorentzIndex>(U, mu); }

  // break up memory layout of gauge field into std::vector of gauge link fields
  inline const GaugeLorentzField toLorentz(const GaugeField& U) 
  { GaugeLorentzField u(Nd, U.Grid()); HISQLOOP0(u[mu] = toLink(U, mu)) return u; }

  // aggregate std::vector of gauge link fields into layout of gauge field
  inline const GaugeField toGauge(GaugeLorentzField& u) 
  { GaugeField U(u[0].Grid()); HISQLOOP0(intoGauge(U, u[mu], mu)) return U; }

public:
  /** @brief rephases gauge links with staggered and Dirichlet boundary phases */
  void rephase(GaugeField& X, const GaugeField& W) 
  { for (int mu = 0; mu < Nd; ++mu) intoGauge(X, stagPhases[mu]*toLink(W, mu), mu); }

//
// HISQ smearing & unitary projection
//

public:
  void smear(
    GaugeField& X,
    GaugeField& WWW,
    const GaugeField& W,
    const HISFContext ctx
  ) {
    /**
     * @brief "Effective 3-link" fat7/asqtad + Lepage (optional) smear
     * @author Curtis Taylor Peterson
     * @details
     * Fat7/Asqtad links can be constructed recursively from 3-link smears. This 
     * insight is attributed to James Osborn (Argonne National Laboratory),
     * who used it to implement the Fat7/Asqtad smearing in QOPQDP and Qauntum
     * EXtressions (QEX). This method follows a similar pattern to James' 
     * implementation and combines it with Grid's padded cell. Note that many other 
     * codes, such as MILC, use a "path-based" approach by default, which tend to
     * less efficiently reuse computations from lower levels in the smearing.
     * 
     * Define
     * (1) Sν(U;n) := Uν(n) U(n+ν) Uν(n+μ)† + Uν(n-ν)† U(n-ν) U(n-ν+μ).
     * for some generic SU(N) field U(n). I call the first contribution the "upper 
     * staple" and the second the "lower staple". From (1), recursively define
     * (2) Sν(n) := Sν(Uμ;n),
     * (3) Sνj(n) := Sj(Sν;n),
     * and
     * (4) Sνij(n) := Si(Sνj;n).
     * Then an fat7 link Vμ(n) in four dimensions can be eXtressed as a composition 
     * of sums
     * (5) Vμ(n) = c0 Uμ(n)            [5a]
     *     + ∑_{ν≠μ}(c1 Sν(n)          [5b]
     *     + ∑_{i≠μ,ν}[c2 Sνi(n)       [5c]
     *     + ∑_{j≠μ,ν,i} c3 Sνij(n)])  [5d]
     * And that's it. The key thing to note is that we can reuse the staples
     * at lower levels to construct staples at higher levels.
     * 
     * The optional Lepage term adds an additional contribution to (5b):
     * (6) + lepage * Sν(Sν(n)),
     * where Sν(n) is treated as the "link" field in the Lepage staple. We can reuse
     * the code that calculates the symmetric staple, so long as we correct for the
     * additional 1-link terms that doing so adds to the smearing by compensating
     * in the 1-link coefficient. 
     * 
     * Reusing staples from lower levels comes at the cost of having to perform more
     * halo exchanges. As such, this method tends to perform poorly on the GPU 
     * architectures of the mid-2020s. This affects the derivative as well. We've 
     * a separate GPU path that does not reuse staples, hence reducing the 
     * communication overhead dramatically.
     * 
     * References:
     * * Quantum EXtressions (QEX): https://github.com/jcosborn/qex
     * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
     * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
     */
    tracePush("HighlyImprovedStaggeredFermionImpl::smear");

    PaddedCell cell(DEPTH, ugrid);
    PeriodicBC::Transporters<Gimpl> w(cell, W);
    GridBase* pgrid = (GridBase*)cell.grids.back();
    GaugeLorentzField x(Nd, pgrid);
    GaugeLinkField s3(pgrid), s5(pgrid);
    
    // 1-, 3-, 5-, and 7-link terms + lepage
    HISQLOOP0( // Eqn 5a
      x[mu] = (ctx.c0 - 6.0*ctx.lepage)*w.link(mu);
      HISQLOOP1( // Eqn 5b
        s3 = w.exchange(w.staple(mu, nu), mu, nu);
        x[mu] += ctx.c1*s3; 
        HISQLEPAGE(x[mu] += ctx.lepage*w.staple(s3, mu, nu)) // Eqn 6
        HISQLOOP2( // Eqn 5c
          s5 = w.exchange(w.staple(s3, mu, i), mu, i);
          x[mu] += ctx.c2*s5;
          HISQLOOP3(x[mu] += ctx.c3*w.staple(s5, mu, j)) // Eqn 5d
    ) ) )
    X = w.toTightGrid(toGauge(x));

    // Naik term
    HISQNAIK(x[mu] = ctx.naik*w.CovShiftFwd(mu, w.exchange(w.CovShiftFwd(mu), mu)))
    if (ctx.naik != 0.0) WWW = w.toTightGrid(toGauge(x));

    tracePop("HighlyImprovedStaggeredFermionImpl::smear");
  }

  void smear(GaugeField& X, GaugeField& WWW, const GaugeField& W) {
    HISFContext asqtadCtx(ASQL1, ASQL3, ASQL5, ASQL7, LEPAGE, NAIK); 
    smear(X, WWW, W, asqtadCtx); 
  }

  void smear(GaugeField& X, const GaugeField& W, const HISFContext ctx) 
  { smear(X, X, W, ctx); }

  void smear(GaugeField& X, const GaugeField& W) { 
    HISFContext fat7Ctx(F7L1, F7L3, F7L5, F7L7);  
    smear(X, X, W, fat7Ctx); 
  }

  void project(GaugeField& V, const GaugeField& U, const HISFContext ctx) { 
    UnitaryProjectionContext projCtx(
      ctx.svdOnly ? SingularValueDecompositionProjection : CayleyHamiltonProjection
    );
    if (!ctx.svdOnly) projCtx.setBackupSVD(ctx.backupSVD);
    projCtx.setRelativeSVDTolerance(ctx.relSVDTolerance);
    projCtx.setAbsoluteSVDTolerance(ctx.absSVDTolerance);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.project(V, U);
  }

  void project(GaugeField& V, const GaugeField& U) { 
    UnitaryProjectionContext projCtx(CayleyHamiltonProjection);
    projCtx.setBackupSVD(BACKUPSVD);
    projCtx.setRelativeSVDTolerance(RELBACKUPSVDTOLERANCE);
    projCtx.setAbsoluteSVDTolerance(ABSBACKUPSVDTOLERANCE);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.project(V, U);
  }

//
// HISQ smearing & unitary projection derivatives
//

public:
  void smearDerivative(
    GaugeField& dXdU,
    const GaugeField& dSdX,
    const GaugeField& dSdWWW,
    const GaugeField& W,
    const HISFContext ctx
  ) {
    /**
     * @brief "Effective 3-link" fat7/asqtad + Lepage (optional) smear
     * @author Curtis Taylor Peterson
     * @details
     * Calculates Wirtinger derivative of fat7/asqtad + Lepage links, including chain
     * rule. See the comments under `smear` method for details on fat7/asqtad + 
     * Lepage smearing. This method was originally implemented by translating the
     * "effective 3-link" force in QOP/QDP and Quantum EXtressions (QEX) written by 
     * James Osborn (Argonne National Laboratory) into Grid. It was then reworked to 
     * make full use of thevpadded cells in Grid, which resulted in an algorithm that 
     * is similar in spirit to that in QOP/QDP and QEX, but which deviates in how it 
     * reasons through each contribution to the derivative. The result is an 
     * algorithm that boasts a significant reduction in memory usage compared to 
     * QOP/QDP and QEX. This code has been tested against its translated counterpart,
     * which itself was thoroughly tested against QEX.
     * 
     * The full force from some smeared action S(Uμ) is
     * (1) δS/δUμ(n) = LieAlgebraProjection[ Uμ(n) ∂S/∂Uμ(n) ]
     * This method calculates ∂S/∂Uμ(n) (i.e., not δS/δUμ(n)); it is to be 
     * interpreted as a matrix Wirtinger derivative, with the rules
     * (2) [∂S/∂Uμ(n)]ij = ∂S/∂Uμ(n)ji,                      [2a]
     *     ∂Uμ(n)ij/∂Uν(m)kl = δ(i,k) δ(j,l) δ(μ,ν) δ(m,n),  [2b]
     *     ∂Uμ(n)/∂Uν†(m) = 0.                               [2c]
     * Taking Vμ(n) to be some fat7/asqtad-smeared link [Eqn. 5 in `smear` method], 
     * the chain rule gives
     * (2) ∂S/∂Uμ(n) = ∑_{δ,m} ∂S/∂Vδ(m) ∂Vδ(m)/∂Uμ(n)
     *               + ∑_{δ,m} ∂S/∂Vδ†(m) ∂Vδ†(m)/∂Uμ(n).
     * For simplicity, consider the top contribution to a single symmetric Sν(n): 
     * ν      ---🠢
     * 🠡      🠡   🠣   (Fig 1)
     * -🠢 μ   ....
     * The forward derivative of Sν(n) with respect to Uμ(n) will receive 
     * contributions from the leftmost link and the top link. To understand what
     * we're doing here, we need to consider these contributions "from the perspective
     * of the force". In other words, before we take the derivative, we think of the 
     * the contribution from the chain rule ∂S/∂Vδ as completing a plaquette 
     * (just with the like ∂S/∂Vδ pointing in the "wrong" direction); in Fig. 1, this 
     * is the dotted line below the staple. We then think of the contribution from 
     * the derivative on the left and top link by "rotating" the perspective in 
     * Fig. 1, such that the link that the derivative ∂/∂Uμ(n) hits lies on the 
     * x-axis where the contribution from ∂S/∂Vδ was. We then get
     * (3)            Uμ(n)
     *        🠠----   ----🠢 
     * ν      ⮾   🠡 + 🠡   🠣 (Fig 2)
     * 🠡      ----🠢   --⮾--
     * -🠢 μ   Uμ(n)
     *         [3a]    [3b]
     * where an "⮾" indicates a replacement with the chain rule; only when the 
     * center link is hit is the chain rule and the derivative aligned along 
     * the same direction. Additionally, notice that the orientation of the staples
     * that comprise the staple derivative are opposite to those in the origanl 
     * link; this is due to rotating the perspective to that of the force. To make 
     * diagrammatics work out without writing redundant procedures for the opposite
     * orientation, we work with the adjoint of the chain rule, calculate the 
     * derivative as if we were completing properly-oriented plaquettes, then correct
     * by returning the adjoint of the full derivative. The full contribution from the 
     * 3-link staples follows by repeating this procedure for the adjoint, reflection, 
     * and adjoint of the reflection. The 5- and 7-link staples can be constructed 
     * from the 3-link staples in a similar manner. For example, a 5-link staple will
     * get the contributions
     * (4)  🠠---- 🠠----    
     * ν    ⮾          🠡
     * 🠡    ----🠢 ----🠢
     * -🠢 μ
     * As the derivative is quite complicated, some of this may seem a bit vague.
     * Please feel free to contact Curtis Peterson (see email above) for any 
     * questions that may arise.
     * 
     * References:
     * * Quantum EXtressions (QEX): https://github.com/jcosborn/qex
     * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
     * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
     */
    tracePush("HighlyImprovedStaggeredFermionImpl::smearDerivative");

    PaddedCell cell(DEPTH, ugrid);
    PeriodicBC::Transporters<Gimpl> w(cell, W);
    GridBase* pgrid = (GridBase*)cell.grids.back();
    GaugeLorentzField dxdw = toLorentz(w.toPaddedGrid(dSdX));
    GaugeLorentzField dxdu(Nd, pgrid);
    GaugeLinkField cnu(pgrid), ci(pgrid), cj(pgrid);
    GaugeLinkField snu(pgrid), si(pgrid), sj(pgrid);
    GaugeLinkField dnu(pgrid), di(pgrid), dj(pgrid);

    // fat7 + lepage (lepage won't execute if lepage == 0.0)
    HISQLOOP0( 
      dxdu[mu] = (ctx.c0 - 6.0*ctx.lepage)*dxdw[mu];
      HISQLOOP1(
        dnu = Zero();            // Eqn 3a
        cnu = ctx.c1*dxdw[mu];   // Eqn 3b
        snu = ctx.c1*w.link(nu); // Eqn 4 
        HISQLEPAGE(
          dnu += ctx.lepage*w.staple(dxdw[nu], nu, mu);
          cnu += ctx.lepage*w.staple(dxdw[mu], mu, nu);
          snu += ctx.lepage*w.staple(nu, mu);
        )
        HISQLOOP2(
          di = w.exchange(w.staple(dxdw[nu], nu, i), nu, i);
          ci = ctx.c2*dxdw[mu];
          si = ctx.c2*w.link(nu);
          HISQLOOP3(
            dj = ctx.c3*w.staple(di, nu, j);
            cj = ctx.c3*w.staple(dxdw[mu], mu, j); 
            sj = w.exchange(ctx.c3*w.staple(nu, j), nu, j); 
          )
          dnu += ctx.c2*di + dj;
          cnu += w.staple(w.exchange(ci + cj, mu, i), mu, i); 
          snu += w.staple(w.exchange(si + sj, nu, i), nu, i);
          dxdu[mu] += w.stapleDerivative(sj, di, mu, nu); // Eqn 3a & 4
        )
        dxdu[mu] += w.stapleDerivative(w.exchange(dnu, mu, nu), mu, nu);
        dxdu[mu] += w.staple(w.exchange(cnu, mu, nu), mu, nu);           
        dxdu[mu] += w.stapleDerivative(w.exchange(snu, mu, nu), dxdw[nu], mu, nu);
    ) )

    // Naik ("long link")
    if (ctx.naik != 0.0) dxdw = toLorentz(w.toPaddedGrid(dSdWWW));
    HISQNAIK( 
      for (int term = 0; term < 3; ++term) { 
        if (term == 0) {
          si = w.link(mu);
          sj = w.exchange(w.CovShiftIdentFwd(mu, si), mu);
          snu = sj*w.CovShiftIdentFwd(mu, sj)*adj(dxdw[mu]);
        } else { snu = w.CovShiftIdentBck(mu, w.exchange(adj(sj)*snu*si, mu)); }
        dxdu[mu] += ctx.naik*adj(snu);
    } )

    // extract from padded grid
    dXdU = w.toTightGrid(toGauge(dxdu));

    tracePop("HighlyImprovedStaggeredFermionImpl::smearDerivative");
  }

  void smearDerivative(
    GaugeField& dXdU,
    const GaugeField& dSdX,
    const GaugeField& dSdWWW,
    const GaugeField& W
  ) { 
    HISFContext asqtadCtx(ASQL1, ASQL3, ASQL5, ASQL7, LEPAGE, NAIK); 
    smearDerivative(dXdU, dSdX, dSdWWW, W, asqtadCtx); 
  }

  void smearDerivative(
    GaugeField& dXdU,
    const GaugeField& dSdX,
    const GaugeField& W,
    const HISFContext ctx
  ) { smearDerivative(dXdU, dSdX, dSdX, W, ctx); }

  void smearDerivative(GaugeField& dXdU, const GaugeField& dSdX, const GaugeField& W) { 
    HISFContext fat7Ctx(F7L1, F7L3, F7L5, F7L7);  
    smearDerivative(dXdU, dSdX, W, fat7Ctx);
  }

  void projectionDerivative(
    GaugeField& dVdU, 
    const GaugeField& dZdV, 
    const GaugeField& U,
    const HISFContext ctx
  ) {
    UnitaryProjectionContext projCtx(MIMDCollaborationDerivative);
    projCtx.setDerivativeEigenvalueCutoff(ctx.eigenCutoff);
    projCtx.setSVDOnlyDerivative(ctx.svdOnly);
    projCtx.setBackupSVD(ctx.backupSVD);
    projCtx.setRelativeSVDTolerance(ctx.relSVDTolerance);
    projCtx.setAbsoluteSVDTolerance(ctx.absSVDTolerance);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.derivative(dVdU, dZdV, U); 
  }

  void projectionDerivative(
    GaugeField& dVdU, 
    const GaugeField& dZdV, 
    const GaugeField& U
  ) { 
    UnitaryProjectionContext projCtx(MIMDCollaborationDerivative);
    projCtx.setDerivativeEigenvalueCutoff(REUNITDERIVCUTOFF);
    projCtx.setBackupSVD(BACKUPSVD);
    projCtx.setRelativeSVDTolerance(RELBACKUPSVDTOLERANCE);
    projCtx.setAbsoluteSVDTolerance(ABSBACKUPSVDTOLERANCE);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.derivative(dVdU, dZdV, U); 
  }

  void projectionDerivative(
    GaugeField& dVdU, 
    const GaugeField& dZdV, 
    const GaugeField& V,
    const GaugeField& U
  ) { 
    UnitaryProjectionContext projCtx(JinOsbornDerivative);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.derivative(dVdU, dZdV, V, U); 
  }

//
// MILC interface for HISQ smearing
//

public:
  void milcSmearDerivative(
    GaugeField& UdSdU,
    const std::vector<GaugeField>& dSdX,
    const std::vector<GaugeField>& dSdWWW,
    const GaugeField& W,
    const GaugeField& V,
    const GaugeField& U,
    const MILCContext ctx
  ) {
    GridBase* grid = U.Grid();
    GaugeField tdSdW(grid), dSdW(grid), dSdV(grid), dSdU(grid);

    dSdW = Zero();
    for (int species = 0; species < ctx.numNaiks(); ++species) {
      RealD eps = ctx.epsilon(species);
      HISFContext asqCtx(
        ctx.asqtad.c0 + eps/8.0, 
        ctx.asqtad.c1, 
        ctx.asqtad.c2, 
        ctx.asqtad.c3, 
        ctx.asqtad.lepage, 
        ctx.asqtad.naik*(1.0 + eps)
      );
      smearDerivative(tdSdW, adj(dSdX[species]), adj(dSdWWW[species]), W, asqCtx);
      dSdW += tdSdW;
    }

    projectionDerivative(dSdV, dSdW, V, ctx.fat7);
    smearDerivative(dSdU, dSdV, U, ctx.fat7);
    
    for (int mu = 0; mu < Nd; ++mu) {
      auto u = PeekIndex<LorentzIndex>(U, mu);
      auto dsdu = PeekIndex<LorentzIndex>(dSdU, mu);
      PokeIndex<LorentzIndex>(UdSdU, -dsdu*adj(u), mu);
    }
  }

  template <typename FermionField>
  void milcSmearDerivative(
    GaugeField& UdSdU,
    const GaugeField& W,
    const GaugeField& V,
    const GaugeField& U,
    const std::vector<FermionField>& vecx, 
    const MILCContext ctx
  ) {
    /**
     * @brief MILC interface for full HISQ smearing derivative
     * @author David Clarke and Curtis Taylor Peterson
     * @details
     * Full HISQ smearing derivative as implemented in the MILC codebase. This method
     * is meant to provide an interface for MILC users to utilize the HISQ
     * implementation in Grid.
     * 
     * We are calculating the force using the rational approximation. The goal is 
     * that we can approximate 
     * (1) (Mdag M)^(-4/q) = alpha_0 + sum_l alpha_l/(M^dag M + beta_l). 
     * Hence the index l runs over the introduce a different "Naik epsilon" for each M. 
     * Hence, we can think of the total applicationof this operator as having an index 
     * species, running over the different Naik epsilons; for each species there is a 
     * possibly different order_species, then the operator has an index l running up to 
     * order_species. All terms with species=0 correspond to epsilon_Naik = 0.
     * 
     * References:
     * * MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501
     */
    int l = 0;
    
    GridBase* grid = vecx[0].Grid();
    GridCartesian* cgrid = static_cast<GridCartesian*>(grid);
    GridRedBlackCartesian* rbgrid = SpaceTimeGrid::makeFourDimRedBlackGrid(cgrid);

    std::vector<GaugeField> dSdX(ctx.numNaiks(), grid); 
    std::vector<GaugeField> dSdWWW(ctx.numNaiks(), grid);
    
    GaugeLinkField t(grid);

    // process MILC inputs for solution vectors with different Naik epsilons
    for (int species = 0; species < ctx.numNaiks(); ++species) {
      dSdX[species] = Zero();
      dSdWWW[species] = Zero();
      
      // outer product
      for (int i = 0; i < ctx.order(species); ++i) {
        GaugeField tdSdX(grid), tdSdWWW(grid);
        FermionField X(grid), Y(grid);
        FermionField x(rbgrid), y(rbgrid);
        
        X = Zero(), Y = Zero();
        x = Zero(), y = Zero();

        // extract Xl solution vector
        pickCheckerboard(Even, x, vecx[l]);
        setCheckerboard(X, x);

        // extract Yl solution vector
        pickCheckerboard(Odd, y, vecx[l]);
        setCheckerboard(Y, y);

        // accumulate outer products
        for (int mu = 0; mu < Nd; ++mu) {
          // 1-link contribution
          t = outerProduct(Cshift(Y, mu, 1), X); 
          t -= outerProduct(Cshift(X, mu, 1), Y);
          PokeIndex<LorentzIndex>(tdSdX, ctx.factor(l)*t, mu);
          
          // 3-link (Naik) contribution
          t = outerProduct(Cshift(Y, mu, 3), X);
          t -= outerProduct(Cshift(X, mu, 3), Y);
          PokeIndex<LorentzIndex>(tdSdWWW, ctx.factor(l)*t, mu);
        }

        // increment
        dSdX[species] += tdSdX;
        dSdWWW[species] += tdSdWWW;
        ++l;
    } }
    
    // calculate full HISQ force
    milcSmearDerivative(UdSdU, dSdX, dSdWWW, W, V, U, ctx);
  }

};

// undefine macros to prevent conflicts
#undef HISQLOOP0
#undef HISQLOOP1
#undef HISQLOOP2
#undef HISQLOOP3
#undef HISQLEPAGE
#undef HISQNAIK
#undef when

NAMESPACE_END(Grid);

#endif // QCD_UTILS_HISQ_IMPL_H