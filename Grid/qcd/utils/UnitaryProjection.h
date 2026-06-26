/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/utils/UnitaryProjection.h
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
 * @file UnitaryProjection.h
 * @brief Defines object representing unitary projection
 * @author Curtis Taylor Peterson
 * @details
 * This header file defines an object for performing unitary projection of gauge
 * links. The projection is performed using Cayley-Hamilton as a default and 
 * optionally Jacobi-based singular value decomposition (SVD) as a fallback.
 * 
 * References:
 * * Hasenfratz, A. et al.: https://doi.org/10.1103/PhysRevD.78.014515
 * * MILC Collaboration: https://doi.org/10.1103/PhysRevD.75.054502
 * * Quantum EXpressions (QEX): https://github.com/jcosborn/qex
 * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
 * 
 * Acknowledgements:
 *   Curtis Taylor Peterson would like to thank James Osborn and Xiao-Yong Jin for 
 *   developing/testing the implementations of unitary projection in Quantum 
 *   EXpressions, from which this implementation is based and has been tested against.
 *   
 *   This material is based upon work supported by the U.S. Department of Energy, 
 *   Office of Science, Office of Advanced Scientific Computing Research, Scientific 
 *   Discovery through Advanced Computing (SciDAC) program.
*/

#include <cassert>
#include <Grid/perfmon/Tracing.h>

#pragma once

#ifndef QCD_UTILS_UNITARYPROJECTION_H
#define QCD_UTILS_UNITARYPROJECTION_H

NAMESPACE_BEGIN(Grid);

const RealD SMALL = std::numeric_limits<double>::epsilon();
const RealD LARGE = 1./SMALL;

//
// useful data structures
//

enum UnitaryProjectionMethod {
  CayleyHamiltonProjection,
  SingularValueDecompositionProjection
};

enum UnitaryProjectionDerivativeMethod {
  MIMDCollaborationDerivative,
  JinOsbornDerivative
};

struct UnitaryProjectionContext {
  UnitaryProjectionMethod projectionMethod;
  UnitaryProjectionDerivativeMethod derivativeMethod;

  RealD derivativeEigenvalueCutoff = SMALL;
  RealD relativeSVDTolerance = LARGE;
  RealD absoluteSVDTolerance = SMALL;

  bool backupSVD = false;
  bool svdOnlyDerivative = false;

  UnitaryProjectionContext(
    UnitaryProjectionMethod projectionMethod = CayleyHamiltonProjection,
    UnitaryProjectionDerivativeMethod derivativeMethod = JinOsbornDerivative
  ): projectionMethod(projectionMethod), derivativeMethod(derivativeMethod) { }
  
  UnitaryProjectionContext(
    UnitaryProjectionDerivativeMethod derivativeMethod,
    UnitaryProjectionMethod projectionMethod = CayleyHamiltonProjection
  ): projectionMethod(projectionMethod), derivativeMethod(derivativeMethod) { }
  
  void setDerivativeEigenvalueCutoff(RealD cutoff) { 
    std::string err = "Cayley-Hamilton derivative only supports eigenvalue cutoff";
    assert(derivativeMethod == MIMDCollaborationDerivative && err.c_str());
    derivativeEigenvalueCutoff = cutoff; 
  }

  void setSVDOnlyDerivative(bool svdOnly) { 
    std::string err = "backup SVD only supported for Cayley-Hamilton projection";
    assert(derivativeMethod == MIMDCollaborationDerivative && err.c_str());
    svdOnlyDerivative = svdOnly; 
  }

  void setBackupSVD(bool backup) { 
    std::string err = "backup SVD only supported for Cayley-Hamilton projection";
    assert(projectionMethod == CayleyHamiltonProjection && err.c_str());
    backupSVD = backup; 
  }

  void setRelativeSVDTolerance(RealD tol) { 
    std::string err = "backup SVD only supported for Cayley-Hamilton projection";
    assert(projectionMethod == CayleyHamiltonProjection && err.c_str());
    relativeSVDTolerance = tol; 
  }

  void setAbsoluteSVDTolerance(RealD tol) { 
    std::string err = "backup SVD only supported for Cayley-Hamilton projection";
    assert(projectionMethod == CayleyHamiltonProjection && err.c_str());
    absoluteSVDTolerance = tol; 
  }
};

//
// UnitaryProjection class
//

template<class Gimpl> 
class UnitaryProjection: public Gimpl {

public: INHERIT_GIMPL_TYPES(Gimpl);

private:
  typedef typename Simd::scalar_type GridScalar;
  typedef iScalar<iScalar<iMatrix<GridScalar, Nc>>> GridScalarMatrix;

  // it is important for GPU build of Grid that this is std::complex
  typedef typename Eigen::Matrix<std::complex<double>, Nc, Nc> EigenScalarMatrix;
  typedef typename Eigen::JacobiSVD<EigenScalarMatrix> EigenSVD;

  UnitaryProjectionContext ctx;

public:
  UnitaryProjection() { 
    assert(Nc == 3 && "unitary projection only supported for SU(3) for now");
    ctx = UnitaryProjectionContext(CayleyHamiltonProjection, JinOsbornDerivative);
  }

  UnitaryProjection(UnitaryProjectionContext ctx): ctx(ctx)
  { assert(Nc == 3 && "unitary projection only supported for Nc = 3 for now"); }

private:
  EigenScalarMatrix toEigen(const GridScalarMatrix& u) {
    EigenScalarMatrix eu;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        GridScalar uij = u()()(i, j);
        eu(i, j) = std::complex<double>(real(uij), imag(uij));
    } }
    return eu;
  }

  GridScalarMatrix toGrid(const EigenScalarMatrix& u) {
    GridScalarMatrix gu;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        ComplexD uij = u(i, j);
        gu()()(i, j) = GridScalar(real(uij), imag(uij));
    } }
    return gu;
  }

private:
  void _adjugate3(GaugeLinkField &Ai, const GaugeLinkField& A) {
    GridBase *grid = A.Grid();
    GaugeLinkField T(grid);
    LatticeComplex trA(grid),trA2(grid);
    T = A*A;
    trA = trace(A), trA2 = trace(T);
    Ai = T - trA*A;
    T = 1.0;
    Ai += 0.5*(trA*trA - trA2)*T;
  }

  void _inverse3(GaugeLinkField &Ai, const GaugeLinkField& A)
  { _adjugate3(Ai,A); Ai = Ai/Determinant(A); }

  void _sylvester3(
    GaugeLinkField& X, 
    const GaugeLinkField& A, 
    const GaugeLinkField& C
  ) {
    GridBase *grid = A.Grid();
    GaugeLinkField adjA(grid);
    GaugeLinkField AC(grid), CA(grid), ACA(grid);
    GaugeLinkField adjAC(grid), CadjA(grid), adjACadjA(grid);
    LatticeComplex unit(grid), t(grid), s(grid), r(grid);
    LatticeComplex c0(grid), c1(grid), c2(grid), c3(grid);
    _adjugate3(adjA,A);
    t = trace(A), s = trace(adjA);
    r = peekColour(A,0,0)*peekColour(adjA,0,0);
    r += peekColour(A,0,1)*peekColour(adjA,1,0);
    r += peekColour(A,0,2)*peekColour(adjA,2,0);
    AC = A*C;
    CA = C*A;
    ACA = AC*A;
    adjAC = adjA*C;
    CadjA = C*adjA;
    adjACadjA = adjAC*adjA;
    unit = 1.0;
    c2 = 0.5*unit/(s*t - r);
    c0 = c2*(s + t*t);
    c3 = c2*t;
    c1 = c3/r;
    X = c0*C + c1*adjACadjA; 
    X += c2*(ACA - adjAC - CadjA);
    X -= c3*(AC + CA);
  }

  LatticeComplex _absmin(const LatticeComplex& x, const LatticeComplex& y) { 
    LatticeReal xr = toReal(x);
    LatticeReal yr = toReal(y);
    xr = abs(xr);
    return where(xr <= yr, x, y); 
  }

  LatticeComplex _absmax(const LatticeComplex& x, const LatticeComplex& y) { 
    LatticeReal xr = toReal(x);
    LatticeReal yr = toReal(y);
    xr = abs(xr);
    return where(xr >= yr, x, y); 
  }

  void _eigs3SVD(
    LatticeComplex& e0,
    LatticeComplex& e1,
    LatticeComplex& e2,
    const GaugeLinkField& u
  ) {
    GridBase* grid = u.Grid();
    {
      autoView(u_v, u, CpuRead);

      autoView(e0_v, e0, CpuWrite);
      autoView(e1_v, e1, CpuWrite);
      autoView(e2_v, e2, CpuWrite);

      thread_for(n, grid->lSites(), {
        Coordinate lcoor;
        GridScalarMatrix gu;

        grid->LocalIndexToLocalCoor(n, lcoor);
        peekLocalSite(gu, u_v, lcoor);
        EigenSVD svd(toEigen(gu));

        auto singularValues = svd.singularValues();

        std::complex<RealD> locale0 = singularValues(1)*singularValues(1);
        std::complex<RealD> locale1 = singularValues(0)*singularValues(0);
        std::complex<RealD> locale2 = singularValues(2)*singularValues(2);

        pokeLocalSite(locale0, e0_v, lcoor);
        pokeLocalSite(locale1, e1_v, lcoor);
        pokeLocalSite(locale2, e2_v, lcoor);
      });
    }
  }

  void _eigs3(
    LatticeComplex& f0,
    LatticeComplex& f1,
    LatticeComplex& f2,
    const GaugeLinkField& q,
    const GaugeLinkField& q2
  ) {
    GridBase *grid = q.Grid();
    Complex k1 = 1.0/3.0, k2 = 0.5*k1, k3 = 2.0*M_PI*k1;
    LatticeComplex ir(grid), uv(grid);
    LatticeComplex a0(grid), a1(grid), a2(grid);

    ir = SMALL, uv = 1.0; 

    a0 = k1*real(trace(q));
    a1 = k2*real(trace(q2)); 
    a2 = k2*real(trace(q*q2));

    f1 = a0*a0;
    a2 += a0*(f1 - 3.0*a1);
    a1 -= 0.5*f1;
    a1 = sqrt(abs(a1));

    a2 = _absmin(a2/_absmax(a1*a1*a1, ir), uv);
    a1 *= 2.0;
    a2 = k1*acos(a2);

    f0 = a0;
    f1 = f0 + a1*cos(a2);
    f2 = f0 + a1*cos(a2 + k3);
    f0 += a1*cos(a2 - k3);
  }

private:
  void _projectU3CH(GaugeLinkField& v, const GaugeLinkField& u) {
    /**
     * @brief U(3) unitary projection via Cayley-Hamilton or SVD
     * @author Curtis Taylor Peterson
     * @details
     * This method implements a U(3) projection of a general complex 3x3 matrix 
     * using Cayley-Hamilton or the Jacobi singular value decomposition implemented
     * by Eigen. For details about the Cayley-Hamilton approach, see the references
     * provided above; namely the OG paper by Hasenfratz et al and later work by the 
     * MILC collaboration. Please note that this method is modelled after the approach
     * taken in Quantum EXpressions by James Osborn and Xiao-Yong Jin, aside from the
     * singular value decomposion.
     */
    GridBase *grid = u.Grid();
    GaugeLinkField unity(grid), q(grid), q2(grid);
    LatticeComplex e0(grid), e1(grid), e2(grid);
    LatticeComplex f0(grid), f1(grid), f2(grid);
    LatticeComplex unit(grid), detA(grid), detB(grid);

    // Cayley-Hamilton: eigenvalues of q = u†u
    unit = 1.0, unity = 1.0;
    q = adj(u)*u; 
    q2 = q*q;
    _eigs3(e0, e1, e2, q, q2);
    detA = Determinant(q), detB = e0*e1*e2;

    // Cayley-Hamilton: "u, v, w" coefficients [Eqn. C6 of PRD(75)054502]
    f0 = sqrt(e0), f1 = sqrt(e1), f2 = sqrt(e2);
    e0 = f0 + f1 + f2;
    e1 = f0*f1;
    e2 = e1*f2;
    e1 += f0*f2 + f1*f2;

    // Cayley-Hamilton: "f0, f1, f2" coefficients [Eqn. C7 of PRD(75)054502]
    f2 = e2*(e0*e1 - e2);
    f2 = unit/f2;
    f1 = e0*e0;
    f0 = e0*e1*e1 - e2*(f1 + e1);
    f0 *= f2;
    f1 = e0*(2.0*e1 - f1) - e2;
    f1 *= f2;
    f2 *= e0;
    v = u*(f0*unity + f1*q + f2*q2);

    // Jacobi-based singular value decomposition: fallback for ill-conditioned links
    // conditions for falling back on SVD: https://doi.org/10.1103/PhysRevD.75.054502
    if (ctx.backupSVD) {{
      RealD relativeSVDTolerance = ctx.relativeSVDTolerance;
      RealD absoluteSVDTolerance = ctx.absoluteSVDTolerance;

      autoView(detA_v, detA, CpuRead);
      autoView(detB_v, detB, CpuRead);

      autoView(u_v, u, CpuRead);

      autoView(e0_v, e0, CpuRead);
      autoView(e1_v, e1, CpuRead);
      autoView(e2_v, e2, CpuRead);

      autoView(v_v, v, CpuWrite);

      thread_for(n, grid->lSites(), { // TODO: mask
        bool detDiffTooLarge, e0TooSmall, e1TooSmall, e2TooSmall;
        Coordinate lcoor;
        GridScalar localDetA, localDetB;
        GridScalar locale0, locale1, locale2;

        grid->LocalIndexToLocalCoor(n, lcoor);

	      peekLocalSite(localDetA, detA_v, lcoor);
        peekLocalSite(localDetB, detB_v, lcoor);

	      peekLocalSite(locale0, e0_v, lcoor);
        peekLocalSite(locale1, e1_v, lcoor);
        peekLocalSite(locale2, e2_v, lcoor);

        detDiffTooLarge = abs((localDetA - localDetB)/localDetB) > relativeSVDTolerance;

	      e0TooSmall = abs(locale0) < absoluteSVDTolerance;
        e1TooSmall = abs(locale1) < absoluteSVDTolerance;
        e2TooSmall = abs(locale2) < absoluteSVDTolerance;

        if (detDiffTooLarge or e0TooSmall or e1TooSmall or e2TooSmall) {
          GridScalarMatrix gu;
          EigenScalarMatrix eu, ev = EigenScalarMatrix::Zero();
          
          peekLocalSite(gu, u_v, lcoor);
          EigenSVD svd(toEigen(gu), Eigen::ComputeFullU | Eigen::ComputeFullV);
          ev = svd.matrixU() * svd.matrixV().adjoint();
          pokeLocalSite(toGrid(ev), v_v, lcoor);
        }
      });
    }}
  }

  void _projectU3SVD(GaugeLinkField& v, const GaugeLinkField& u) {
    /**
     * @brief U(3) unitary projection via SVD
     * @author Curtis Taylor Peterson
     * @details
     * This method implements a U(3) projection of a general complex 3x3 matrix 
     * using the Jacobi singular value decomposition implemented by Eigen.
     */
    GridBase *grid = u.Grid();

    {
      autoView(u_v, u, CpuRead);
      autoView(v_v, v, CpuWrite);

      // Jacobi-based singular value decomposition
      thread_for(n, grid->lSites(), {
        Coordinate lcoor;
        GridScalarMatrix gu;
        EigenScalarMatrix eu, ev = EigenScalarMatrix::Zero();
        
        grid->LocalIndexToLocalCoor(n, lcoor);

        peekLocalSite(gu, u_v, lcoor);
        EigenSVD svd(toEigen(gu), Eigen::ComputeFullU | Eigen::ComputeFullV);
        ev = svd.matrixU() * svd.matrixV().adjoint();
        pokeLocalSite(toGrid(ev), v_v, lcoor);
      });
    }
  }

  void _projectU3(GaugeLinkField& v, const GaugeLinkField& u) {
    switch (ctx.projectionMethod) {
      case CayleyHamiltonProjection: _projectU3CH(v, u); break;
      case SingularValueDecompositionProjection: _projectU3SVD(v, u); break;
      default: assert(false && "invalid unitary projection method");
    }
  }

public:
  void _derivativeU3JO(
    GaugeLinkField& dvdu, 
    const GaugeLinkField& dzdv,
    const GaugeLinkField& v,
    const GaugeLinkField& u
  ) {
    /**
     * @brief Derivative of unitary projection
     * @author Curtis Taylor Peterson
     * @details
     * This is a very clever approach to calculating the derivative of the unitary 
     * projection; it was invented by the authors of the original HISQ paper 
     * [PRD72(2007)054502] and expanded upon by James Osborn and Xiao-Yong Jin in the 
     * Quantum EXpressions code. The objective is to calculate 
     * (1) dQ/dU = CZ + U dZ/dU,
     * where
     * (2) Z = (X'X)^{-1/2}
     * (3) Q = XZ,
     * (4) C = dX/dU.
     * We can solve for dZ/dU using the identity
     * (5) dZ/dU = -Q'CZ = Y dZ/dU + dZ/dU Y
     * with
     * (6) Y = (X'X)^{1/2}.
     * This is a so-called "Sylvester" system of equations, and it has an analytic 
     * solution for N = 3 that is calculated in the _sylvester3 method.
     */
    GridBase *grid = u.Grid();
    GaugeLinkField t1(grid), t2(grid), t3(grid);

    _inverse3(t1, v);        // (u'u)^(1/2) u^-1
    t2 = t1*u;               // (u'u)^(1/2)  [6]
    _inverse3(t3, t2);       // (u'u)^(-1/2) [3]
    dvdu = dzdv*t3;          //
    t1 = adj(v)*dvdu;        // second equality of Eqn. [5]
    _sylvester3(t3, t2, t1); // solve Sylvester [5]
    t2 = t3 + adj(t3);       // d/dX & d/dX'
    dvdu -= u*t2;            //
  }

  void _derivativeU3MILC(
    GaugeLinkField& dvdu, 
    const GaugeLinkField& dzdv,
    const GaugeLinkField& u
  ) {
    /**
     * @brief Derivative of unitary projection via Cayley-Hamilton
     * @author David Clarke and Curtis Taylor Peterson
     * @details
     * This method implements the derivative of the U(3) projection using the 
     * Cayley-Hamilton approach. For details about this approach, see the references
     * provided above; namely the OG paper by Hasenfratz et al and later work by the 
     * MILC collaboration.
     * 
     * See _derivativeU3JO for description of derivative.
     */
    std::string err = "svdOnlyDerivative = true and backupSVD = true incompatible";
    assert((!(ctx.svdOnlyDerivative && ctx.backupSVD)) && err.c_str());

    GridBase* grid = u.Grid();
    GaugeLinkField unity(grid), q(grid), q2(grid);
    GaugeLinkField d0(grid), d1(grid), d2(grid);
    LatticeComplex e0(grid), e1(grid), e2(grid);
    LatticeComplex f0(grid), f1(grid), f2(grid);
    LatticeComplex unit(grid), zero(grid);
    LatticeComplex detA(grid), detB(grid), relDetDiff(grid);
    LatticeReal eps(grid), minei(grid);

    // numerical constants
    unit = 1.0, unity = 1.0, zero = 0.0, eps = ctx.derivativeEigenvalueCutoff;

    // eigenvalues of q = u†u
    q = adj(u)*u; 
    q2 = q*q;
    if (ctx.svdOnlyDerivative) _eigs3SVD(e0, e1, e2, u); 
    else {
      _eigs3(e0, e1, e2, q, q2);
      if (ctx.backupSVD) { 
        detA = Determinant(q); 
        detB = e0*e1*e2;
        relDetDiff = (detA - detB)/detB; 
      }
    }

    // Jacobi-based singular value decomposition: fallback for ill-conditioned links
    // conditions for falling back on SVD: https://doi.org/10.1103/PhysRevD.75.054502
    // Replaces eigenvalues of Vdag V wtih squared eigenvalues of SVD if conditions are met
    if ((ctx.backupSVD) && (!ctx.svdOnlyDerivative)) {
      LatticeComplex oe0 = e0, oe1 = e1, oe2 = e2;
      RealD relativeSVDTolerance = ctx.relativeSVDTolerance;
      RealD absoluteSVDTolerance = ctx.absoluteSVDTolerance;

      autoView(u_v, u, CpuRead);

      autoView(relDetDiff_v, relDetDiff, CpuRead);

      autoView(oe0_v, oe0, CpuRead);
      autoView(oe1_v, oe1, CpuRead);
      autoView(oe2_v, oe2, CpuRead);

      autoView(e0_v, e0, CpuWrite);
      autoView(e1_v, e1, CpuWrite);
      autoView(e2_v, e2, CpuWrite);

      thread_for(n, grid->lSites(), { // TODO: mask
        bool detDiffTooLarge, e0TooSmall, e1TooSmall, e2TooSmall;
        Coordinate lcoor;
        GridScalar localRelDetDiff;
        GridScalar localoe0, localoe1, localoe2;

        grid->LocalIndexToLocalCoor(n, lcoor);

        peekLocalSite(localRelDetDiff, relDetDiff_v, lcoor);

	      peekLocalSite(localoe0, oe0_v, lcoor);
        peekLocalSite(localoe1, oe1_v, lcoor);
        peekLocalSite(localoe2, oe2_v, lcoor);

        detDiffTooLarge = abs(localRelDetDiff) > relativeSVDTolerance;
	      e0TooSmall = abs(localoe0) < absoluteSVDTolerance;
        e1TooSmall = abs(localoe1) < absoluteSVDTolerance;
        e2TooSmall = abs(localoe2) < absoluteSVDTolerance;

        // Eqn (C22) of https://doi.org/10.1103/PhysRevD.75.054502
        if (detDiffTooLarge or e0TooSmall or e1TooSmall or e2TooSmall) {
          GridScalarMatrix gu;
          EigenScalarMatrix eu, ev = EigenScalarMatrix::Zero();
          
          peekLocalSite(gu, u_v, lcoor);
          EigenSVD svd(toEigen(gu));

          auto singularValues = svd.singularValues();

          std::complex<RealD> locale0 = singularValues(1)*singularValues(1);
          std::complex<RealD> locale1 = singularValues(0)*singularValues(0);
          std::complex<RealD> locale2 = singularValues(2)*singularValues(2);

          pokeLocalSite(locale0, e0_v, lcoor);
          pokeLocalSite(locale1, e1_v, lcoor);
          pokeLocalSite(locale2, e2_v, lcoor);
        }
      });
    }

    // force filter [Eqn. C23 of PRD(75)054502]
    minei = toReal(_absmin(e0, _absmin(e1, e2)));
    e0 = where(minei < eps, e0 + ctx.derivativeEigenvalueCutoff, e0);
    e1 = where(minei < eps, e1 + ctx.derivativeEigenvalueCutoff, e1);
    e2 = where(minei < eps, e2 + ctx.derivativeEigenvalueCutoff, e2);

    // Cayley-Hamilton: "u, v, w" coefficients [Eqn. C6 of PRD(75)054502]
    f0 = sqrt(e0), f1 = sqrt(e1), f2 = sqrt(e2);
    e0 = f0 + f1 + f2;
    e1 = f0*f1;
    e2 = e1*f2;
    e1 += f0*f2 + f1*f2;

    // Cayley-Hamilton: "f0, f1, f2" coefficients [Eqn. C7 of PRD(75)054502]
    f2 = e2*(e0*e1 - e2);
    f2 = unit/f2;
    f1 = e0*e0;
    f0 = e0*e1*e1 - e2*(f1 + e1);
    f0 *= f2;
    f1 = e0*(2.0*e1 - f1) - e2;
    f1 *= f2;
    f2 *= e0;

    // calculate derivative
    d0 = f0*unity + f1*q + f2*q2; // (u'u)^(-1/2)
    _inverse3(d1, d0);            // (u'u)^(1/2)
    d2 = u*d0;                    // u (u'u)^(-1/2)
    dvdu = dzdv*d0;
    d0 = adj(d2)*dvdu;
    _sylvester3(d2, d1, d0);
    d0 = d2 + adj(d2);
    dvdu -= u*d0;
  }

public:
  void project(GaugeLinkField& v, const GaugeLinkField& u) { 
    tracePush("UnitaryProjection::project");
    _projectU3(v, u); 
    tracePop("UnitaryProjection::project");
  }

  void project(GaugeField& V, const GaugeField& U) {
    GaugeLinkField v(U.Grid());

    tracePush("UnitaryProjection::project");
    for (int mu = 0; mu < Nd; ++mu) {
      _projectU3(v, PeekIndex<LorentzIndex>(U, mu));
      PokeIndex<LorentzIndex>(V, v, mu);
    }
    tracePop("UnitaryProjection::project");
  }

  void derivative(GaugeField& dVdU, const GaugeField& dZdV, const GaugeField& U) {
    GaugeLinkField dzdv(U.Grid()), dvdu(U.Grid());
    GaugeLinkField u(U.Grid());

    tracePush("UnitaryProjection::derivative");
    for (int mu = 0; mu < Nd; ++mu){
      u = PeekIndex<LorentzIndex>(U, mu);
      dzdv = PeekIndex<LorentzIndex>(dZdV, mu);
      switch (ctx.derivativeMethod) {
        case MIMDCollaborationDerivative: _derivativeU3MILC(dvdu, dzdv, u); break;
        case JinOsbornDerivative: {
          GaugeLinkField v(U.Grid());
          _projectU3(v, u);
          _derivativeU3JO(dvdu, dzdv, v, u);
          break;
        }
        default: assert(false && "invalid unitary projection derivative method");
      } 
      PokeIndex<LorentzIndex>(dVdU, dvdu, mu);
    }
    tracePop("UnitaryProjection::derivative");
  }

  void derivative(
    GaugeField& dVdU, 
    const GaugeField& dZdV, 
    const GaugeField& V, 
    const GaugeField& U
  ) {
    std::string err = "explicit specification of reunitarized link only supported for Jin-Osborn derivative";
    assert(ctx.derivativeMethod == JinOsbornDerivative && err.c_str());

    tracePush("UnitaryProjection::derivative");
    GaugeLinkField dvdu(U.Grid());
    for (int mu = 0; mu < Nd; ++mu){
      _derivativeU3JO(
        dvdu, 
        PeekIndex<LorentzIndex>(dZdV, mu), 
        PeekIndex<LorentzIndex>(V, mu),
        PeekIndex<LorentzIndex>(U, mu)
      );
      PokeIndex<LorentzIndex>(dVdU, dvdu, mu);
    }
    tracePop("UnitaryProjection::derivative");
  }

};

NAMESPACE_END(Grid);

#endif // QCD_UTILS_UNITARYPROJECTION_H

/*
private:
  void _CayleyHamiltonU3(
    GaugeLinkField& v,
    const GaugeLinkField& u,
    const GaugeLinkField& q,
    const GaugeLinkField& q2,
    const LatticeComplex& eig0,
    const LatticeComplex& eig1,
    const LatticeComplex& eig2
  ) {
    GridBase* grid = u.Grid();
    GaugeLinkField unity(grid);
    LatticeComplex e0(grid), e1(grid), e2(grid);
    LatticeComplex f0(grid), f1(grid), f2(grid);
    LatticeComplex unit(grid);

    // Cayley-Hamilton: "u, v, w" coefficients [Eqn. C6 of PRD(75)054502]
    f0 = sqrt(eig0), f1 = sqrt(eig1), f2 = sqrt(eig2);
    e0 = f0 + f1 + f2;
    e1 = f0*f1;
    e2 = e1*f2;
    e1 += f0*f2 + f1*f2;

    // Cayley-Hamilton: "f0, f1, f2" coefficients [Eqn. C7 of PRD(75)054502]
    f2 = e2*(e0*e1 - e2);
    f2 = unit/f2;
    f1 = e0*e0;
    f0 = e0*e1*e1 - e2*(f1 + e1);
    f0 *= f2;
    f1 = e0*(2.0*e1 - f1) - e2;
    f1 *= f2;
    f2 *= e0;

    // final projection
    v = u*(f0*unity + f1*q + f2*q2);
  }

  void _JacobiU3(
    GaugeLinkField& v,
    const GaugeLinkField& u,
    const GaugeLinkField& q,
    const LatticeComplex& e0,
    const LatticeComplex& e1,
    const LatticeComplex& e2
  ) {
    // Jacobi-based singular value decomposition: fallback for ill-conditioned links
    // conditions for falling back on SVD: https://doi.org/10.1103/PhysRevD.75.054502
    GridBase* grid = u.Grid();
    LatticeComplex detA = Determinant(q), detB = e0*e1*e2;

    {
      autoView(detA_v, detA, CpuRead);
      autoView(detB_v, detB, CpuRead);
      autoView(u_v, u, CpuRead);
      autoView(e0_v, e0, CpuRead);
      autoView(e1_v, e1, CpuRead);
      autoView(e2_v, e2, CpuRead);
      autoView(v_v, v, CpuWrite);

      thread_for(n, grid->lSites(), { // TODO: mask
        Coordinate lcoor;
        GridScalar localDetA, localDetB;

        grid->LocalIndexToLocalCoor(n, lcoor);
        peekLocalSite(localDetA, detA_v, lcoor);
        peekLocalSite(localDetB, detB_v, lcoor);

        if (abs(localDetA - localDetB) > svdtol) {
          GridScalarMatrix gu;
          EigenScalarMatrix eu, ev = EigenScalarMatrix::Zero();
          
          peekLocalSite(gu, u_v, lcoor);
          EigenSVD svd(toEigen(gu), Eigen::ComputeFullU | Eigen::ComputeFullV);
          ev = svd.matrixU() * svd.matrixV().adjoint();
          pokeLocalSite(toGrid(ev), v_v, lcoor);
        }
      });
    }
  }

  void _projectU3(GaugeLinkField& v, const GaugeLinkField& u) {
    #
    # * @brief U(3) unitary projection via Cayley-Hamilton or SVD
    # * @author Curtis Taylor Peterson
    # * @details
    # * This method implements a U(3) projection of a general complex 3x3 matrix 
    # * using Cayley-Hamilton or the Jacobi singular value decomposition implemented
    # * by Eigen. For details about the Cayley-Hamilton approach, see the references
    # * provided above; namely the OG paper by Hasenfratz et al and later work by the 
    # * MILC collaboration. Please note that this method is modelled after the approach
    # * taken in Quantum EXpressions by James Osborn and Xiao-Yong Jin.
    # 
    GridBase* grid = u.Grid();
    GaugeLinkField unity(grid), q(grid), q2(grid);
    LatticeComplex e0(grid), e1(grid), e2(grid);

    _eigs3(e0, e1, e2, q, q2, u);
    _CayleyHamiltonU3(v, u, q, q2, e0, e1, e2);
    if (backupSVD) { _JacobiU3(v, u, q, e0, e1, e2); }
  }
*/
