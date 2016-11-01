/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/fermion/ImprovedStaggeredFermion.cc

Copyright (C) 2015

Author: Azusa Yamaguchi, Peter Boyle

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
#include <Grid.h>

namespace Grid {
namespace QCD {

const std::vector<int> 
ImprovedStaggeredFermionStatic::directions({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
const std::vector<int> 
ImprovedStaggeredFermionStatic::displacements({1, 1, 1, 1, -1, -1, -1, -1, 3, 3, 3, 3, -3, -3, -3, -3});

/////////////////////////////////
// Constructor and gauge import
/////////////////////////////////

template <class Impl>
ImprovedStaggeredFermion<Impl>::ImprovedStaggeredFermion(GaugeField &_Umu, GridCartesian &Fgrid,
							 GridRedBlackCartesian &Hgrid, RealD _mass,
							 const ImplParams &p)
    : Kernels(p),
      _grid(&Fgrid),
      _cbgrid(&Hgrid),
      Stencil(&Fgrid, npoint, Even, directions, displacements),
      StencilEven(&Hgrid, npoint, Even, directions, displacements),  // source is Even
      StencilOdd(&Hgrid, npoint, Odd, directions, displacements),  // source is Odd
      mass(_mass),
      Lebesgue(_grid),
      LebesgueEvenOdd(_cbgrid),
      Umu(&Fgrid),
      UmuEven(&Hgrid),
      UmuOdd(&Hgrid),
      UUUmu(&Fgrid),
      UUUmuEven(&Hgrid),
      UUUmuOdd(&Hgrid) {
  // Allocate the required comms buffer
  ImportGauge(_Umu);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::ImportGauge(const GaugeField &_Umu) {
  GaugeField HUmu(_Umu._grid);
  HUmu = _Umu * (-0.5);
  Impl::DoubleStore(GaugeGrid(), Umu, UUUmu, HUmu);
  pickCheckerboard(Even, UmuEven, Umu);
  pickCheckerboard(Odd, UmuOdd, Umu);
  pickCheckerboard(Even, UUUmuEven, UUUmu);
  pickCheckerboard(Odd, UUUmuOdd, UUUmu);
}

/////////////////////////////
// Implement the interface
/////////////////////////////

template <class Impl>
RealD ImprovedStaggeredFermion<Impl>::M(const FermionField &in, FermionField &out) {
  out.checkerboard = in.checkerboard;
  Dhop(in, out, DaggerNo);
  return axpy_norm(out, mass, in, out);
}

template <class Impl>
RealD ImprovedStaggeredFermion<Impl>::Mdag(const FermionField &in, FermionField &out) {
  out.checkerboard = in.checkerboard;
  Dhop(in, out, DaggerYes);
  return axpy_norm(out, mass, in, out);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Meooe(const FermionField &in, FermionField &out) {
  if (in.checkerboard == Odd) {
    DhopEO(in, out, DaggerNo);
  } else {
    DhopOE(in, out, DaggerNo);
  }
}
template <class Impl>
void ImprovedStaggeredFermion<Impl>::MeooeDag(const FermionField &in, FermionField &out) {
  if (in.checkerboard == Odd) {
    DhopEO(in, out, DaggerYes);
  } else {
    DhopOE(in, out, DaggerYes);
  }
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Mooee(const FermionField &in, FermionField &out) {
  out.checkerboard = in.checkerboard;
  typename FermionField::scalar_type scal(mass);
  out = scal * in;
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::MooeeDag(const FermionField &in, FermionField &out) {
  out.checkerboard = in.checkerboard;
  Mooee(in, out);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::MooeeInv(const FermionField &in, FermionField &out) {
  out.checkerboard = in.checkerboard;
  out = (1.0 / (mass)) * in;
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::MooeeInvDag(const FermionField &in,
                                      FermionField &out) {
  out.checkerboard = in.checkerboard;
  MooeeInv(in, out);
}

///////////////////////////////////
// Internal
///////////////////////////////////

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DerivInternal(StencilImpl &st, DoubledGaugeField &U, DoubledGaugeField &UUU, 
						   GaugeField & mat,
						   const FermionField &A, const FermionField &B, int dag) {
  assert((dag == DaggerNo) || (dag == DaggerYes));

  Compressor compressor;

  FermionField Btilde(B._grid);
  FermionField Atilde(B._grid);
  Atilde = A;

  st.HaloExchange(B, compressor);

  for (int mu = 0; mu < Nd; mu++) {

    ////////////////////////
    // Call the single hop
    ////////////////////////
    PARALLEL_FOR_LOOP
    for (int sss = 0; sss < B._grid->oSites(); sss++) {
      Kernels::DhopDir(st, U, UUU, st.CommBuf(), sss, sss, B, Btilde, mu,1);
    }

    // Force in three link terms
    //
    //    Impl::InsertForce4D(mat, Btilde, Atilde, mu);
    //
    // dU_ac(x)/dt = i p_ab U_bc(x)
    //
    // => dS_f/dt = dS_f/dU_ac(x) . dU_ac(x)/dt =  i p_ab U_bc(x) dS_f/dU_ac(x) 
    //
    // One link: form fragments S_f = A U B 
    //
    //         write Btilde = U(x) B(x+mu)
    //
    // mat+= TraceIndex<SpinIndex>(outerProduct(Btilde,A)); 
    // 
    // Three link: form fragments S_f = A UUU B 
    //
    // mat+= outer ( A, UUUB) <-- Best take DhopDeriv with one linke or identity matrix
    // mat+= outer ( AU, UUB) <-- and then use covariant cshift?
    // mat+= outer ( AUU, UB) <-- Returned from call to DhopDir

    assert(0);// need to figure out the force interface with a blasted three link term.
    
  }
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDeriv(GaugeField &mat, const FermionField &U, const FermionField &V, int dag) {

  conformable(U._grid, _grid);
  conformable(U._grid, V._grid);
  conformable(U._grid, mat._grid);

  mat.checkerboard = U.checkerboard;

  DerivInternal(Stencil, Umu, UUUmu, mat, U, V, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDerivOE(GaugeField &mat, const FermionField &U, const FermionField &V, int dag) {

  conformable(U._grid, _cbgrid);
  conformable(U._grid, V._grid);
  conformable(U._grid, mat._grid);

  assert(V.checkerboard == Even);
  assert(U.checkerboard == Odd);
  mat.checkerboard = Odd;

  DerivInternal(StencilEven, UmuOdd, UUUmuOdd, mat, U, V, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDerivEO(GaugeField &mat, const FermionField &U, const FermionField &V, int dag) {

  conformable(U._grid, _cbgrid);
  conformable(U._grid, V._grid);
  conformable(U._grid, mat._grid);

  assert(V.checkerboard == Odd);
  assert(U.checkerboard == Even);
  mat.checkerboard = Even;

  DerivInternal(StencilOdd, UmuEven, UUUmuEven, mat, U, V, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Dhop(const FermionField &in, FermionField &out, int dag) {
  conformable(in._grid, _grid);  // verifies full grid
  conformable(in._grid, out._grid);

  out.checkerboard = in.checkerboard;

  DhopInternal(Stencil, Lebesgue, Umu, UUUmu, in, out, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopOE(const FermionField &in, FermionField &out, int dag) {
  conformable(in._grid, _cbgrid);    // verifies half grid
  conformable(in._grid, out._grid);  // drops the cb check

  assert(in.checkerboard == Even);
  out.checkerboard = Odd;

  DhopInternal(StencilEven, LebesgueEvenOdd, UmuOdd, UUUmuOdd, in, out, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopEO(const FermionField &in, FermionField &out, int dag) {
  conformable(in._grid, _cbgrid);    // verifies half grid
  conformable(in._grid, out._grid);  // drops the cb check

  assert(in.checkerboard == Odd);
  out.checkerboard = Even;

  DhopInternal(StencilOdd, LebesgueEvenOdd, UmuEven, UUUmuEven, in, out, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Mdir(const FermionField &in, FermionField &out, int dir, int disp) {
  DhopDir(in, out, dir, disp);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDir(const FermionField &in, FermionField &out, int dir, int disp) {

  Compressor compressor;
  Stencil.HaloExchange(in, compressor);

PARALLEL_FOR_LOOP
  for (int sss = 0; sss < in._grid->oSites(); sss++) {
    Kernels::DhopDir(Stencil, Umu, UUUmu, Stencil.CommBuf(), sss, sss, in, out, dir, disp);
  }
};

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopInternal(StencilImpl &st, LebesgueOrder &lo,
						  DoubledGaugeField &U,
						  DoubledGaugeField &UUU,
						  const FermionField &in,
						  FermionField &out, int dag) {
  assert((dag == DaggerNo) || (dag == DaggerYes));

  Compressor compressor;
  st.HaloExchange(in, compressor);

  if (dag == DaggerYes) {
    PARALLEL_FOR_LOOP
    for (int sss = 0; sss < in._grid->oSites(); sss++) {
      Kernels::DhopSiteDag(st, lo, U, UUU, st.CommBuf(), sss, sss, in, out);
    }
  } else {
    PARALLEL_FOR_LOOP
    for (int sss = 0; sss < in._grid->oSites(); sss++) {
      Kernels::DhopSite(st, lo, U, UUU, st.CommBuf(), sss, sss, in, out);
    }
  }
};

FermOpStaggeredTemplateInstantiate(ImprovedStaggeredFermion);

  //AdjointFermOpTemplateInstantiate(ImprovedStaggeredFermion);
  //TwoIndexFermOpTemplateInstantiate(ImprovedStaggeredFermion);

}}