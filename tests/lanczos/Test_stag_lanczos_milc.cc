/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/Test_dwf_lanczos.cc

Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>

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
#include <Grid/Grid.h>

using namespace std;
using namespace Grid;

// typedef ImprovedStaggeredFermionR FermionOp;
// typedef typename ImprovedStaggeredFermionR::FermionField FermionField;
typedef NaiveStaggeredFermionR FermionOp;
typedef typename NaiveStaggeredFermionR::FermionField FermionField;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
              GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()),
              GridDefaultMpi());
  GridCartesian* FGrid = UGrid;
  GridRedBlackCartesian* UrbGrid =
    SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridRedBlackCartesian* FrbGrid = UrbGrid;
  printf("UGrid=%p UrbGrid=%p FGrid=%p FrbGrid=%p\n", UGrid, UrbGrid, FGrid,
         FrbGrid);

  std::vector<int> seeds4({1, 2, 3, 4});
  std::vector<int> seeds5({5, 6, 7, 8});
  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG RNG5rb(FrbGrid);
  RNG5.SeedFixedIntegers(seeds5);

  LatticeGaugeField Umu(UGrid), Ulong(UGrid), Ufat(UGrid);
  SU<Nc>::HotConfiguration(RNG4, Umu);

  // FieldMetaData header;
  // IldgReader IR;
  // IR.open("configs/l6496/l6496f211b630m0012m0363m432.ildg.s.2408");
  // IR.open("configs/lat.sample.l4444.ildg.20");
  // IR.readConfiguration(Umu,header);
  // IR.close();
  //IR.open("configs/l6496/fat6496f211b630m0012m0363m432.ildg.s.2408");
  // IR.open("configs/l6496/fatanti6496f211b630m0012m0363m432.ildg.s.2408");
  // IR.open("configs/fatlinks.l4444.ildg.20");
  // IR.open("configs/l6496/fat6496f211b630m0012m0363m432.ildg.h.2208");
  // IR.readConfiguration(Ufat,header);
  // IR.close();
  //IR.open("configs/l6496/long6496f211b630m0012m0363m432.ildg.s.2408");
  // IR.open("configs/longlinks.l4444.ildg.20");
  // IR.open("configs/l6496/longanti6496f211b630m0012m0363m432.ildg.s.2408");
  //IR.open("configs/l6496/long6496f211b630m0012m0363m432.ildg.h.2208");
  // IR.readConfiguration(Ulong,header);
  // IR.close();

  StaggeredImplParams implParams;
  implParams.boundary_phases = std::vector<Complex>{1.0,1.0,1.0,1.0};
  //  implParams.boundary_phases = std::vector<Complex>{1.0,1.0,1.0,1.0};
  implParams.twist_n_2pi_L = std::vector<Real>{0.0,0.0,0.0,0.0};

  RealD mass = 0.2, c1 = 2.0, c2 = 2.0, u0=1.0;
  FermionOp StagOperator(Umu,*FGrid,*FrbGrid,mass,c1,u0,implParams);
  // FermionOp StagOperator(Umu,Ufat,Ulong,*FGrid,*FrbGrid,mass,c1,c2,u0,implParams);
  // FermionOp StagOperator(*FGrid,*FrbGrid,mass,c1,c2,u0);

  //StagOperator.ImportGaugeSimple(Ulong,Ufat);

  // MdagMLinearOperator<FermionOp,LatticeStaggeredFermion> HermOp(StagOperator);
  SchurStaggeredOperator<FermionOp,FermionField> HermOp(StagOperator);

  // const int Nstop = 1000;
  // const int Nk = 1010;
  // const int Np = 190;
  // const int MaxIt = 5000;
  const int Nstop = 384;
  const int Nk = 384;
  const int Np = 1;
  const int MaxIt = 100;

  const int Nm = Nk + Np;

  RealD resid = 1.0e-13;

  //  Chebyshev<FermionField> Cheby(0.001, 24.0, 401);
  // Chebyshev<FermionField> Cheby(2.8, 24.0, 401);
  Chebyshev<FermionField> Cheby(21.0, 22.0, 2);

  FunctionHermOp<FermionField> OpCheby(Cheby,HermOp);
  PlainHermOp<FermionField>    Op     (HermOp);

  ImplicitlyRestartedLanczos<FermionField> IRL(OpCheby, Op, Nstop, Nk, Nm, resid, MaxIt);

  std::vector<RealD> eval(Nm);
  FermionField src(FrbGrid);

  src = FermionField::scalar_type(1.0);
  src.Checkerboard() = Even;

  // gaussian(RNG5, src);
  std::vector<FermionField> evec(Nm, FrbGrid);
  for (auto &ferm: evec) {
    ferm.Checkerboard() = Even;
  }

  int Nconv;
  IRL.calc(eval, evec, src, Nconv);

  std::cout << eval << std::endl;

  Grid_finalize();
}
