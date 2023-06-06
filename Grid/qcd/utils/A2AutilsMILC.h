#pragma once
#include <Grid/Grid_Eigen_Tensor.h>

#ifndef accelerator_for2dNB
#define accelerator_for2dNB_no_err(iter1, num1, iter2, num2, nsimd, ... )  accelerator_for2d(iter1, num1, iter2, num2, nsimd, { __VA_ARGS__ } );
#else
#define accelerator_for2dNB_no_err(iter1, num1, iter2, num2, nsimd, ... )  accelerator_for2dNB(iter1, num1, iter2, num2, nsimd, { __VA_ARGS__ } );
#endif

#ifndef MF_SUM_ARRAY_SIZE
#define MF_SUM_ARRAY_SIZE 16
#endif

NAMESPACE_BEGIN(Grid);

template <typename FImpl>
class A2AutilsMILC 
{
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

  typedef typename FImpl::SiteSpinor vobj;

  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  typedef iSinglet<vector_type> Scalar_v;
  typedef iSinglet<scalar_type> Scalar_s;

  typedef decltype(coalescedRead(Scalar_v())) calcScalar;
  typedef decltype(coalescedRead(vobj())) calcSpinor;
  typedef decltype(coalescedRead(vColourMatrix())) calcColourMatrix;
  
  typedef LatticeView<vobj> FermView;
  typedef CartesianStencil<vobj,vobj,int> FermStencil;
  typedef CartesianStencilView<vobj,vobj,int> FermStencilView;

  typedef LatticeView<vColourMatrix> GaugeView;

  typedef typename ComplexField::vector_object cobj;
  typedef LatticeView<cobj> ComplexView;

public:
  struct utilHelper {
  public:
    std::vector<int> shiftDirs, shiftDisplacements;
    Vector<GaugeView> viewGauge;
    Vector<ComplexView> viewGamma, viewMom;
    Vector<Coordinate> iCoorContainer;
    Vector<Integer> oCoordsEven, oCoordsOdd, gammaIndicesComm, gammaIndicesLocal;
    int checkerL, checkerR, sizeL, sizeR, sizeLOut, sizeROut, nMom, orthogDir;
    GridBase *grid;
    Vector<FermView> viewLeft_e, viewLeft_o, viewRight_e, viewRight_o;
    Vector<FermStencilView> viewStencilRight_e, viewStencilRight_o;
    Vector<FermStencilView> viewStencilLeft_e,  viewStencilLeft_o;

    Vector<FermView> &getView(bool leftView, int checkerboard) {
      if (checkerboard == Even) {
        if (leftView) {
          return viewLeft_e;
        } else {
          return viewRight_e;
        }
      } else {
        if (leftView) {
          return viewLeft_o;
        } else {
          return viewRight_o;
        }
      }
    }

    Vector<FermStencilView> &getStencilView(bool leftView, int checkerboard) {
      if (checkerboard == Even) {
        if (leftView) {
          return viewStencilLeft_e;
        } else {
          return viewStencilRight_e;
        }
      } else {
        if (leftView) {
          return viewStencilLeft_o;
        } else {
          return viewStencilRight_o;
        }
      }
    }
  };

public:
  template <typename TensorType> // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  static void StagMesonFieldNoGlobalSum(TensorType &mat,
                                     const FermionField *lhs_wi_e, const FermionField *lhs_wi_o,
                                     const FermionField *rhs_vj_e, const FermionField *rhs_vj_o,
                                     const std::vector<StagGamma::SpinTastePair>& gammas,
                                     const std::vector<ComplexField > &mom,
                                     int orthogDir, LatticeGaugeField* U = nullptr, double *t_kernel = nullptr);

  template <typename TensorType>
  static void contractSimd(TensorType &result, bool do_comm, 
                            Vector<Scalar_v> &simd_sum_e, Vector<Scalar_v> &simd_sum_o, utilHelper &helper);
  template <typename TensorType>
  static void contractSimd(TensorType &result, Vector<Scalar_v> &simd_sum, Vector<Integer> &gamma_indices, utilHelper &helper);

  static void spatialContractComm(Vector<Scalar_v>& result, int checkerboardL, int checkerboardR, utilHelper &helper);
  static void spatialContractLocal(Vector<Scalar_v>& result, int checkerboardL, int checkerboardR, utilHelper &helper);

  static void init(const std::vector<StagGamma::SpinTastePair>& gammas, LatticeGaugeField* U, GridBase *grid, utilHelper &helper){


    StagGamma spinTaste;
    if (U != nullptr) {
      spinTaste.setGaugeField(*U);
    }

    // Organize gammas into local/non-local
    for (int i = 0; i < gammas.size(); i++) {
  
      spinTaste.setSpinTaste(gammas[i]);
      int shift = (spinTaste._spin ^ spinTaste._taste);
      if (shift != 0) {
        helper.gammaIndicesComm.push_back(i);

        // Assume 1-link for now -- break loop when you find a shift direction
        for (int i = 0; i < StagGamma::gmu.size(); i++) {
          if (StagGamma::gmu[i] & shift) {
            helper.shiftDirs.push_back(i);
            helper.shiftDisplacements.push_back(1);
            break;
          }
        }
      } else {
        helper.gammaIndicesLocal.push_back(i);
      }
    }

    if (helper.gammaIndicesLocal.size() > MF_SUM_ARRAY_SIZE || helper.gammaIndicesComm.size() > MF_SUM_ARRAY_SIZE) {
      std::cout << GridLogError << "Parameter space too large: Need num Momenta * num Gammas < " << MF_SUM_ARRAY_SIZE << "." << std::endl;
      assert(0);
    }

    // Grab SIMD coordinates from indices
    helper.iCoorContainer.resize(grid->Nsimd(), Coordinate(grid->_ndimension));
    for(int p = 0; p < grid->Nsimd(); p++) {
      grid->iCoorFromIindex(helper.iCoorContainer[p],p);
    }

    // Map checkerboarded indices to full lattice indices
    if (helper.checkerR || helper.checkerL) {
      auto cbGrid = helper.grid;

      assert(cbGrid->CheckerBoarded(helper.orthogDir) != 1);

      helper.oCoordsEven.resize(cbGrid->oSites(),0);
      helper.oCoordsOdd.resize(cbGrid->oSites(),0);

      // Create map between checkerboarded indices and full lattice indices
      thread_for(so,grid->oSites(),{
        int oSiteCheckerboard;
        Coordinate coor;

        grid->oCoorFromOindex(coor,so);
        oSiteCheckerboard=cbGrid->CheckerBoard(coor);
          
        int cbSite=cbGrid->oIndex(coor);

        if (oSiteCheckerboard == Even)
          helper.oCoordsEven[cbSite]=so;
        else
          helper.oCoordsOdd[cbSite]=so;
      });
    }
  }

  // Setup lists of pointers to share with accelerators
  template <class Vtype>
  static void makeView(Vector<LatticeView<Vtype> >& view, const Lattice<Vtype> *field, int size)
  {
    if (view.size() != 0)
      return;

    view.reserve(size);

    for(int p=0;p<size;p++) {
      view.push_back(field[p].View(AcceleratorRead));
    }
  }

  // Setup lists of pointers to share with accelerators
  static void makeView(Vector<GaugeView>& view, std::vector<LatticeColourMatrix> &Umu, 
                  const LatticeGaugeField *U, std::vector<int> shift_dirs)
  {
    int size = shift_dirs.size();
    if (view.size() != 0)
      return;

    for(int p=0;p<size;p++) {
      Umu[p] = PeekIndex<LorentzIndex>(*U,shift_dirs[p]); // Store full lattice links in shift direction
    }
    makeView(view, &Umu[0], size);
  }

  // Setup lists of pointers to share with accelerators
  static void makeStencilView(Vector<FermStencilView> &view, std::vector<std::unique_ptr<FermStencil> > &stencil, 
                        const FermionField* field, int size)
  {
    if (view.size() != 0)
      return;

    GridBase *grid = field[0].Grid();
    SimpleCompressor<vobj> compressor;

    view.reserve(size);

    for(int p=0;p<size;p++) {
      stencil[p]->HaloExchange(field[p],compressor);
      view.push_back(stencil[p]->View(AcceleratorRead));
    }
  }
};
template <class FImpl>
template <typename TensorType>
void A2AutilsMILC<FImpl>::StagMesonFieldNoGlobalSum(TensorType &mat,
                                     const FermionField *lhs_wi_e,
                                     const FermionField *lhs_wi_o,
                                     const FermionField *rhs_vj_e,
                                     const FermionField *rhs_vj_o,
                                     const std::vector<StagGamma::SpinTastePair>& gammas,
                                     const std::vector<ComplexField > &mom,
                                     int orthogDir, 
                                     LatticeGaugeField* U, double *t_kernel)
{
  if (t_kernel) *t_kernel = -usecond();

  utilHelper helper;

  helper.sizeL = mat.dimension(3);
  helper.sizeR = mat.dimension(4);
  helper.sizeLOut = mat.dimension(3);
  helper.sizeROut = mat.dimension(4);
  helper.nMom   = mom.size();
  helper.orthogDir = orthogDir;

  GridBase *grid  = mom[0].Grid();

  helper.grid = grid;
  GridBase *cbGrid = nullptr;

  helper.checkerL = lhs_wi_e[0].Grid()->_isCheckerBoarded;
  helper.checkerR = rhs_vj_e[0].Grid()->_isCheckerBoarded;

  if (helper.checkerL) {
    helper.sizeL /= 2;
    cbGrid = lhs_wi_e[0].Grid();
    helper.grid = cbGrid;
  } 
  if(helper.checkerR) {
    helper.sizeR /= 2;
    cbGrid = rhs_vj_e[0].Grid();
    helper.grid = cbGrid;
  }

  init(gammas,U,grid, helper);

  int nGamma       = gammas.size();
  int nGamma_comm  = helper.gammaIndicesComm.size();
  int nGamma_local = helper.gammaIndicesLocal.size();

  std::vector<ComplexField> stagPhase(nGamma,grid);

  { // Set up staggered phases
    StagGamma spinTaste;
    for (int mu = 0; mu < nGamma; mu++) {
      stagPhase[mu] = 1.0;
      spinTaste.setSpinTaste(gammas[mu]);
      spinTaste.applyPhase(stagPhase[mu],stagPhase[mu]); // store spin-taste phase
    }
  }

  Vector<Scalar_v> simd_sum_comm_e, simd_sum_comm_o;
  Vector<Scalar_v> simd_sum_local_e, simd_sum_local_o;

  makeView(helper.viewGamma,&stagPhase[0],nGamma);
  makeView(helper.viewMom,  &mom[0],helper.nMom);

  makeView(helper.viewLeft_e,  lhs_wi_e, helper.sizeL);
  makeView(helper.viewRight_e, rhs_vj_e, helper.sizeR);
  if (helper.checkerL)
    makeView(helper.viewLeft_o,  lhs_wi_o, helper.sizeL);
  if (helper.checkerR)
    makeView(helper.viewRight_o, rhs_vj_o, helper.sizeR);

  auto lgrid = helper.checkerL?cbGrid:grid;
  auto rgrid = helper.checkerR?cbGrid:grid;

  std::vector<std::unique_ptr<FermStencil> > stencilLeft_e(helper.sizeL);
  std::vector<std::unique_ptr<FermStencil> > stencilRight_e(helper.sizeR);
  std::vector<std::unique_ptr<FermStencil> > stencilLeft_o(helper.checkerL ? helper.sizeL : 0);
  std::vector<std::unique_ptr<FermStencil> > stencilRight_o(helper.checkerR ? helper.sizeR : 0);

  std::vector<LatticeColourMatrix> Umu(nGamma_comm,grid);

  // Run any nonlocal gamma operators
  if (nGamma_comm > 0) {
    for (auto &item : stencilLeft_e) {
      item = std::move(std::unique_ptr<FermStencil>(new FermStencil(lgrid, nGamma_comm, Even, helper.shiftDirs, helper.shiftDisplacements, 0)));
    }
    for (auto &item : stencilRight_e) {
      item = std::move(std::unique_ptr<FermStencil>(new FermStencil(rgrid, nGamma_comm, Even, helper.shiftDirs, helper.shiftDisplacements, 0)));
    }
    for (auto &item : stencilLeft_o) {
      item = std::move(std::unique_ptr<FermStencil>(new FermStencil(lgrid, nGamma_comm, Odd, helper.shiftDirs, helper.shiftDisplacements, 0)));
    }
    for (auto &item : stencilRight_o) {
      item = std::move(std::unique_ptr<FermStencil>(new FermStencil(rgrid, nGamma_comm, Odd, helper.shiftDirs, helper.shiftDisplacements, 0)));
    }

    makeStencilView(helper.viewStencilLeft_e,  stencilLeft_e,  lhs_wi_e, helper.sizeL);
    makeStencilView(helper.viewStencilRight_e, stencilRight_e, rhs_vj_e, helper.sizeR);
    if (helper.checkerL)
      makeStencilView(helper.viewStencilLeft_o,  stencilLeft_o,  lhs_wi_o, helper.sizeL);
    if (helper.checkerR)
      makeStencilView(helper.viewStencilRight_o, stencilRight_o, rhs_vj_o, helper.sizeR);

    makeView(helper.viewGauge, Umu, U,helper.shiftDirs);


    if (helper.checkerL && helper.checkerR) {
      spatialContractComm(simd_sum_comm_e, Even, Odd, helper);
      spatialContractComm(simd_sum_comm_o, Odd,  Even, helper);
    } else if (helper.checkerL) {
      spatialContractComm(simd_sum_comm_e, Even, Even, helper);
      spatialContractComm(simd_sum_comm_o, Odd,  Even, helper);
    } else if (helper.checkerR) {
      spatialContractComm(simd_sum_comm_e, Even, Odd, helper);
      spatialContractComm(simd_sum_comm_o, Even, Even, helper);
    } else {
      spatialContractComm(simd_sum_comm_e, Even, Even, helper);
    }
  }

  // Run any local gamma operators
  if (nGamma_local > 0) {
    if (helper.checkerL && helper.checkerR) {
      spatialContractLocal(simd_sum_local_e, Even, Even, helper);
      spatialContractLocal(simd_sum_local_o, Odd,  Odd, helper);
    } else if (helper.checkerL) {
      spatialContractLocal(simd_sum_local_e, Even, Even, helper);
      spatialContractLocal(simd_sum_local_o, Odd,  Even, helper);
    } else if (helper.checkerR) {
      spatialContractLocal(simd_sum_local_e, Even, Even, helper);
      spatialContractLocal(simd_sum_local_o, Even, Odd, helper);
    } else {
      spatialContractLocal(simd_sum_local_e, Even, Even, helper);
    }
  }

  accelerator_barrier();

  // Contract SIMD vectors
  if (helper.checkerL || helper.checkerR) {
    contractSimd(mat, true, simd_sum_comm_e, simd_sum_comm_o, helper);

    contractSimd(mat, false, simd_sum_local_e, simd_sum_local_o, helper);
  } else {
    contractSimd(mat, simd_sum_comm_e, helper.gammaIndicesComm, helper);

    contractSimd(mat, simd_sum_local_e, helper.gammaIndicesLocal, helper);
  }

  accelerator_barrier();

  if (t_kernel) *t_kernel += usecond();

  for(int p=0;p<helper.viewLeft_e.size();p++)  helper.viewLeft_e[p].ViewClose();
  for(int p=0;p<helper.viewRight_e.size();p++) helper.viewRight_e[p].ViewClose();
  for(int p=0;p<helper.viewLeft_o.size();p++)  helper.viewLeft_o[p].ViewClose();
  for(int p=0;p<helper.viewRight_o.size();p++) helper.viewRight_o[p].ViewClose();
  for(int p=0;p<helper.viewStencilLeft_e.size();p++)  helper.viewStencilLeft_e[p].ViewClose();
  for(int p=0;p<helper.viewStencilRight_e.size();p++) helper.viewStencilRight_e[p].ViewClose();
  for(int p=0;p<helper.viewStencilLeft_o.size();p++)  helper.viewStencilLeft_o[p].ViewClose();
  for(int p=0;p<helper.viewStencilRight_o.size();p++) helper.viewStencilRight_o[p].ViewClose();
  for(int p=0;p<helper.viewGamma.size();p++)   helper.viewGamma[p].ViewClose();
  for(int p=0;p<helper.viewGauge.size();p++)   helper.viewGauge[p].ViewClose();
  for(int p=0;p<helper.viewMom.size();p++)     helper.viewMom[p].ViewClose();

}

template <class FImpl>
template <typename TensorType>
void A2AutilsMILC<FImpl>::contractSimd(TensorType &result, bool do_comm, Vector<Scalar_v> &simd_sum_e, 
                                        Vector<Scalar_v> &simd_sum_o, utilHelper &helper) {

  if (simd_sum_e.size() == 0)
    return;

  if (simd_sum_o.size() == 0)
    return;

  auto grid = helper.grid;
  int sizeL = helper.sizeL;
  int sizeR = helper.sizeR;
  int sizeLOut = helper.sizeLOut;
  int sizeROut = helper.sizeROut;
  int nMom  = helper.nMom;
  int orthogDir = helper.orthogDir;
  int checkerL  = helper.checkerL;
  int checkerR  = helper.checkerR;

  const int simdSize       = grid->Nsimd();

  int mult_l = checkerL ? 2 : 1;
  int mult_r = checkerR ? 2 : 1;

  int reducedOrthogDimSize = grid->_rdimensions[orthogDir];
  int localOrthogDimSize   = grid->_ldimensions[orthogDir];

  int Nt     = grid->GlobalDimensions()[orthogDir];
  int nGamma, nGammaTotal = helper.viewGamma.size();

  Scalar_v   *simd_sum_e_p = & simd_sum_e[0];
  Scalar_v   *simd_sum_o_p = & simd_sum_o[0];
  Coordinate *icoor_p        = & helper.iCoorContainer[0];
  Integer    *indexG_p;

  if (do_comm) {
    nGamma = helper.gammaIndicesComm.size();
    indexG_p = & helper.gammaIndicesComm[0]; 
  } else {
    nGamma = helper.gammaIndicesLocal.size();
    indexG_p = & helper.gammaIndicesLocal[0]; 
  }

  int pd = grid->_processors[orthogDir];
  int pc = grid->_processor_coor[orthogDir];

  auto result_p = result.data();
  for (int mu = 0; mu < nGamma; mu++)
  for ( int m=0;m<nMom;m++){
    accelerator_forNB(l_index,sizeL,1,{
      // Sum across simd lanes in the plane, breaking out orthog dir.
      ExtractBuffer<Scalar_s> extracted_e(simdSize),extracted_o(simdSize);

      for(int r_index=0;r_index<sizeR;r_index++){

        for (int rt=0;rt<reducedOrthogDimSize;rt++) {

          int base = nGamma*nMom*l_index+nGamma*nMom*sizeL*r_index+nGamma*nMom*sizeL*sizeR*rt;
              
              // Final matrix layout is mat[nMom,nGamma,orthogDimSize,Nw,Nv]
              // Thus, ignoring gamma and time we get the index of simd_sum
              int ij_rdx = base+mu*nMom+m;
              
              extract(simd_sum_e_p[ij_rdx],extracted_e);
              extract(simd_sum_o_p[ij_rdx],extracted_o);
              
              for(int idx=0;idx<simdSize;idx++){
                  
                  // Calculate local time slice
                  int lt = rt+icoor_p[idx][orthogDir]*reducedOrthogDimSize;
                  // Calculate global time slice
                  int t = lt + pc*localOrthogDimSize;

                  int ij_dx = mult_r*( r_index + sizeR*mult_l*( l_index + sizeL*( t + Nt*( indexG_p[mu] + nGammaTotal*m ) ) ) );

                  // Fix Me: for > 1-link shift operations, this needs to be "if even odd shifts", not do_comm
                  result_p[ij_dx]            = result_p[ij_dx] + extracted_e[idx];
                  result_p[ij_dx]            = result_p[ij_dx] + extracted_o[idx];
                  result_p[ij_dx+sizeROut]   = result_p[ij_dx+sizeROut] + extracted_e[idx];
                  result_p[ij_dx+sizeROut]   = result_p[ij_dx+sizeROut] - extracted_o[idx];
                  if (do_comm) {
                    result_p[ij_dx+1]          = result_p[ij_dx+1] + extracted_o[idx];
                    result_p[ij_dx+1]          = result_p[ij_dx+1] - extracted_e[idx];
                    result_p[ij_dx+sizeROut+1] = result_p[ij_dx+sizeROut+1] - extracted_e[idx];
                    result_p[ij_dx+sizeROut+1] = result_p[ij_dx+sizeROut+1] - extracted_o[idx];
                  } else {
                    result_p[ij_dx+1]          = result_p[ij_dx+1] + extracted_e[idx];
                    result_p[ij_dx+1]          = result_p[ij_dx+1] - extracted_o[idx];
                    result_p[ij_dx+sizeROut+1] = result_p[ij_dx+sizeROut+1] + extracted_e[idx];
                    result_p[ij_dx+sizeROut+1] = result_p[ij_dx+sizeROut+1] + extracted_o[idx];
                  }
              }
            }
          }
      });
  }
}

template <class FImpl>
template <typename TensorType>
void A2AutilsMILC<FImpl>::contractSimd(TensorType &result,Vector<Scalar_v> &simd_sum, Vector<Integer> &gamma_indices, utilHelper &helper) {

  if (simd_sum.size() == 0)
    return;

  auto grid = helper.grid;
  int sizeL = helper.sizeL;
  int sizeR = helper.sizeR;
  int nMom  = helper.nMom;
  int orthogDir = helper.orthogDir;

  const int simdSize       = grid->Nsimd();

  int reducedOrthogDimSize = grid->_rdimensions[orthogDir];
  int localOrthogDimSize   = grid->_ldimensions[orthogDir];

  int Nt     = grid->GlobalDimensions()[orthogDir];
  int nGamma = gamma_indices.size();
  int nGammaTotal = helper.viewGamma.size();

  Scalar_v   *simd_sum_p = & simd_sum[0];
  Coordinate *icoor_p        = & helper.iCoorContainer[0];
  Integer    *indexG_p = & gamma_indices[0];

  int pd = grid->_processors[orthogDir];
  int pc = grid->_processor_coor[orthogDir];

  auto result_p = result.data();
  for (int mu = 0; mu < nGamma; mu++)
  for ( int m=0;m<nMom;m++){
    accelerator_forNB(l_index,sizeL,1,{
      // Sum across simd lanes in the plane, breaking out orthog dir.
      ExtractBuffer<Scalar_s> extracted(simdSize);

      for(int r_index=0;r_index<sizeR;r_index++){

        for (int rt=0;rt<reducedOrthogDimSize;rt++) {

          int base = nGamma*nMom*l_index+nGamma*nMom*sizeL*r_index+nGamma*nMom*sizeL*sizeR*rt;
              
              // Final matrix layout is mat[nMom,nGamma,orthogDimSize,Nw,Nv]
              int ij_rdx = base+mu*nMom+m;
              
              extract(simd_sum_p[ij_rdx],extracted);
              
              for(int idx=0;idx<simdSize;idx++){
                  
                  int lt = rt+icoor_p[idx][orthogDir]*reducedOrthogDimSize;
                  int t = lt + pc*localOrthogDimSize;

                  int ij_dx = r_index + sizeR*l_index + sizeR*sizeL*t + sizeR*sizeL*Nt*indexG_p[mu] + sizeR*sizeL*Nt*nGammaTotal*m;

                  result_p[ij_dx]=result_p[ij_dx]+extracted[idx];
              }
            }
          }
      });
  }
}

template <class FImpl>
void A2AutilsMILC<FImpl>::spatialContractComm(Vector<Scalar_v>& result, int checkerboardL, int checkerboardR, utilHelper &helper)
{
  Vector<FermView> &viewLeft  = helper.getView(true,  checkerboardL);
  Vector<FermView> &viewRight = helper.getView(false, checkerboardR);

  Vector<FermStencilView> &viewStencilLeft  = helper.getStencilView(true,  checkerboardL);
  Vector<FermStencilView> &viewStencilRight = helper.getStencilView(false, checkerboardR);

  auto grid = helper.grid;
  int sizeL = helper.sizeL;
  int sizeR = helper.sizeR;
  int nMom  = helper.nMom;
  int orthogDir = helper.orthogDir;
  int checkerL  = helper.checkerL;
  int checkerR  = helper.checkerR;

  const int simdSize     = grid->Nsimd();

  // Help in iterating over elements in the same time slice
  const int nBlocks      = grid->_slice_nblock[orthogDir];
  const int blockStride  = grid->_slice_stride[orthogDir];
  const int rtStride     = grid->_ostride[orthogDir];
  const int vecsPerSlicePerBlock = grid->_slice_block[orthogDir];

  // Output matrix dimensions
  const int reducedOrthogDimSize = grid->_rdimensions[orthogDir];
  const int nGamma = helper.gammaIndicesComm.size();

  const int MFrvol = reducedOrthogDimSize *sizeL *sizeR *nMom * nGamma;

  result.resize(MFrvol);

  Scalar_v        *result_p = & result[0];     // Return object
  accelerator_for(r, MFrvol,1,{
    result_p[r] = Zero();
  });  

  // Pass data to the device
  FermView        *view_pL = & viewLeft[0];    // bra vectors
  FermView        *view_pR = & viewRight[0];   // ket vectors
  ComplexView     *view_pG = & helper.viewGamma[0];   // Gammas
  ComplexView     *view_pM = & helper.viewMom[0];     // Momenta
  GaugeView       *view_pU = & helper.viewGauge[0];   // Gauge links

  FermStencilView  *rightStencil = & viewStencilRight[0]; // Stencil for shifted kets
  FermStencilView  *leftStencil = & viewStencilLeft[0]; // Stencil for shifted bras

  Integer *oCoordsL_p= & helper.oCoordsEven[0]; // maps checkerboarded indices to corresponding full grid index
  Integer *oCoordsR_p= & helper.oCoordsEven[0]; 

  Integer *indexG_p= & helper.gammaIndicesComm[0]; 

  if (checkerL && checkerboardL == Odd) {
      oCoordsL_p = & helper.oCoordsOdd[0];
  } else if (checkerR && checkerboardR == Odd) {
      oCoordsR_p = & helper.oCoordsOdd[0];
  }

  accelerator_for2dNB_no_err(l_index,sizeL,r_index,sizeR,simdSize,{

    calcColourMatrix link;
    calcSpinor left, right, shift_ahead;
    calcScalar temp_site, gamma_phase, momentum_phase, sum[MF_SUM_ARRAY_SIZE];
    StencilEntry *SE;
    int ptype, so, base, sumIndex;

    for (int rt=0;rt<reducedOrthogDimSize;rt++) {

      so = rt*rtStride; // base offset for start of the local plane
      base = nGamma*nMom*(sizeL*sizeR*rt + sizeL*r_index + l_index);

      for (int p = 0; p < nGamma*nMom; p++) {
        sum[p] = 0.0;
      }
 
      for(int n=0;n<nBlocks;n++)
      for(int b=0;b<vecsPerSlicePerBlock;b++){

        int ss = so+n*blockStride+b;
        int ssL, ssR, fullssL, fullssR;

        if (checkerL) {
          fullssL = oCoordsL_p[ss];
          ssL = ss;
        } else {
          ssL = oCoordsR_p[ss];
          fullssL = ssL;
        }
        acceleratorSynchronise();

        if (checkerR) {
          fullssR = oCoordsR_p[ss];
          ssR = ss;
        } else {
          ssR = oCoordsL_p[ss];
          fullssR = ssR;
        }
        acceleratorSynchronise();

        left   = coalescedRead(view_pL[l_index][ssL]);
        right  = coalescedRead(view_pR[r_index][ssR]);

        sumIndex = 0;
        for (int mu = 0; mu < nGamma; mu++) {
          gamma_phase = coalescedRead(view_pG[indexG_p[mu]][fullssL]);
          link = coalescedRead(view_pU[mu][fullssL]);

          // Grab field from ahead of ssL
          SE=rightStencil[r_index].GetEntry(ptype,mu,ssL);
          if(SE->_is_local) { 
            shift_ahead = coalescedReadPermute(view_pR[r_index][SE->_offset],ptype,SE->_permute);
          } else {
            shift_ahead = coalescedRead(rightStencil[r_index].CommBuf()[SE->_offset]);
          }
          acceleratorSynchronise();

          shift_ahead =  link * shift_ahead;
          temp_site = innerProduct(left,shift_ahead);

          for ( int m=0;m<nMom;m++) {
            momentum_phase = coalescedRead(view_pM[m][fullssL]);
            sum[sumIndex+m]  += gamma_phase*momentum_phase*temp_site;
          }

          gamma_phase = coalescedRead(view_pG[indexG_p[mu]][fullssR]);
          link = coalescedRead(view_pU[mu][fullssR]);

          // Grab field from ahead of ssL
          SE=leftStencil[l_index].GetEntry(ptype,mu,ssR);
          if(SE->_is_local) { 
            shift_ahead = coalescedReadPermute(view_pL[l_index][SE->_offset],ptype,SE->_permute);
          } else {
            shift_ahead = coalescedRead(leftStencil[l_index].CommBuf()[SE->_offset]);
          }
          acceleratorSynchronise();

          shift_ahead =  link * shift_ahead;
          temp_site = innerProduct(shift_ahead,right);

          for ( int m=0;m<nMom;m++) {
            momentum_phase = coalescedRead(view_pM[m][fullssR]);
            sum[sumIndex+m]  += gamma_phase*momentum_phase*temp_site;
          }

          sumIndex+=nMom;
        }
      }
      for (int p = 0; p < nMom*nGamma; p++) {
        int idx = base+p;
        coalescedWrite(result_p[idx],sum[p]);
      }
    }
  });
}

template <class FImpl>
void A2AutilsMILC<FImpl>::spatialContractLocal(Vector<Scalar_v>& result, int checkerboardL, int checkerboardR, utilHelper &helper)
{
  Vector<FermView> &viewLeft  = helper.getView(true,  checkerboardL);
  Vector<FermView> &viewRight = helper.getView(false, checkerboardR);

  auto grid = helper.grid;
  int sizeL = helper.sizeL;
  int sizeR = helper.sizeR;
  int orthogDir = helper.orthogDir;
  int checkerL  = helper.checkerL;
  int checkerR  = helper.checkerR;
  const int nMom  = helper.nMom;

  const int simdSize             = grid->Nsimd();
  const int reducedOrthogDimSize = grid->_rdimensions[orthogDir];

  const int nBlocks      = grid->_slice_nblock[orthogDir];
  const int blockStride  = grid->_slice_stride[orthogDir];
  const int rtStride     = grid->_ostride[orthogDir];
  const int vecsPerSlicePerBlock = grid->_slice_block[orthogDir];

  const int nGamma = helper.gammaIndicesLocal.size();

  const int MFrvol = reducedOrthogDimSize * sizeL * sizeR * nMom * nGamma;

  result.resize(MFrvol);

  Scalar_v     *result_p   = & result[0];
  accelerator_for(r, MFrvol,1,{
    result_p[r] = Zero();
  });  

  // Pointers for accelerator indexing
  FermView     *view_pL    = &viewLeft[0];
  FermView     *view_pR    = &viewRight[0];

  ComplexView  *view_pG    = & helper.viewGamma[0];
  ComplexView  *view_pM    = & helper.viewMom[0];

  Integer      *indexG_p= & helper.gammaIndicesLocal[0]; 
  Integer      *oCoords_p;


  // If needed, Grab appropriate mapping between checkerboarded lattice and full lattice
  if(checkerL || checkerR) {
    if ((checkerL && checkerboardL == Odd)
        || (checkerR && checkerboardR == Odd)) {
      oCoords_p = & helper.oCoordsOdd[0];
    } else {
      oCoords_p = & helper.oCoordsEven[0];
    }
  }

  accelerator_for2dNB_no_err(l_index,sizeL,r_index,sizeR,simdSize,{

    int so, ss, base, sumIndex, ssL, ssR, fullss;
    calcSpinor left, right;
    calcScalar temp_site, gamma_phase, momentum_phase, sum[MF_SUM_ARRAY_SIZE];

    for (int rt=0;rt<reducedOrthogDimSize;rt++) {

      so=rt*rtStride; // base offset for start of the local plane
      base = nGamma*nMom*l_index+nGamma*nMom*sizeL*r_index+nGamma*nMom*sizeL*sizeR*rt;

      for (int p = 0; p < nGamma*nMom; p++) {
        sum[p] = 0.0;
      }

      // Loop through all points on the same time slice
      for(int n=0;n<nBlocks;n++)
      for(int b=0;b<vecsPerSlicePerBlock;b++){

          ss = so+n*blockStride+b;
          ssL = ssR = fullss = ss;

          if (checkerL || checkerR) {
            fullss = oCoords_p[ss];
            ssL = fullss;
            ssR = fullss;

            if (checkerL)
              ssL = ss;
            if (checkerR)
              ssR = ss;
          }
          acceleratorSynchronise();

          left  = coalescedRead(view_pL[l_index][ssL]);
          right = coalescedRead(view_pR[r_index][ssR]);
      
          temp_site  = innerProduct(left,right);

          sumIndex = 0;
          for (int mu = 0; mu < nGamma; mu++) {
            gamma_phase = coalescedRead(view_pG[indexG_p[mu]][fullss]);
            for ( int m=0;m<nMom;m++) {
              momentum_phase = coalescedRead(view_pM[m][fullss]);
              
              sum[sumIndex] += gamma_phase*momentum_phase*temp_site;
              sumIndex++;
            }
          }
        }
      for (int p = 0; p < nGamma*nMom; p++) {
        int idx = base+p;
        coalescedWrite(result_p[idx],sum[p]);
      }
    }
  });
}

NAMESPACE_END(Grid);

#undef accelerator_for2dNB_no_err
