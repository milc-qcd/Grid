#pragma once
#include <Grid/Grid_Eigen_Tensor.h>

#ifndef MF_SUM_ARRAY_MAX
#define MF_SUM_ARRAY_MAX 16
#endif

NAMESPACE_BEGIN(Grid);

template <typename FImpl>
class A2AWorkerMILC 
{
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

  typedef typename FImpl::SiteSpinor vobj;

  typedef typename FImpl::ImplParams FImplParams;

  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  typedef iSinglet<vector_type> Scalar_v;
  typedef iSinglet<scalar_type> Scalar_s;

  typedef decltype(coalescedRead(Scalar_v())) calcScalar;
  typedef decltype(coalescedRead(vobj())) calcSpinor;
  typedef decltype(coalescedRead(vColourMatrix())) calcColourMatrix;
  
  typedef LatticeView<vobj> FermView;
  typedef typename FImpl::StencilImpl FermStencil;
  typedef typename FImpl::StencilView FermStencilView;

  typedef LatticeView<vColourMatrix> GaugeView;
  typedef CartesianStencil<vColourMatrix,vColourMatrix,FImplParams> GaugeStencil;
  typedef CartesianStencilView<vColourMatrix,vColourMatrix,FImplParams> GaugeStencilView;

  typedef typename ComplexField::vector_object cobj;
  typedef LatticeView<cobj> ComplexView;

public:
  A2AWorkerMILC() = delete;
  A2AWorkerMILC(GridBase *grid, const std::vector<StagGamma::SpinTastePair>& gammas, const std::vector<ComplexField> &mom, LatticeGaugeField* U = nullptr, GridBase *cbGrid = nullptr)
  : _grid(grid), _gammas(gammas), _mom(mom), _U(U), _cb_grid(cbGrid) {

    StagGamma spinTaste;
    if (_U != nullptr) {
      spinTaste.setGaugeField(*_U);
    }

    // Organize gammas into local/non-local
    for (int i = 0; i < _gammas.size(); i++) {
  
      spinTaste.setSpinTaste(_gammas[i]);

      int shift = (spinTaste._spin ^ spinTaste._taste);
      if (shift != 0) {

        assert(_U != nullptr);

        _gamma_indices_comm.push_back(i);

        // Assume 1-link for now -- break loop when you find a shift direction
        for (int j = 0; j < StagGamma::gmu.size(); j++) {
          if (StagGamma::gmu[j] & shift) {
            _shift_dirs.push_back(j);
            _shift_dirs.push_back(j);
            _shift_displacements.push_back(1);
            _shift_displacements.push_back(-1);
            break;
          }
        }
      } else {
        _gamma_indices_local.push_back(i);
      }
    }

    if (_gamma_indices_local.size() > MF_SUM_ARRAY_MAX || _gamma_indices_comm.size() > MF_SUM_ARRAY_MAX) {
      std::cout << GridLogError << "Parameter space too large: Need num Momenta * num Gammas < " << MF_SUM_ARRAY_MAX << "." << std::endl;
      assert(0);
    }

    // Grab SIMD coordinates from indices
    _i_coor_container.resize(grid->Nsimd(), Coordinate(grid->_ndimension));
    for(int p = 0; p < grid->Nsimd(); p++) {
      grid->iCoorFromIindex(_i_coor_container[p],p);
    }

    // Map checkerboarded indices to full lattice indices
    if (cbGrid != nullptr) {
      _o_coords_E.resize(cbGrid->oSites(),0);
      _o_coords_O.resize(cbGrid->oSites(),0);

      // Create map between checkerboarded indices and full lattice indices
      thread_for(so,grid->oSites(),{
        int oSiteCheckerboard;
        Coordinate coor;

        grid->oCoorFromOindex(coor,so);
        oSiteCheckerboard=cbGrid->CheckerBoard(coor);
          
        int cbSite=cbGrid->oIndex(coor);

        if (oSiteCheckerboard == Even)
          _o_coords_E[cbSite]=so;
        else
          _o_coords_O[cbSite]=so;
      });
    }
  }

  double getFlops() {
    // One complex multiply takes 6 floating point ops (4 mult, 2 add) 
    // --> complex inner product is 3 complex mult, 2 complex add = 3*6 + 2*2 = 22 double precision floating ops

    // For each vector and at each lattice site:
    //  - one inner product
    //  - For each gamma and momentum
    //    - multiply by gamma phase
    //    - multiply by momentum phase
    //    - sum
    double local_flops = 0.0;

    if (_gamma_indices_local.size() > 0) {
      local_flops = 22.0+(6.0+6.0+2.0)*(_N_mom*_gamma_indices_local.size());
    }

    // matrix*vector = 3 inner products
    // current code: innerProduct(left,link_ahead*shift_ahead+adj(link_behind)*shift_behind)
    //  = matrix*vector + matrix* vector --> inner product = 7 inner products and 1 complex sum

    // For each vector, each gamma, and at each lattice site:
    //  - one inner product
    //  - two su(3) matrix*vector ops
    //  - one sum
    //  - For each momentum
    //    - one multiply
    //    - one sum
    double one_link_flops = ((7*22.0+2.0) + _N_mom*8.0)*_gamma_indices_comm.size();

    return local_flops + one_link_flops;
  }

  const FermionField* getRight() {
    if (_checker_R && _checkerboard_R == Odd) {
      return _right_O;
    } else {
      return _right_E;
    }
  }

  const FermionField* getLeft() {
    if (_checker_L && _checkerboard_L == Odd) {
      return _left_O;
    } else {
      return _left_E;
    }
  }

  Vector<Integer> &getRightCoords() {
    if (_checker_R && _checkerboard_R == Odd) {
      return _o_coords_O;
    } else {
      return _o_coords_E;
    }
  }

  Vector<Integer> &getLeftCoords() {
    if (_checker_L && _checkerboard_L == Odd) {
      return _o_coords_O;
    } else {
      return _o_coords_E;
    }
  }

  // Setup lists of pointers to share with accelerators
  template <class Vtype>
  void makeView(Vector<LatticeView<Vtype> >& view, const Lattice<Vtype> *field,int size)
  {
    if (view.size() != 0 || size == 0)
      return;

    view.reserve(size);

    for(int p=0;p<size;p++) {
      view.push_back(field[p].View(AcceleratorRead));
    }
  }

  void makeGammaView(const ComplexField *field) { makeView(_view_gamma,field,_gamma_indices_local.size()); }

  void makeMomentumView(const ComplexField *field) { makeView(_view_mom,field,_N_mom); }

  void makeLeftView(Vector<FermView> &view) {//const FermionField *field, int checkerboard) { 
    if (_checkerboard_L == Odd)
      makeView(view,_left_O,_size_L); 
    else
      makeView(view,_left_E,_size_L); 
  }

  void makeRightView(Vector<FermView> &view) {//const FermionField *field, int checkerboard) { 
    if (_checkerboard_R == Odd)
      makeView(view,_right_O,_size_R); 
    else
      makeView(view,_right_E,_size_R); 
  }

  void makeGauge(std::vector<LatticeColourMatrix> &Umu, int checkerboard = -1)
  {
    StagGamma spinTaste;
    LatticeColourMatrix Umu_temp(_U->Grid());
    int mu, size = _gamma_indices_comm.size();

    for (int i = 0; i < size; i++) {
      mu = _gamma_indices_comm[i];
      spinTaste.setSpinTaste(_gammas[mu]);

      if (checkerboard != -1 ) {
        Umu_temp = PeekIndex<LorentzIndex>(*_U,_shift_dirs[2*i]); // Store full lattice links in shift direction

        spinTaste.applyPhase(Umu_temp,Umu_temp); // store spin-taste phase

        pickCheckerboard(checkerboard,Umu[i],Umu_temp);
      } else {
        Umu[i] = PeekIndex<LorentzIndex>(*_U,_shift_dirs[2*i]);
        spinTaste.applyPhase(Umu[i],Umu[i]); // store spin-taste phase
      }

    }
  }

  void makeGaugeLeft()
  {
    _Umu_L.resize(_gamma_indices_comm.size(),_grid_L);

    if (_checker_L) {
      if (_checkerboard_L == Odd) {
        makeGauge(_Umu_L,Odd);
      } else {
        makeGauge(_Umu_L,Even);
      }
    } else {
      makeGauge(_Umu_L);
    }
  }

  void makeGaugeRight()
  {
    _Umu_R.resize(_gamma_indices_comm.size(),_grid_R);

    if (_checker_R) {
      if (_checkerboard_R == Odd) {
        makeGauge(_Umu_R,Odd);
      } else {
        makeGauge(_Umu_R,Even);
      }
    } else {
      makeGauge(_Umu_R);
    }
  }

  void makeGaugeViewLeft(Vector<GaugeView> &view)
  {
    makeView(view, &_Umu_L[0],_gamma_indices_comm.size());
  }

  void makeGaugeViewRight(Vector<GaugeView> &view)
  {
    makeView(view, &_Umu_R[0],_gamma_indices_comm.size());
  }

  // Setup lists of pointers to share with accelerators
  template <class Vtype>
  void makeStencil(Vector<Vtype> &buffer, std::unique_ptr<CartesianStencil<Vtype,Vtype,FImplParams> > &stencil, 
                        const Lattice<Vtype>* field, int size)
  {
    GridBase *grid = field[0].Grid();
    SimpleCompressor<Vtype> compressor;

    int comm_buf_size;
    Vtype *buf_p;

    for(int p=0;p<size;p++) {

      stencil->HaloExchange(field[p],compressor);

      if (p == 0) {
        comm_buf_size = stencil->_unified_buffer_size;
        buffer.resize(size*comm_buf_size);
        buf_p = &buffer[0];
      }
      if (comm_buf_size > 0) {
        Vtype *comm_buf_p = stencil->CommBuf();
        accelerator_for(i,comm_buf_size,1,{
          buf_p[p*comm_buf_size+i] = comm_buf_p[i];
        });
      }
    }
  }

  void makeGaugeStencil(Vector<vColourMatrix> &buffer, std::vector<std::unique_ptr<GaugeStencil> > &stencil, 
                        Vector<GaugeStencilView> &stencilView) {

    Vector<vColourMatrix> bufMu;
    int size = _shift_dirs.size()/2;

    _gauge_stencil_buf_offsets.resize(size);

    stencilView.reserve(size);
    for (int i=0;i<size;i++) {
      stencil[i] = std::move(std::unique_ptr<GaugeStencil>(new GaugeStencil(_grid_R, 1, _checkerboard_R, {_shift_dirs[2*i]}, {-1})));

      makeStencil(bufMu,stencil[i],&_Umu_R[i],1);

      _gauge_stencil_buf_offsets[i] = buffer.size();

      buffer.resize(buffer.size()+bufMu.size());

      if (bufMu.size() > 0) {
        vColourMatrix *buf_p = &buffer[_gauge_stencil_buf_offsets[i]], *bufMu_p = &bufMu[0];
        accelerator_for(i,bufMu.size(),1,{
          buf_p[i] = bufMu_p[i];
        });
      }

      stencilView.push_back(stencil[i]->View(AcceleratorRead));
    }
  }

  void makeRightStencil(Vector<vobj> &buffer, std::unique_ptr<FermStencil> &stencil) {
    stencil = std::move(std::unique_ptr<FermStencil>(new FermStencil(_grid_R, _shift_dirs.size(), _checkerboard_R, 
                                                                      _shift_dirs, _shift_displacements)));
    makeStencil(buffer, stencil, getRight(), _size_R);
  }

public:
  GridBase *_grid, *_cb_grid;
  LatticeGaugeField *_U;
  const std::vector<ComplexField> &_mom;
  const std::vector<StagGamma::SpinTastePair> &_gammas;
  Vector<ComplexView> _view_gamma, _view_mom;
  Vector<Integer> _o_coords_E, _o_coords_O, _gamma_indices_comm, _gamma_indices_local, _gauge_stencil_buf_offsets;
  Vector<Coordinate> _i_coor_container;
  std::vector<int> _shift_dirs, _shift_displacements;
  int _N_mom, _orthog_dir;

  GridBase *_grid_L, *_grid_R; 
  std::vector<LatticeColourMatrix> _Umu_L,_Umu_R;
  const FermionField *_left_E, *_left_O, *_right_E, *_right_O;
  int _checker_L, _checkerboard_L, _size_L, _size_L_out;
  int _checker_R, _checkerboard_R, _size_R, _size_R_out;

public:
  template <typename TensorType> // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  void StagMesonFieldNoGlobalSum(TensorType &mat,
                                     const FermionField *lhs_wi_E, const FermionField *lhs_wi_O,
                                     const FermionField *rhs_vj_E, const FermionField *rhs_vj_O,
                                     int orthog_dir, double *t_kernel = nullptr);

  template <typename TensorType>
  void contractSimd(TensorType &result, bool do_comm, 
                            Vector<Scalar_v> &simd_sum_E, Vector<Scalar_v> &simd_sum_O);
  template <typename TensorType>
  void contractSimd(TensorType &result, Vector<Scalar_v> &simd_sum, Vector<Integer> &gamma_indices);

  void spatialContractComm(Vector<Scalar_v>& result);
  void spatialContractLocal(Vector<Scalar_v>& result);
};

template <class FImpl>
template <typename TensorType>
void A2AWorkerMILC<FImpl>::StagMesonFieldNoGlobalSum(TensorType &mat,
                                     const FermionField *lhs_wi_E,
                                     const FermionField *lhs_wi_O,
                                     const FermionField *rhs_vj_E,
                                     const FermionField *rhs_vj_O,
                                     int orthog_dir, double *t_kernel)
{
  assert(_cb_grid->CheckerBoarded(orthog_dir) != 1);

  _size_L = mat.dimension(3);
  _size_R = mat.dimension(4);
  _size_L_out = mat.dimension(3);
  _size_R_out = mat.dimension(4);
  _N_mom   = _mom.size();
  _orthog_dir = orthog_dir;

  _grid_L = _grid;
  _grid_R = _grid;

  _checker_L = lhs_wi_E[0].Grid()->_isCheckerBoarded;
  _checker_R = rhs_vj_E[0].Grid()->_isCheckerBoarded;

  if (_checker_L) {
    _size_L /= 2;
    _grid_L = _cb_grid;
  } 

  if(_checker_R) {
    _size_R /= 2;
    _grid_R = _cb_grid;
  }

  _left_E = lhs_wi_E;
  _left_O = lhs_wi_O;
  _right_E = rhs_vj_E;
  _right_O = rhs_vj_O;

  int nGamma       = _gammas.size();
  int nGamma_local = _gamma_indices_local.size();

  std::vector<ComplexField> stagPhase(nGamma_local,_grid);

  { // Set up staggered phases
    StagGamma spinTaste;
    int mu;
    for (int i = 0; i < nGamma_local; i++) {
      mu = _gamma_indices_local[i];
      stagPhase[i] = 1.0;
      spinTaste.setSpinTaste(_gammas[mu]);
      spinTaste.applyPhase(stagPhase[i],stagPhase[i]); // store spin-taste phase
    }
  }

  Vector<Scalar_v> simd_sum_comm_E, simd_sum_comm_O;
  Vector<Scalar_v> simd_sum_local_E, simd_sum_local_O;

  makeGammaView(&stagPhase[0]);
  makeMomentumView(&_mom[0]);

  if (t_kernel) *t_kernel = -usecond();

  // Run any nonlocal gamma operators
  if (_gamma_indices_comm.size() > 0) {

    _checkerboard_L = Even;
    _checkerboard_R = Even;
    if (_checker_L && _checker_R) {

      _checkerboard_R = Odd;
      spatialContractComm(simd_sum_comm_E);

      _checkerboard_L = Odd;
      _checkerboard_R = Even;
      spatialContractComm(simd_sum_comm_O);

    } else if (_checker_L) {

      spatialContractComm(simd_sum_comm_E);

      _checkerboard_L = Odd;
      spatialContractComm(simd_sum_comm_O);

    } else if (_checker_R) {

      spatialContractComm(simd_sum_comm_O);

      _checkerboard_R = Odd;
      spatialContractComm(simd_sum_comm_E);
    } else {
      spatialContractComm(simd_sum_comm_E);
    }
  }

  // Run any local gamma operators
  if (nGamma_local > 0) {

    _checkerboard_L = Even;
    _checkerboard_R = Even;
    if (_checker_L && _checker_R) {

      spatialContractLocal(simd_sum_local_E);

      _checkerboard_L = Odd;
      _checkerboard_R = Odd;
      spatialContractLocal(simd_sum_local_O);
    } else if (_checker_L) {

      spatialContractLocal(simd_sum_local_E);

      _checkerboard_L = Odd;
      spatialContractLocal(simd_sum_local_O);
    } else if (_checker_R) {

      spatialContractLocal(simd_sum_local_E);

      _checkerboard_R = Odd;
      spatialContractLocal(simd_sum_local_O);
    } else {

      spatialContractLocal(simd_sum_local_E);
    }
  }
  if (t_kernel) *t_kernel += usecond();

  // Contract SIMD vectors
  if (_checker_L || _checker_R) {
    contractSimd(mat, true, simd_sum_comm_E, simd_sum_comm_O);

    contractSimd(mat, false, simd_sum_local_E, simd_sum_local_O);
  } else {
    contractSimd(mat, simd_sum_comm_E, _gamma_indices_comm);

    contractSimd(mat, simd_sum_local_E, _gamma_indices_local);
  }

  accelerator_barrier();

  for(int p=0;p<_view_gamma.size();p++)   _view_gamma[p].ViewClose();
  for(int p=0;p<_view_mom.size();p++)     _view_mom[p].ViewClose();

    _view_gamma.resize(0);
    _view_mom.resize(0);
}

template <class FImpl>
template <typename TensorType>
void A2AWorkerMILC<FImpl>::contractSimd(TensorType &result, bool do_comm, Vector<Scalar_v> &simd_sum_E, 
                                        Vector<Scalar_v> &simd_sum_O) {

  if (simd_sum_E.size() == 0)
    return;

  if (simd_sum_O.size() == 0)
    return;

  auto grid = _grid_L;
  int sizeL = _size_L;
  int sizeR = _size_R;
  int sizeROut = _size_R_out;
  int nMom  = _N_mom;
  int orthogDir = _orthog_dir;
  int checkerL  = _checker_L;
  int checkerR  = _checker_R;

  const int simdSize       = grid->Nsimd();

  int mult_L = checkerL ? 2 : 1;
  int mult_R = checkerR ? 2 : 1;

  int reducedOrthogDimSize = grid->_rdimensions[orthogDir];
  int localOrthogDimSize   = grid->_ldimensions[orthogDir];

  int Nt     = grid->GlobalDimensions()[orthogDir];
  int nGamma, nGammaTotal = _view_gamma.size();

  Scalar_v   *simd_sum_E_p = & simd_sum_E[0];
  Scalar_v   *simd_sum_O_p = & simd_sum_O[0];
  Coordinate *icoor_p        = & _i_coor_container[0];
  Integer    *indexG_p;

  if (do_comm) {
    nGamma = _gamma_indices_comm.size();
    indexG_p = & _gamma_indices_comm[0]; 
  } else {
    nGamma = _gamma_indices_local.size();
    indexG_p = & _gamma_indices_local[0]; 
  }

  int pd = grid->_processors[orthogDir];
  int pc = grid->_processor_coor[orthogDir];

  auto result_p = result.data();
  for (int mu = 0; mu < nGamma; mu++)
  for ( int m=0;m<nMom;m++){
    accelerator_forNB(l_index,sizeL,1,{
      // Sum across simd lanes in the plane, breaking out orthog dir.
      ExtractBuffer<Scalar_s> extracted_E(simdSize),extracted_O(simdSize);

      for(int r_index=0;r_index<sizeR;r_index++){

        for (int rt=0;rt<reducedOrthogDimSize;rt++) {

          int base = nGamma*nMom*l_index+nGamma*nMom*sizeL*r_index+nGamma*nMom*sizeL*sizeR*rt;
              
              // Final matrix layout is mat[nMom,nGamma,orthogDimSize,Nw,Nv]
              // Thus, ignoring gamma and time we get the index of simd_sum
              int ij_rdx = base+mu*nMom+m;
              
              extract(simd_sum_E_p[ij_rdx],extracted_E);
              extract(simd_sum_O_p[ij_rdx],extracted_O);
              
              for(int idx=0;idx<simdSize;idx++){
                  
                  // Calculate local time slice
                  int lt = rt+icoor_p[idx][orthogDir]*reducedOrthogDimSize;
                  // Calculate global time slice
                  int t = lt + pc*localOrthogDimSize;

                  int ij_dx = mult_R*( r_index + sizeR*mult_L*( l_index + sizeL*( t + Nt*( indexG_p[mu] + nGammaTotal*m ) ) ) );

                  // Fix Me: for > 1-link shift operations, this needs to be "if even odd shifts", not do_comm
                  result_p[ij_dx]            = result_p[ij_dx] + extracted_E[idx];
                  result_p[ij_dx]            = result_p[ij_dx] + extracted_O[idx];
                  result_p[ij_dx+sizeROut]   = result_p[ij_dx+sizeROut] + extracted_E[idx];
                  result_p[ij_dx+sizeROut]   = result_p[ij_dx+sizeROut] - extracted_O[idx];
                  if (do_comm) {
                    result_p[ij_dx+1]          = result_p[ij_dx+1] + extracted_O[idx];
                    result_p[ij_dx+1]          = result_p[ij_dx+1] - extracted_E[idx];
                    result_p[ij_dx+sizeROut+1] = result_p[ij_dx+sizeROut+1] - extracted_E[idx];
                    result_p[ij_dx+sizeROut+1] = result_p[ij_dx+sizeROut+1] - extracted_O[idx];
                  } else {
                    result_p[ij_dx+1]          = result_p[ij_dx+1] + extracted_E[idx];
                    result_p[ij_dx+1]          = result_p[ij_dx+1] - extracted_O[idx];
                    result_p[ij_dx+sizeROut+1] = result_p[ij_dx+sizeROut+1] + extracted_E[idx];
                    result_p[ij_dx+sizeROut+1] = result_p[ij_dx+sizeROut+1] + extracted_O[idx];
                  }
              }
            }
          }
      });
  }
}

template <class FImpl>
template <typename TensorType>
void A2AWorkerMILC<FImpl>::contractSimd(TensorType &result,Vector<Scalar_v> &simd_sum, Vector<Integer> &gamma_indices) {

  if (simd_sum.size() == 0)
    return;

  auto grid = _grid_L;
  int sizeL = _size_L;
  int sizeR = _size_R;
  int nMom  = _N_mom;
  int orthogDir = _orthog_dir;

  const int simdSize       = grid->Nsimd();

  int reducedOrthogDimSize = grid->_rdimensions[orthogDir];
  int localOrthogDimSize   = grid->_ldimensions[orthogDir];

  int Nt     = grid->GlobalDimensions()[orthogDir];
  int nGamma = gamma_indices.size();
  int nGammaTotal = _view_gamma.size();

  Scalar_v   *simd_sum_p = & simd_sum[0];
  Coordinate *icoor_p        = & _i_coor_container[0];
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
void A2AWorkerMILC<FImpl>::spatialContractComm(Vector<Scalar_v>& result)
{
  const int nGamma = _gamma_indices_comm.size();

  std::unique_ptr<FermStencil>  stencilRight;
  std::vector<std::unique_ptr<GaugeStencil> > stencilGauge(nGamma);

  Vector<FermView>  viewRight, viewLeft;
  Vector<GaugeView> viewGaugeLeft,viewGaugeRight;
  Vector<GaugeStencilView> viewStencilGauge;

  Vector<vobj> haloBufferRight;
  Vector<vColourMatrix> haloBufferGauge;

  double t0=usecond();
  makeGaugeLeft();
  makeGaugeRight();
  double t1=usecond();

  makeRightStencil(haloBufferRight, stencilRight);
  double t2=usecond();
  makeGaugeStencil(haloBufferGauge,   stencilGauge, viewStencilGauge);
  double t3=usecond();

  makeLeftView(viewLeft);
  makeRightView(viewRight);
  double t4=usecond();

  makeGaugeViewLeft(viewGaugeLeft);
  makeGaugeViewRight(viewGaugeRight);
  double t5=usecond();

  std::cout << GridLogPerformance << " MesonField one link timings: build link fields:" << (t1-t0)/1000 << "ms, right comms:" << (t2-t1)/1000 << "ms" << std::endl;   
  std::cout << GridLogPerformance << " MesonField one link timings: gauge comms:" << (t3-t2)/1000 << "ms, build left+right views:" << (t4-t3)/1000 << "ms" << std::endl;   
  std::cout << GridLogPerformance << " MesonField one link timings: build left+right gauage views:" << (t5-t4)/1000 << "ms" << std::endl;   

  int haloBuffRightSize = stencilRight->_unified_buffer_size;

  GridBase* grid = _grid_L;
  if (_checker_R)
    grid = _grid_R;

  int sizeL = _size_L;
  int sizeR = _size_R;
  int nMom  = _N_mom;
  int orthogDir = _orthog_dir;
  int checkerL  = _checker_L;
  int checkerR  = _checker_R;

  const int simdSize     = grid->Nsimd();

  // Help in iterating over elements in the same time slice
  const int nBlocks      = grid->_slice_nblock[orthogDir];
  const int blockStride  = grid->_slice_stride[orthogDir];
  const int rtStride     = grid->_ostride[orthogDir];
  const int vecsPerSlicePerBlock = grid->_slice_block[orthogDir];

  // Output matrix dimensions
  const int reducedOrthogDimSize = grid->_rdimensions[orthogDir];

  const int MFrvol = reducedOrthogDimSize *sizeL *sizeR *nMom * nGamma;

  result.resize(MFrvol);

  Scalar_v        *result_p = & result[0];     // Return object
  accelerator_for(r, MFrvol,1,{
    result_p[r] = Zero();
  });  

  // Pass data to the device
  ComplexView      *viewM_p  = & _view_mom[0];     // Momenta
  FermView         *viewL_p  = & viewLeft[0];    // bra vectors
  FermView         *viewR_p  = & viewRight[0];   // ket vectors
  GaugeView        *viewGL_p = & viewGaugeLeft[0];   // Gauge links for bras
  GaugeView        *viewGR_p = & viewGaugeRight[0];   // Gauge links for kets
  GaugeStencilView *stencilG_p = & viewStencilGauge[0]; // Gauge tencil for shifted links

  vobj          *bufRight_p = &haloBufferRight[0]; // buffer for shifted kets in halo region
  vColourMatrix *bufGauge_p = &haloBufferGauge[0]; // buffer for shifted links in halo region

  Integer *offsetG_p = & _gauge_stencil_buf_offsets[0]; // buffer offsets for shifted links in halo region

  {
  autoView(stencilR_p,(*stencilRight),AcceleratorRead); // Stencil for shifted kets

  Integer *oCoords_p = & (getLeftCoords())[0]; // maps checkerboarded indices to corresponding full grid index

  if (checkerR && !checkerL)
    oCoords_p = & (getRightCoords())[0];

  accelerator_for2d(l_index,sizeL,r_index,sizeR,simdSize,{

    calcColourMatrix link_ahead, link_behind;
    calcSpinor left, shift_ahead, shift_behind;
    calcScalar temp_site, momentum_phase, sum[MF_SUM_ARRAY_MAX];
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

        int fullss = ss, ssL = ss, ssR = ss;

        if (checkerL && checkerR) {
          fullss = oCoords_p[ss];
        } else if (checkerL) {
          fullss = oCoords_p[ss];
          ssR = fullss;
        } else if (checkerR) {
          fullss = oCoords_p[ss];
          ssL = fullss;
        }
        acceleratorSynchronise();

        left   = coalescedRead(viewL_p[l_index][ssL]);

        sumIndex = 0;
        for (int mu = 0; mu < nGamma; mu++) {
          link_ahead = coalescedRead(viewGL_p[mu][ssL]);

          SE=stencilR_p.GetEntry(ptype,2*mu,ssR);
          if(SE->_is_local) { 
            shift_ahead = coalescedReadPermute(viewR_p[r_index][SE->_offset],ptype,SE->_permute);
          } else {
            shift_ahead = coalescedRead(bufRight_p[r_index*haloBuffRightSize+SE->_offset]);
          }
          acceleratorSynchronise();

          SE=stencilR_p.GetEntry(ptype,2*mu+1,ssR);
          if(SE->_is_local) { 
            shift_behind = coalescedReadPermute(viewR_p[r_index][SE->_offset],ptype,SE->_permute);
          } else {
            shift_behind = coalescedRead(bufRight_p[r_index*haloBuffRightSize+SE->_offset]);
          }
          acceleratorSynchronise();

          SE=stencilG_p[mu].GetEntry(ptype,0,ssR);
          if(SE->_is_local) { 
            link_behind = coalescedReadPermute(viewGR_p[mu][SE->_offset],ptype,SE->_permute);
          } else {
            link_behind = coalescedRead(bufGauge_p[offsetG_p[mu]+SE->_offset]);
          }
          acceleratorSynchronise();

          temp_site = innerProduct(left,link_ahead*shift_ahead+adj(link_behind)*shift_behind);

          for ( int m=0;m<nMom;m++) {
            momentum_phase = coalescedRead(viewM_p[m][fullss]);
            sum[sumIndex+m]  += momentum_phase*temp_site;
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
  for(int p=0;p<viewLeft.size();p++)  viewLeft[p].ViewClose();
  for(int p=0;p<viewRight.size();p++) viewRight[p].ViewClose();
  for(int p=0;p<viewGaugeRight.size();p++)   viewGaugeRight[p].ViewClose();
  for(int p=0;p<viewGaugeLeft.size();p++)   viewGaugeLeft[p].ViewClose();
  for(int p=0;p<viewStencilGauge.size();p++)   viewStencilGauge[p].ViewClose();
    viewLeft.resize(0);
    viewRight.resize(0);
    viewGaugeRight.resize(0);
    viewGaugeLeft.resize(0);
    viewStencilGauge.resize(0);
}

template <class FImpl>
void A2AWorkerMILC<FImpl>::spatialContractLocal(Vector<Scalar_v>& result)
{
  Vector<FermView> viewLeft; 
  Vector<FermView> viewRight;

  makeLeftView(viewLeft);
  makeRightView(viewRight);

  auto grid = _grid_L;
  int sizeL = _size_L;
  int sizeR = _size_R;
  int orthogDir = _orthog_dir;
  int checkerL  = _checker_L;
  int checkerR  = _checker_R;
  const int nMom  = _N_mom;

  const int simdSize             = grid->Nsimd();
  const int reducedOrthogDimSize = grid->_rdimensions[orthogDir];

  const int nBlocks      = grid->_slice_nblock[orthogDir];
  const int blockStride  = grid->_slice_stride[orthogDir];
  const int rtStride     = grid->_ostride[orthogDir];
  const int vecsPerSlicePerBlock = grid->_slice_block[orthogDir];

  const int nGamma = _gamma_indices_local.size();

  const int MFrvol = reducedOrthogDimSize * sizeL * sizeR * nMom * nGamma;

  result.resize(MFrvol);

  Scalar_v     *result_p   = & result[0];
  accelerator_for(r, MFrvol,1,{
    result_p[r] = Zero();
  });  

  // Pointers for accelerator indexing
  FermView     *viewL_p    = &viewLeft[0];
  FermView     *viewR_p    = &viewRight[0];

  ComplexView  *viewG_p    = & _view_gamma[0];
  ComplexView  *viewM_p    = & _view_mom[0];

  Integer      *indexG_p  = & _gamma_indices_local[0]; 
  Integer      *oCoords_p;


  // If needed, Grab appropriate mapping between checkerboarded lattice and full lattice
  if(checkerL) {
    oCoords_p = & (getLeftCoords())[0];
  } else {
    oCoords_p = & (getRightCoords())[0];
  }

  accelerator_for2d(l_index,sizeL,r_index,sizeR,simdSize,{

    int so, ss, base, sumIndex, ssL, ssR, fullss;
    calcSpinor left, right;
    calcScalar temp_site, gamma_phase, momentum_phase, sum[MF_SUM_ARRAY_MAX];

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

          left  = coalescedRead(viewL_p[l_index][ssL]);
          right = coalescedRead(viewR_p[r_index][ssR]);
      
          temp_site  = innerProduct(left,right);

          sumIndex = 0;
          for (int mu = 0; mu < nGamma; mu++) {
            gamma_phase = coalescedRead(viewG_p[indexG_p[mu]][fullss]);
            for ( int m=0;m<nMom;m++) {
              momentum_phase = coalescedRead(viewM_p[m][fullss]);
              
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
  for(int p=0;p<viewLeft.size();p++)  viewLeft[p].ViewClose();
  for(int p=0;p<viewRight.size();p++) viewRight[p].ViewClose();
    viewLeft.resize(0);
    viewRight.resize(0);
}

NAMESPACE_END(Grid);
