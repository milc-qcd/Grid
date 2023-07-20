#pragma once
#include <Grid/Grid_Eigen_Tensor.h>

#ifndef accelerator_for2dNB
#define accelerator_for2dNB_no_err(iter1, num1, iter2, num2, nsimd, ... )  accelerator_for2d(iter1, num1, iter2, num2, nsimd, { __VA_ARGS__ } );
#else
#define accelerator_for2dNB_no_err(iter1, num1, iter2, num2, nsimd, ... )  accelerator_for2dNB(iter1, num1, iter2, num2, nsimd, { __VA_ARGS__ } );
#endif

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

protected:
  template<typename Vtype>
  class A2AView {
    Vector<LatticeView<Vtype> > _view;

  public:
    void buildView(const Lattice<Vtype> *field, int size) {
      _view.reserve(size);

      for(int p=0;p<size;p++) {
        _view.push_back(field[p].View(AcceleratorRead));
      }
    }

    int size() {return _view.size();}

    LatticeView<Vtype> &operator[](size_t i) { return _view[i];}
    
    ~A2AView() {
      for(int p=0;p<_view.size();p++)   _view[p].ViewClose();
      _view.resize(0);
    }
  };

public:
  GridBase *_grid, *_cb_grid;
  LatticeGaugeField *_U;
  const std::vector<StagGamma::SpinTastePair> &_gammas;
  const std::vector<ComplexField> &_mom;
  std::vector<ComplexField> _stag_phase;
  std::vector<LatticeColourMatrix> _Umu_E,_Umu_O;
  A2AView<cobj> _view_gamma, _view_mom;
  A2AView<vColourMatrix> _view_links_E, _view_links_O;
  Vector<Integer> _o_coords_E, _o_coords_O, _gamma_indices_comm, _gamma_indices_local, _gauge_stencil_buf_offsets;
  Vector<Coordinate> _i_coor_container;
  std::vector<std::unique_ptr<GaugeStencil> > _stencil_gauge_E,_stencil_gauge_O;
  Vector<GaugeStencilView> _view_stencil_gauge_E,_view_stencil_gauge_O;
  Vector<vColourMatrix> _halo_buffer_gauge_E,_halo_buffer_gauge_O;
  std::vector<int> _shift_dirs, _shift_displacements;
  int _N_mom, _orthog_dir;

  GridBase *_grid_L, *_grid_R; 
  const FermionField *_left_E, *_left_O, *_right_E, *_right_O;
  int _checker_L, _checkerboard_L, _size_L, _size_L_out;
  int _checker_R, _checkerboard_R, _size_R, _size_R_out;

public:
  A2AWorkerMILC() = delete;
  A2AWorkerMILC(GridBase *grid, const std::vector<StagGamma::SpinTastePair>& gammas, const std::vector<ComplexField> &mom, LatticeGaugeField* U = nullptr, GridBase *cbGrid = nullptr)
  : _grid(grid), _gammas(gammas), _mom(mom), _U(U), _cb_grid(cbGrid) {

    _N_mom   = _mom.size();

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

    if (_gamma_indices_local.size() > 0) {
      buildLocalPhases();
    }

    _view_mom.buildView(&_mom[0],_mom.size());

    if (_gamma_indices_comm.size() > 0) {
      double t0 = usecond();
      buildGaugeLinks();
      double t1 = usecond();
      std::cout << GridLogPerformance << " MesonField one link timings: build link fields:" << (t1-t0)/1000 << "ms" << std::endl;   
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

  ~A2AWorkerMILC() {
    for(int p=0;p<_view_stencil_gauge_E.size();p++) {
      _view_stencil_gauge_E[p].ViewClose(); 
      _view_stencil_gauge_O[p].ViewClose();
    }
    _view_stencil_gauge_E.resize(0);
    _view_stencil_gauge_O.resize(0);
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
      //local_flops = 22.0+(6.0+6.0+2.0)*(_N_mom*_gamma_indices_local.size());
      local_flops = 22.0+(6.0+2.0)*(_gamma_indices_local.size());
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

  A2AView<vColourMatrix> &getGaugeViewLeft() {
    if (_checkerboard_L == Odd)
      return _view_links_O;
    else {
      return _view_links_E;
    }
  }

  A2AView<vColourMatrix> &getGaugeViewRight() {
    if (_checkerboard_R == Odd)
      return _view_links_O;
    else {
      return _view_links_E;
    }
  }

  Vector<GaugeStencilView> &getGaugeStencilView() {
    if (_checkerboard_R == Odd)
      return _view_stencil_gauge_O;
    else {
      return _view_stencil_gauge_E;
    }
  }

  Vector<vColourMatrix> &getGaugeStencilHalo() {
    if (_checkerboard_R == Odd)
      return _halo_buffer_gauge_O;
    else {
      return _halo_buffer_gauge_E;
    }
  }

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

  void buildLocalPhases() {
    int nGamma_local = _gamma_indices_local.size();

    _stag_phase.resize(nGamma_local,_grid);

    { // Set up staggered phases
      StagGamma spinTaste;
      int mu;
      for (int i = 0; i < nGamma_local; i++) {
        mu = _gamma_indices_local[i];
        _stag_phase[i] = 1.0;
        spinTaste.setSpinTaste(_gammas[mu]);
        spinTaste.applyPhase(_stag_phase[i],_stag_phase[i]); // store spin-taste phase
      }
    }

    _view_gamma.buildView(&_stag_phase[0],nGamma_local);
  }

  void buildGaugeLinks()
  {
    int mu, nGamma_comm = _gamma_indices_comm.size();

    _Umu_E.resize(nGamma_comm,_cb_grid);
    _Umu_O.resize(nGamma_comm,_cb_grid);

    StagGamma spinTaste;
    LatticeColourMatrix Umu_temp(_U->Grid());

    for (int i = 0; i < nGamma_comm; i++) {
      mu = _gamma_indices_comm[i];
      spinTaste.setSpinTaste(_gammas[mu]);

      Umu_temp = PeekIndex<LorentzIndex>(*_U,_shift_dirs[2*i]); // Store full lattice links in shift direction

      spinTaste.applyPhase(Umu_temp,Umu_temp); // store spin-taste phase

      pickCheckerboard(Even,_Umu_E[i],Umu_temp);
      pickCheckerboard(Odd,_Umu_O[i],Umu_temp);
    }
   
    double t0=usecond();
    makeGaugeStencil();
    double t1=usecond();

    std::cout << GridLogPerformance << " MesonField one link timings: gauge comms:" << (t1-t0)/1000 << "ms" << std::endl;   

    _view_links_E.buildView(&_Umu_E[0],nGamma_comm);
    _view_links_O.buildView(&_Umu_O[0],nGamma_comm);
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

  void makeGaugeStencil() {

    Vector<vColourMatrix> bufMuE, bufMuO;
    int size = _shift_dirs.size()/2;

    _gauge_stencil_buf_offsets.resize(size);

    _view_stencil_gauge_E.reserve(size);
    _view_stencil_gauge_O.reserve(size);
    _stencil_gauge_E.resize(size);
    _stencil_gauge_O.resize(size);

    for (int i=0;i<size;i++) {
      _stencil_gauge_E[i] = std::move(std::unique_ptr<GaugeStencil>(new GaugeStencil(_cb_grid, 1, Even, {_shift_dirs[2*i]}, {-1})));
      _stencil_gauge_O[i] = std::move(std::unique_ptr<GaugeStencil>(new GaugeStencil(_cb_grid, 1, Odd, {_shift_dirs[2*i]}, {-1})));

      makeStencil(bufMuE,_stencil_gauge_E[i],&_Umu_E[i],1);
      makeStencil(bufMuO,_stencil_gauge_O[i],&_Umu_O[i],1);

      _gauge_stencil_buf_offsets[i] = _halo_buffer_gauge_E.size();

      _halo_buffer_gauge_E.resize(_halo_buffer_gauge_E.size()+bufMuE.size());
      _halo_buffer_gauge_O.resize(_halo_buffer_gauge_O.size()+bufMuO.size());

      if (bufMuE.size() > 0) {
        vColourMatrix *bufE_p = &_halo_buffer_gauge_E[_gauge_stencil_buf_offsets[i]], *bufMuE_p = &bufMuE[0];
        vColourMatrix *bufO_p = &_halo_buffer_gauge_O[_gauge_stencil_buf_offsets[i]], *bufMuO_p = &bufMuO[0];

        accelerator_for(i,bufMuE.size(),1,{
          bufE_p[i] = bufMuE_p[i];
          bufO_p[i] = bufMuO_p[i];
        });
      }

      _view_stencil_gauge_E.push_back(_stencil_gauge_E[i]->View(AcceleratorRead));
      _view_stencil_gauge_O.push_back(_stencil_gauge_O[i]->View(AcceleratorRead));
    }
  }

  void makeRightStencil(Vector<vobj> &buffer, std::unique_ptr<FermStencil> &stencil) {
    stencil = std::move(std::unique_ptr<FermStencil>(new FermStencil(_grid_R, _shift_dirs.size(), _checkerboard_R, 
                                                                      _shift_dirs, _shift_displacements)));
    makeStencil(buffer, stencil, getRight(), _size_R);
  }

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

  Vector<Scalar_v> simd_sum_comm_E, simd_sum_comm_O;
  Vector<Scalar_v> simd_sum_local_E, simd_sum_local_O;

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
  Vector<FermView>  viewRight, viewLeft;
  Vector<vobj> haloBufferRight;
 
  double t1=usecond();
  makeRightStencil(haloBufferRight, stencilRight);
  double t2=usecond();
  makeLeftView(viewLeft);
  makeRightView(viewRight);
  double t3=usecond();

  A2AView<vColourMatrix>   &viewGaugeLeft    = getGaugeViewLeft();
  A2AView<vColourMatrix>   &viewGaugeRight   = getGaugeViewRight();
  Vector<GaugeStencilView> &viewGaugeStencil = getGaugeStencilView();
  Vector<vColourMatrix>    &haloBufferGauge  = getGaugeStencilHalo();

  std::cout << GridLogPerformance << " MesonField one link timings: right comms:" << (t2-t1)/1000 << "ms" << std::endl;   
  std::cout << GridLogPerformance << " MesonField one link timings: build left+right views:" << (t3-t2)/1000 << "ms" << std::endl;   

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
  GaugeStencilView *stencilG_p = & viewGaugeStencil[0]; // Gauge tencil for shifted links

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
    viewLeft.resize(0);
    viewRight.resize(0);
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

  const int simdSize             = grid->Nsimd();
  const int reducedOrthogDimSize = grid->_rdimensions[orthogDir];

  const int nBlocks      = grid->_slice_nblock[orthogDir];
  const int blockStride  = grid->_slice_stride[orthogDir];
  const int rtStride     = grid->_ostride[orthogDir];
  const int vecsPerSlicePerBlock = grid->_slice_block[orthogDir];

  const int nGamma = _gamma_indices_local.size();

  const int MFrvol = reducedOrthogDimSize * sizeL * sizeR * nGamma;

  result.resize(MFrvol);

  Scalar_v     *result_p   = & result[0];
  accelerator_for(r, MFrvol,1,{
    result_p[r] = Zero();
  });  

  // Pointers for accelerator indexing
  FermView     *viewL_p    = &viewLeft[0];
  FermView     *viewR_p    = &viewRight[0];

  ComplexView  *viewG_p    = & _view_gamma[0];

  Integer      *indexG_p  = & _gamma_indices_local[0]; 
  Integer      *oCoords_p = & (getLeftCoords())[0];

  assert(nGamma <= MF_SUM_ARRAY_MAX);

  int pointsPerBatch = max(1,MF_SUM_ARRAY_MAX/nGamma); 
  int gammasPerPoint = nGamma;
  int batchSize = gammasPerPoint*pointsPerBatch;

  //Vector<Integer> gamma_map(MF_SUM_ARRAY_MAX);
  //Vector<Integer> point_map(MF_SUM_ARRAY_MAX);

  //Integer *mapG_p = &gamma_map[0];
  //Integer *mapS_p = &point_map[0];

  //for(int i=0;i<batchSize;i++) { //Example: 2 points, 2 gammas per point, 2 momenta per point
  //  mapS_p[i] = i%pointsPerBatch; //0123012301230123
  //  mapG_p[i] = i/gammasPerPoint;    //0000111122223333
  //};

  int localSpatialVolume = nBlocks*vecsPerSlicePerBlock;
  int localVolume = reducedOrthogDimSize*localSpatialVolume;

  assert(localSpatialVolume % pointsPerBatch == 0);

  Vector<Integer> slice_indices(localVolume);
  Integer *ss_p = &slice_indices[0];

  accelerator_for(ss,localVolume,1,{
    int rt = ss/localSpatialVolume;
    int block = (ss/vecsPerSlicePerBlock)%nBlocks;
    int vec = ss%vecsPerSlicePerBlock;
    ss_p[ss] = rt*rtStride + block*blockStride + vec;
  });

  for (int rt=0; rt < reducedOrthogDimSize; rt++) {

    ss_p = &slice_indices[rt*localSpatialVolume];

    accelerator_for2dNB_no_err(l_index,sizeL,r_index,sizeR,simdSize,{

      calcScalar temp_site[MF_SUM_ARRAY_MAX], gamma_phase[MF_SUM_ARRAY_MAX], sum[MF_SUM_ARRAY_MAX];

      for (int p = 0; p < batchSize; p++) {
        sum[p] = 0.0;
      }

      for (int so=0;so < localSpatialVolume; so+=pointsPerBatch) {

        // Read data
        for (int i=0;i<batchSize;i++) {
          const int ii = i/pointsPerBatch;
          const int si = i%pointsPerBatch;
          gamma_phase[i] = coalescedRead(viewG_p[indexG_p[ii]][oCoords_p[ss_p[so+si]]]);
        }

        // Inner product
        for (int i=0;i<pointsPerBatch;i++) {
          temp_site[i] = innerProduct(coalescedRead(viewL_p[l_index][ss_p[so+i]]),coalescedRead(viewR_p[r_index][ss_p[so+i]]));
        }

        // splat
        /*for (int i = pointsPerBatch; i < batchSize; i++)
        {
           temp_site[i] = temp_site[i%pointsPerBatch];
        }
        for (int i = 0; i < batchSize; i++)
        {
          gamma_phase[i] = gamma_phase[i/pointsPerBatch];
        }*/

        // mac
        for (int i=0; i<batchSize;i++) {
          int ii = i/pointsPerBatch;
          int iii = i%pointsPerBatch;
          sum[ii] += gamma_phase[i]*temp_site[iii];
        }
      }

      // Reduce points with same gamma and Write
      int write_idx = nGamma*(l_index+sizeL*r_index+sizeL*sizeR*rt);
      for (int i=0; i<gammasPerPoint;i++) {
        coalescedWrite(result_p[write_idx+i],sum[i]);
      }
    });
  }
  accelerator_barrier();
  for(int p=0;p<viewLeft.size();p++)  viewLeft[p].ViewClose();
  for(int p=0;p<viewRight.size();p++) viewRight[p].ViewClose();
    viewLeft.resize(0);
    viewRight.resize(0);
}

NAMESPACE_END(Grid);
