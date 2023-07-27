#pragma once
#include <Grid/Grid_Eigen_Tensor.h>

#ifndef MF_SUM_ARRAY_MAX
#define MF_SUM_ARRAY_MAX 16
#endif

NAMESPACE_BEGIN(Grid);

#ifndef accelerator_for2dNB
#define accelerator_for2dNB_no_err(iter1, num1, iter2, num2, nsimd, ... )  accelerator_for2d(iter1, num1, iter2, num2, nsimd, { __VA_ARGS__ } );
#else
#define accelerator_for2dNB_no_err(iter1, num1, iter2, num2, nsimd, ... )  accelerator_for2dNB(iter1, num1, iter2, num2, nsimd, { __VA_ARGS__ } );
#endif

#if 0
#ifndef GRID_SIMT
#define accelerator_for2dNB_shm(iter1, num1, iter2, num2, nsimd, shm, ... )  \
accelerator_for2d(iter1, num1, iter2, num2, nsimd, { \
  cobj sum[shm]; \
  __VA_ARGS__ \
});
#else

#define accelerator_for2dNB_shm(iter1, num1, iter2, num2, nsimd, shm, ... )  \
  {                 \
    int nt=acceleratorThreads();          \
    typedef uint64_t Iterator;            \
    auto lambda = [=] accelerator         \
      (Iterator iter1,Iterator iter2,Iterator lane, cobj* sum) mutable {   \
      __VA_ARGS__;              \
    };                  \
    dim3 cu_threads(nsimd,acceleratorThreads(),1);      \
    dim3 cu_blocks ((num1+nt-1)/nt,num2,1);       \
    ShmLambdaApply<cobj,decltype(lambda)><<<cu_blocks,cu_threads,shm*sizeof(cobj),computeStream>>>(num1,num2,nsimd,lambda);  \
  }
#endif
#endif

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
  template<typename Vtype, typename obj>
  class A2AViewBase {
  public:
    Vector<Vtype> _view;

  public:
    Vtype &operator[](size_t i) { return _view[i];}
    
    int size() {return _view.size();}
    void reserve(int size) { _view.reserve(size); }

    void closeViews() {
      for(int p=0;p<_view.size();p++)   _view[p].ViewClose();
        _view.resize(0);
    }
  };

  template<typename obj>
  class A2AView: public A2AViewBase<LatticeView<obj>,obj> {
  public:
    void addView(const Lattice<obj> &field) {
      this->_view.push_back(field.View(AcceleratorRead));
    }
  };

  template<typename obj>
  class A2AStencilView: public A2AViewBase<CartesianStencilView<obj,obj,FImplParams>, obj>{
  public:
    std::vector<std::unique_ptr<CartesianStencil<obj,obj,FImplParams> > > _stencils;
    Vector<obj> _buffer;
    Vector<Integer> _offset;

  public:
    CartesianStencil<obj,obj,FImplParams> & getStencil(int i) { return *_stencils[i]; }
    obj & getBuffer(int i) { return _buffer[i]; }
    Integer & getOffset(int i) { return _offset[i]; }
    
    void addStencil(std::unique_ptr<CartesianStencil<obj,obj,FImplParams> > &stencil) {
      _stencils.push_back(std::move(stencil));
    }

    void openViews(){
      for (auto &stencil: _stencils) {
        this->_view.push_back(stencil->View(AcceleratorRead));
      }
    }

    void append(const Lattice<obj> &field) {
      GridBase *grid = field.Grid();
      SimpleCompressor<obj> compressor;

      auto &stencil = _stencils.back();

      int comm_buf_size;
      obj *buf_p;

      stencil->HaloExchange(field,compressor);

      comm_buf_size = stencil->_unified_buffer_size;
      _offset.push_back(_buffer.size());

      _buffer.resize(_buffer.size()+comm_buf_size);
      buf_p = &_buffer[_offset.back()];

      if (comm_buf_size > 0) {
        obj *comm_buf_p = stencil->CommBuf();
        accelerator_for(i,comm_buf_size,1,{
          buf_p[i] = comm_buf_p[i];
        });
      }
    }
  };

  class A2ATaskHalfHalfLocal {
    A2AView<vobj> &_left, &_right;
    A2AView<cobj> &_gamma;
    Vector<Integer> &_gamma_indices;
    GridBase *_grid;
    int _cb;

  public:
    A2ATaskHalfHalfLocal(A2AView<vobj> &left, A2AView<vobj> &right, A2AView<cobj> &gamma,
                          Vector<Integer> &gammaIndices, GridBase *grid, int cb = Even):
    _left(left),_right(right),_gamma(gamma),_gamma_indices(gammaIndices),_grid(grid),_cb(cb) {}

    double getFlops() {
      // One complex multiply takes 6 floating point ops (4 mult, 2 add) 
      // --> complex inner product is 3 complex mult, 2 complex add = 3*6 + 2*2 = 22 double precision floating ops

      // For each vector and at each lattice site:
      //  - one inner product
      //  - For each gamma and momentum
      //    - multiply by gamma phase
      //    - multiply by momentum phase
      //    - sum
      return (22.0+(6.0+2.0)*(_gamma_indices.size()));
    }

    template <typename MatType>
    void execute(MatType &result, int orthogDir) {

      assert(orthogDir == Tdir);

      FermView &left = _left[0];
      FermView &right = _right[0];
      ComplexView &gamma = _gamma[0];

      const int sizeLOut = result.dimension(3);
      const int sizeROut = result.dimension(4);
      const int sizeR = sizeROut/2;
      const int sizeL = sizeLOut/2;

      const int nGamma = _gamma_indices.size();

      const int simdSize             = _grid->Nsimd();
      const int reducedOrthogDimSize = _grid->_rdimensions[orthogDir];

      const int nBlocks      = _grid->_slice_nblock[orthogDir];
      const int vecsPerSlicePerBlock = _grid->_slice_block[orthogDir];

      assert(nBlocks == 1);     // This kernel assumes lattice is coalesced over time slices

      const int localSpatialVolume     = _grid->_ostride[orthogDir];

      // Pointers for accelerator indexing
      FermView     *viewL_p    = &left;
      FermView     *viewR_p    = &right;
      ComplexView  *viewG_p    = &gamma;

      int mult_L = 2;
      int mult_R = 2;
      int localOrthogDimSize   = _grid->_ldimensions[orthogDir];

      int Nt     = _grid->GlobalDimensions()[orthogDir];

      Integer* indexG_p = & _gamma_indices[0]; 

      int pd = _grid->_processors[orthogDir];
      int pc = _grid->_processor_coor[orthogDir];

      auto result_p = result.data();
      
      bool even = _cb == Even;

      int at = acceleratorThreads();

      const int MFrvol = reducedOrthogDimSize *sizeL *sizeR * nGamma;

      Vector<cobj> shmem(MFrvol); // FIXME: use shared memory...
      cobj *shm_p = &shmem[0];

      accelerator_for(r, MFrvol,1,{
        shm_p[r] = Zero();
      });  

      accelerator_for2d(l_index,sizeL,r_index,sizeR,simdSize,{

        int ss, shmem_base = nGamma * reducedOrthogDimSize*(l_index+sizeL*r_index);
        calcScalar gamma_phase, temp_site, sum[MF_SUM_ARRAY_MAX];

        for (int rt=0;rt < reducedOrthogDimSize; rt++) {

          for (int mu=0;mu<nGamma;mu++) {
            sum[mu] = Zero();
          }

          int shmem_idx = nGamma*rt+shmem_base;
          for (int so=0;so < localSpatialVolume; so++) {
            ss = rt*localSpatialVolume+so;
            temp_site = innerProduct(coalescedRead(viewL_p[l_index][ss]),coalescedRead(viewR_p[r_index][ss]));

            for (int mu = 0; mu < nGamma; mu++) {
              gamma_phase = coalescedRead(viewG_p[mu][ss]);
              sum[mu] += gamma_phase*temp_site;
            }
          }
          for (int mu=0;mu<nGamma;mu++) {
            coalescedWrite(shm_p[shmem_idx+mu],sum[mu]);
          }
        }
      });

      accelerator_for2d(l_index,sizeL,r_index,sizeR,1,{
        ExtractBuffer<Scalar_s> extracted(simdSize);
        Scalar_s temp;
        int shmem_idx = reducedOrthogDimSize*nGamma*(l_index+sizeL*r_index);
        for (int i=0;i<nGamma*reducedOrthogDimSize;i++) {
          int mu = i%nGamma;
          int rt = i/nGamma;
          extract(shm_p[shmem_idx+i],extracted);

          for(int idx=0;idx<simdSize;idx +=2){

            // Calculate local time slice
            int lt = rt+idx/2*reducedOrthogDimSize;
            // Calculate global time slice
            int t_out = lt + pc*localOrthogDimSize;

            int ij_dx = mult_R*( r_index + sizeR*mult_L*( l_index + sizeL*( t_out + Nt*indexG_p[mu] ) ) );

            temp = extracted[idx] + extracted[idx+1];

            result_p[ij_dx]            += temp;
            result_p[ij_dx+sizeROut+1] += temp;

            if (even) {
              result_p[ij_dx+sizeROut]   += temp;
              result_p[ij_dx+1]          += temp;
            } else {
              result_p[ij_dx+sizeROut]   += -temp;
              result_p[ij_dx+1]          += -temp;
            }
            acceleratorSynchronise();
          }
        }
      });
      accelerator_barrier();
    }
  };

  class A2ATaskHalfHalfOneLink {
    A2AView<vobj> &_left, &_right;
    A2AView<vColourMatrix>   &_links_left,&_links_right;
    A2AStencilView<vobj> &_right_stencil;
    A2AStencilView<vColourMatrix> &_link_stencil;
    Vector<Integer> &_gamma_indices;
    GridBase *_grid;
    int _cb;

  public:
    A2ATaskHalfHalfOneLink(A2AView<vobj> &left, A2AView<vobj> &right, A2AStencilView<vobj> &rightStencil, A2AView<vColourMatrix> &linksLeft,
                          A2AView<vColourMatrix> &linksRight, A2AStencilView<vColourMatrix> &linkStencil, 
                          Vector<Integer> &gammaIndices, GridBase *grid, int cb = Even):
    _left(left),_right(right),_links_left(linksLeft),_links_right(linksRight),_link_stencil(linkStencil),_right_stencil(rightStencil),
    _gamma_indices(gammaIndices),_grid(grid),_cb(cb) {}

    double getFlops() {
      // matrix*vector = 3 inner products
      // current code: innerProduct(left,link_ahead*shift_ahead+adj(link_behind)*shift_behind)
      //  = matrix*vector + matrix* vector --> inner product = 7 inner products and 1 complex sum

      // For each vector, each gamma, and at each lattice site:
      //  - one inner product
      //  - two su(3) matrix*vector ops
      //  - one complex sum
      return ((7*22.0+2.0)*_gamma_indices.size());
    }

    template <typename MatType>
    void execute(MatType &result, int orthogDir) {

      assert(orthogDir == Tdir);

      FermView         &left  = _left[0];
      FermView         &right = _right[0];

      GaugeView        &linksLeft  = _links_left[0];
      GaugeView        &linksRight = _links_right[0];

      GaugeStencilView &linkStencil  = _link_stencil[0];
      FermStencilView  &rightStencil = _right_stencil[0];

      const int sizeLOut = result.dimension(3);
      const int sizeROut = result.dimension(4);
      const int sizeR = sizeROut/2;
      const int sizeL = sizeLOut/2;

      const int nGamma = _gamma_indices.size();

      const int simdSize             = _grid->Nsimd();
      const int reducedOrthogDimSize = _grid->_rdimensions[orthogDir];

      const int nBlocks      = _grid->_slice_nblock[orthogDir];

      assert(nBlocks == 1);     // This kernel assumes lattice is coalesced over time slices

      const int localSpatialVolume = _grid->_ostride[orthogDir];

      // Pointers for accelerator indexing
      FermView  *viewL_p    = &left;
      FermView  *viewR_p    = &right;
      GaugeView *viewGL_p   = &linksLeft;
      GaugeView *viewGR_p   = &linksRight;

      GaugeStencilView *stencilG_p = & linkStencil; // Gauge tencil for shifted links
      FermStencilView  *stencilR_p = & rightStencil; // Gauge tencil for shifted links

      vobj          *bufRight_p = &_right_stencil.getBuffer(0); // buffer for shifted kets in halo region
      vColourMatrix *bufGauge_p = &_link_stencil.getBuffer(0); // buffer for shifted links in halo region

      Integer *offsetG_p = & _link_stencil.getOffset(0); // buffer offsets for shifted links in halo region
      int haloBuffRightSize = _right_stencil.getOffset(1);

      int mult_L = 2;
      int mult_R = 2;
      int localOrthogDimSize   = _grid->_ldimensions[orthogDir];

      int Nt     = _grid->GlobalDimensions()[orthogDir];

      Integer* indexG_p = & _gamma_indices[0]; 

      int pd = _grid->_processors[orthogDir];
      int pc = _grid->_processor_coor[orthogDir];

      auto result_p = result.data();
      
      bool even = _cb == Even;

      //int at = acceleratorThreads();

      const int MFrvol = reducedOrthogDimSize *sizeL *sizeR * nGamma;

      Vector<cobj> shmem(MFrvol); // FIXME: use shared memory...
      cobj *shm_p = &shmem[0];

      accelerator_for(r, MFrvol,1,{
        shm_p[r] = Zero();
      });  

      accelerator_for2d(l_index,sizeL,r_index,sizeR,simdSize,{

        calcColourMatrix link_ahead, link_behind;
        calcSpinor left, shift_ahead, shift_behind;
        calcScalar sum[MF_SUM_ARRAY_MAX];

        StencilEntry *SE;
        int ptype, ss;
        int shmem_base = nGamma * reducedOrthogDimSize*(l_index+sizeL*r_index);

        for (int rt=0;rt < reducedOrthogDimSize; rt++) {

          for (int mu=0;mu<nGamma;mu++) {
            sum[mu] = Zero();
          }

          int shmem_idx = nGamma*rt+shmem_base;
          for (int so=0;so < localSpatialVolume; so++) {
            ss = rt*localSpatialVolume+so;

            left   = coalescedRead(viewL_p[l_index][ss]);

            for (int mu = 0; mu < nGamma; mu++) {
              link_ahead = coalescedRead(viewGL_p[mu][ss]);

              SE=stencilR_p->GetEntry(ptype,2*mu,ss);
              if(SE->_is_local) { 
                shift_ahead = coalescedReadPermute(viewR_p[r_index][SE->_offset],ptype,SE->_permute);
              } else {
                shift_ahead = coalescedRead(bufRight_p[r_index*haloBuffRightSize+SE->_offset]);
              }
              acceleratorSynchronise();

              SE=stencilR_p->GetEntry(ptype,2*mu+1,ss);
              if(SE->_is_local) { 
                shift_behind = coalescedReadPermute(viewR_p[r_index][SE->_offset],ptype,SE->_permute);
              } else {
                shift_behind = coalescedRead(bufRight_p[r_index*haloBuffRightSize+SE->_offset]);
              }
              acceleratorSynchronise();

              SE=stencilG_p[mu].GetEntry(ptype,0,ss);
              if(SE->_is_local) { 
                link_behind = adj(coalescedReadPermute(viewGR_p[mu][SE->_offset],ptype,SE->_permute));
              } else {
                link_behind = adj(coalescedRead(bufGauge_p[offsetG_p[mu]+SE->_offset]));
              }
              acceleratorSynchronise();

              sum[mu] += innerProduct(left,link_ahead*shift_ahead+link_behind*shift_behind);
            }
            for (int mu=0;mu<nGamma;mu++) {
              coalescedWrite(shm_p[shmem_idx+mu],sum[mu]);
            }
          }
        }
      });

      accelerator_for2d(l_index,sizeL,r_index,sizeR,1,{
        ExtractBuffer<Scalar_s> extracted(simdSize);
        Scalar_s temp;
        int shmem_idx = reducedOrthogDimSize*nGamma*(l_index+sizeL*r_index);
        for (int i=0;i<nGamma*reducedOrthogDimSize;i++) {
          int mu = i%nGamma;
          int rt = i/nGamma;
          extract(shm_p[shmem_idx+i],extracted);

          for(int idx=0;idx<simdSize;idx +=2){

            // Calculate local time slice
            int lt = rt+idx/2*reducedOrthogDimSize;
            // Calculate global time slice
            int t_out = lt + pc*localOrthogDimSize;

            int ij_dx = mult_R*( r_index + sizeR*mult_L*( l_index + sizeL*( t_out + Nt*indexG_p[mu] ) ) );

            temp = extracted[idx] + extracted[idx+1];

            result_p[ij_dx]            += temp;
            result_p[ij_dx+sizeROut+1] -= temp;

            if (even) {
              result_p[ij_dx+1]          -= temp;
              result_p[ij_dx+sizeROut]   += temp;
              } else {
              result_p[ij_dx+1]          += temp;
              result_p[ij_dx+sizeROut]   -= temp;
            }
            acceleratorSynchronise();
          }
        }
      });
    }
  };

public:
  GridBase *_grid, *_cb_grid;

  LatticeGaugeField *_U;
  const std::vector<StagGamma::SpinTastePair> &_gammas;
  const std::vector<ComplexField> &_mom;

  std::vector<ComplexField> _stag_phase_E,_stag_phase_O;
  std::vector<LatticeColourMatrix> _Umu_E,_Umu_O;

  Vector<Integer> _gamma_indices_local,_gamma_indices_comm;

  A2AView<cobj> _view_gamma_E,_view_gamma_O,   _view_mom;
  A2AView<vColourMatrix>   _view_links_E, _view_links_O;
  A2AStencilView<vColourMatrix> _view_stencil_gauge_E, _view_stencil_gauge_O;
  std::vector<int> _shift_dirs, _shift_displacements;

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

    if (_gamma_indices_local.size() > 0) {
      buildLocalPhases();
    }

    for (int i = 0; i < _mom.size(); ++i)
    {
      _view_mom.addView(_mom[i]);
    }

    if (_gamma_indices_comm.size() > 0) {
      double t0 = usecond();
      buildGaugeLinks();
      double t1 = usecond();
      std::cout << GridLogPerformance << " MesonField one link timings: build link fields:" << (t1-t0)/1000 << "ms" << std::endl;   
    }
  }

  ~A2AWorkerMILC() {
    _view_links_E.closeViews();
    _view_links_O.closeViews();
    _view_stencil_gauge_E.closeViews();
    _view_stencil_gauge_O.closeViews();
    _view_gamma_E.closeViews();
    _view_gamma_O.closeViews();
    _view_mom.closeViews();

  }
  double getFlops() {
    return _flops;
  }
  void setFlops(flops) {_flops = flops; }

  void buildLocalPhases() {
    int nGamma_local = _gamma_indices_local.size();

    _stag_phase_E.resize(nGamma_local,_cb_grid);
    _stag_phase_O.resize(nGamma_local,_cb_grid);

    { // Set up staggered phases
      StagGamma spinTaste;
      ComplexField temp(_grid);
      int mu;
      for (int i = 0; i < nGamma_local; i++) {
        mu = _gamma_indices_local[i];

        temp = 1.0;

        spinTaste.setSpinTaste(_gammas[mu]);
        spinTaste.applyPhase(temp,temp); // store spin-taste phase

        pickCheckerboard(Even,_stag_phase_E[i],temp);
        pickCheckerboard(Odd,_stag_phase_O[i],temp);

        _view_gamma_E.addView(_stag_phase_E[i]);
        _view_gamma_O.addView(_stag_phase_O[i]);
      }
    }
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
    int size = _shift_dirs.size()/2;

    for (int i = 0; i < size; ++i)
    {
      auto ptr = std::move(std::unique_ptr<GaugeStencil>(new GaugeStencil(_cb_grid, 1, Even, {_shift_dirs[2*i]}, {-1})));
      _view_stencil_gauge_E.addStencil(ptr);
      _view_stencil_gauge_E.append(_Umu_E[i]);

      ptr = std::move(std::unique_ptr<GaugeStencil>(new GaugeStencil(_cb_grid, 1, Odd, {_shift_dirs[2*i]}, {-1})));
      _view_stencil_gauge_O.addStencil(ptr);
      _view_stencil_gauge_O.append(_Umu_O[i]);      
    }
    _view_stencil_gauge_E.openViews();
    _view_stencil_gauge_O.openViews();

    double t1=usecond();

    std::cout << GridLogPerformance << " MesonField one link timings: gauge comms:" << (t1-t0)/1000 << "ms" << std::endl;   

    for (int i = 0; i < nGamma_comm; ++i)
    {
      _view_links_E.addView(_Umu_E[i]);
      _view_links_O.addView(_Umu_O[i]);
    }
  }

public:
  template <typename TensorType> // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  void StagMesonFieldNoGlobalSum(TensorType &mat,
   const FermionField *lhs_wi_E, const FermionField *lhs_wi_O,
   const FermionField *rhs_vj_E, const FermionField *rhs_vj_O,
   int orthog_dir, double *t_kernel = nullptr);
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

  bool checkerL = lhs_wi_E[0].Grid()->_isCheckerBoarded;
  bool checkerR = rhs_vj_E[0].Grid()->_isCheckerBoarded;

  int sizeL = mat.dimension(3)/2;
  int sizeR = mat.dimension(4)/2;


  A2AView<vobj> viewRightE, viewRightO,viewLeftE, viewLeftO;
  A2AStencilView<vobj> viewStencilRightE, viewStencilRightO;

  if (_gamma_indices_comm.size() > 0) {
    auto ptr = std::move(std::unique_ptr<FermStencil>(new FermStencil(_cb_grid, _shift_dirs.size(), Even,
                                                              _shift_dirs, _shift_displacements)));
    viewStencilRightE.addStencil(ptr);

    ptr = std::move(std::unique_ptr<FermStencil>(new FermStencil(_cb_grid, _shift_dirs.size(), Odd,
                                                              _shift_dirs, _shift_displacements)));
    viewStencilRightO.addStencil(ptr);
    for (int i = 0; i < sizeR; ++i)
    {
      viewStencilRightE.append(rhs_vj_E[i]);
      viewStencilRightO.append(rhs_vj_O[i]);
    }
    viewStencilRightE.openViews();
    viewStencilRightO.openViews();
  }

  for (int i = 0; i < sizeR; ++i)
  {
    viewRightE.addView(rhs_vj_E[i]);
    viewRightO.addView(rhs_vj_O[i]);
  }

  for (int i = 0; i < sizeL; ++i)
  {
    viewLeftE.addView(lhs_wi_E[i]);
    viewLeftO.addView(lhs_wi_O[i]);
  }

  if (t_kernel) *t_kernel = -usecond();

  // Run any nonlocal gamma operators
  if (_gamma_indices_comm.size() > 0) {
    if (checkerL && checkerR) {
      A2ATaskHalfHalfOneLink task_e(viewLeftE, viewRightO, viewStencilRightO, _view_links_E, _view_links_O,
                                    _view_stencil_gauge_O, _gamma_indices_comm, _cb_grid, Even);
      task_e.execute(mat,orthog_dir);

      A2ATaskHalfHalfOneLink task_o(viewLeftO, viewRightE, viewStencilRightE, _view_links_O, _view_links_E,
                                    _view_stencil_gauge_E, _gamma_indices_comm, _cb_grid, Odd);
      task_o.execute(mat,orthog_dir);

      setFlops(task_e.getFlops());
    } else if (checkerL) {
    assert(0);
    } else if (checkerR) {
    assert(0);
    } else {
    assert(0);
    }
  }

  // Run any local gamma operators
  if (_gamma_indices_local.size() > 0) {

    if (checkerL && checkerR) {
      A2ATaskHalfHalfLocal task_e(viewLeftE, viewRightE, _view_gamma_E, _gamma_indices_local, _cb_grid, Even);
      task_e.execute(mat,orthog_dir);

      A2ATaskHalfHalfLocal task_o(viewLeftO, viewRightO, _view_gamma_O, _gamma_indices_local, _cb_grid, Odd);
      task_o.execute(mat,orthog_dir);
      
      setFlops(getFlops()+task_e.getFlops());
    } else if (checkerL) {
    assert(0);
    } else if (checkerR) {
    assert(0);
    } else {
    assert(0);
    }
  }
  viewStencilRightE.closeViews();
  viewStencilRightO.closeViews();
  viewRightE.closeViews();
  viewRightO.closeViews();
  viewLeftE.closeViews();
  viewLeftO.closeViews();
  if (t_kernel) *t_kernel += usecond();
}

NAMESPACE_END(Grid);
