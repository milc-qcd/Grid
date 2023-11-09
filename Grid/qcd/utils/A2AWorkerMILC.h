#pragma once
#include <Grid/Grid_Eigen_Tensor.h>

#ifndef MF_SUM_ARRAY_MAX
#define MF_SUM_ARRAY_MAX 16
#endif

NAMESPACE_BEGIN(Grid);

#define A2A_TASK_COMMON \
  int sizeLOut = this->_left.size(); \
  int sizeROut = this->_right.size(); \
  int sizeR = sizeROut; \
  int sizeL = sizeLOut; \
  \
  const int nGamma = this->_gamma_indices.size(); \
  \
  const int simdSize             = this->_grid->Nsimd(); \
  const int reducedOrthogDimSize = this->_grid->_rdimensions[orthogDir]; \
  \
  const int nBlocks      = this->_grid->_slice_nblock[orthogDir]; \
  \
  const int localSpatialVolume     = this->_grid->_ostride[orthogDir]; \
  \
  FermView     *viewL_p    = this->_left.getView(); \
  FermView     *viewR_p    = this->_right.getView(); \
  int localOrthogDimSize   = this->_grid->_ldimensions[orthogDir]; \
  \
  int Nt     = this->_grid->GlobalDimensions()[orthogDir]; \
  \
  Integer* indexG_p = this->_gamma_indices_device; \
  \
  int pd = this->_grid->_processors[orthogDir]; \
  int pc = this->_grid->_processor_coor[orthogDir]; \
  \
  int localStride = 2; \
  int orthogSimdSize = this->_grid->_simd_layout[orthogDir]; \
  Coordinate *icoor_p = this->_i_coor_container_device;


#define A2A_TASK_HALF_COMMON \
  A2A_TASK_COMMON; \
  sizeLOut=2*sizeL; \
  sizeROut=2*sizeR; \
  int mult_L = 2; \
  int mult_R = 2; \
  bool even = this->_cb == Even; \
  bool oddShifts = this->_odd_shifts; \
  const int MFrvol = localStride*reducedOrthogDimSize *sizeL *sizeR * nGamma; \
  \
  cobj *shm_p = (cobj *)acceleratorAllocDevice(MFrvol*sizeof(cobj)); \
  \
  accelerator_for(r, MFrvol,1,{ \
    shm_p[r] = Zero(); \
  });  

#define A2A_KERNEL_COMPUTE_ONELINK_TIMEDIR_NOMOM \
accelerator_for2d(ls_index,localStride*sizeL,r_index,sizeR,simdSize,{ \
  \
  int site_offset = ls_index % localStride; \
  int l_index = ls_index/localStride; \
  \
  calcColourMatrix link_ahead, link_behind; \
  calcSpinor left, shift_ahead, shift_behind; \
  calcScalar sum[MF_SUM_ARRAY_MAX]; \
  \
  StencilEntry *SE; \
  int ptype, ss; \
  int shmem_base = localStride*nGamma*reducedOrthogDimSize*(l_index+sizeL*r_index)+site_offset; \
  \
  for (int rt=0;rt < reducedOrthogDimSize; rt++) { \
    \
    for (int mu=0;mu<nGamma;mu++) { \
      sum[mu] = Zero(); \
    } \
    \
    int shmem_idx = localStride*nGamma*rt+shmem_base; \
    for (int so=0;so < localSpatialVolume; so+=localStride) { \
      ss = rt*localSpatialVolume+so+site_offset; \
    \
      left   = coalescedRead(viewL_p[l_index][ss]); \
      \
      for (int mu = 0; mu < nGamma; mu++) { \
        link_ahead = coalescedRead(viewGL_p[mu][ss]); \
      \
        SE=stencilR_p->GetEntry(ptype,2*mu,ss); \
        if(SE->_is_local) {  \
          shift_ahead = coalescedReadPermute(viewR_p[r_index][SE->_offset],ptype,SE->_permute); \
        } else { \
          shift_ahead = coalescedRead(bufRight_p[r_index*haloBuffRightSize+SE->_offset]); \
        } \
        acceleratorSynchronise(); \
        \
        SE=stencilR_p->GetEntry(ptype,2*mu+1,ss); \
        if(SE->_is_local) { \
          shift_behind = coalescedReadPermute(viewR_p[r_index][SE->_offset],ptype,SE->_permute); \
        } else { \
          shift_behind = coalescedRead(bufRight_p[r_index*haloBuffRightSize+SE->_offset]); \
        } \
        acceleratorSynchronise(); \
        \
        SE=stencilG_p[mu].GetEntry(ptype,0,ss); \
        if(SE->_is_local) {  \
          link_behind = adj(coalescedReadPermute(viewGR_p[mu][SE->_offset],ptype,SE->_permute)); \
        } else { \
          link_behind = adj(coalescedRead(bufGauge_p[offsetG_p[mu]+SE->_offset])); \
        } \
        acceleratorSynchronise(); \
        \
        sum[mu] += innerProduct(left,link_ahead*shift_ahead+link_behind*shift_behind); \
      } \
      for (int mu=0;mu<nGamma;mu++) { \
        coalescedWrite(shm_p[shmem_idx+mu*localStride],sum[mu]); \
      } \
    } \
  } \
});

#define A2A_KERNEL_COMPUTE_LOCAL_TIMEDIR_NOMOM \
  accelerator_for2d(ls_index,localStride*sizeL,r_index,sizeR,simdSize,{ \
    \
    int site_offset = ls_index % localStride; \
    int l_index = ls_index/localStride; \
    \
    int ss, shmem_base = site_offset + localStride * nGamma * reducedOrthogDimSize*(l_index+sizeL*r_index); \
    calcScalar gamma_phase, temp_site, sum[MF_SUM_ARRAY_MAX]; \
    \
    for (int rt=0;rt < reducedOrthogDimSize; rt++) { \
      \
      for (int mu=0;mu<nGamma;mu++) { \
        sum[mu] = Zero(); \
      } \
        \
      int shmem_idx = localStride*nGamma*rt+shmem_base; \
      for (int so=0;so < localSpatialVolume; so+=localStride) { \
        ss = rt*localSpatialVolume+so+site_offset; \
        temp_site = innerProduct(coalescedRead(viewL_p[l_index][ss]),coalescedRead(viewR_p[r_index][ss])); \
        \
        for (int mu = 0; mu < nGamma; mu++) { \
          gamma_phase = coalescedRead(viewG_p[mu][ss]); \
          sum[mu] += gamma_phase*temp_site; \
        } \
      } \
      for (int mu=0;mu<nGamma;mu++) { \
        coalescedWrite(shm_p[shmem_idx+mu*localStride],sum[mu]); \
      } \
    } \
  });

#define A2A_KERNEL_IO_NOMOM \
  accelerator_for2d(li_index,sizeL*orthogSimdSize,r_index,sizeR,1,{ \
    int simdOffset = li_index % orthogSimdSize; \
    int l_index = li_index / orthogSimdSize; \
    ExtractBuffer<Scalar_s> extracted(simdSize); \
    cobj strideSum; \
    Scalar_s temp; \
    int shmem_idx = localStride*reducedOrthogDimSize*nGamma*(l_index+sizeL*r_index); \
    for (int i=0;i<nGamma*reducedOrthogDimSize;i++) { \
      \
      int mu = i%nGamma; \
      int rt = i/nGamma; \
      \
      strideSum = Zero(); \
      for (int j=0;j<localStride;j++) { \
        strideSum += shm_p[shmem_idx+i*localStride+j]; \
      } \
      extract(strideSum,extracted); \
      \
      temp = Scalar_s(0.0);    \
      for(int idx=0;idx<simdSize;idx++){ \
        if (icoor_p[idx][orthogDir] == simdOffset) { \
          temp +=  extracted[idx]; \
        } \
        acceleratorSynchronise(); \
      } \
      \
      int lt = rt+simdOffset*reducedOrthogDimSize; \
      int t_out = lt + pc*localOrthogDimSize; \
      \
      int ij_dx = mult_R*( r_index + sizeR*mult_L*( l_index + sizeL*( t_out + Nt*indexG_p[mu] ) ) ); \
        \
      result_p[ij_dx]            += temp; \
      \
      if (even && oddShifts) {        \
        result_p[ij_dx+1]          -= temp; \
        result_p[ij_dx+sizeROut+1] -= temp; \
        result_p[ij_dx+sizeROut]   += temp; \
      } else if (even) {          \
        result_p[ij_dx+1]          += temp; \
        result_p[ij_dx+sizeROut+1] += temp; \
        result_p[ij_dx+sizeROut]   += temp; \
        \
      } else if (oddShifts) { \
        result_p[ij_dx+1]          += temp; \
        result_p[ij_dx+sizeROut+1] -= temp; \
        result_p[ij_dx+sizeROut]   -= temp; \
      } else { \
        result_p[ij_dx+1]          -= temp; \
        result_p[ij_dx+sizeROut+1] += temp; \
        result_p[ij_dx+sizeROut]   -= temp; \
      } \
      acceleratorSynchronise(); \
    } \
  });

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
    std::vector<Vtype> _view;
    Vtype *_view_device;
    size_t _view_device_size;

  public:
    Vtype &operator[](size_t i) { return _view[i];}
    
    int size() {return _view.size();}

    void reserve(int size) { 
      _view.reserve(size); 
      _view_device_size = size*sizeof(Vtype);
      _view_device = (Vtype *)acceleratorAllocDevice(_view_device_size);
    }

    Vtype *getView() { return _view_device; }

    virtual void copyToDevice() {
      acceleratorCopyToDevice(_view.data(),_view_device,_view_device_size);
    }

    virtual void closeViews() {
      for(int p=0;p<_view.size();p++)   _view[p].ViewClose();

      if (_view_device_size > 0) {
        acceleratorFreeDevice(_view_device);
        _view_device_size = 0;
      }
    }
  };

  template<typename obj>
  class A2AFieldView: public A2AViewBase<LatticeView<obj>,obj> {
  public:
    void openViews(const Lattice<obj> *fields, int size) {
      this->reserve(size);
      for (int i = 0; i < size; i++) {
        this->_view.push_back(fields[i].View(AcceleratorRead));
      }
      this->copyToDevice();
    }
  };

  template<typename obj>
  class A2AStencilView: public A2AViewBase<CartesianStencilView<obj,obj,FImplParams>, obj>{
  protected:
    std::vector<std::unique_ptr<CartesianStencil<obj,obj,FImplParams> > > _stencils;

    obj *_buffer_device;
    size_t _buffer_device_size;

    std::vector<Integer> _offset;
    Integer *_offset_device;
    size_t _offset_device_size;

  public:
    obj *getBuffer() { return _buffer_device; }
    Integer *getOffset() { return _offset_device; }
    Integer & getOffset(int i) { return _offset[i]; }
    
    void addStencil(std::unique_ptr<CartesianStencil<obj,obj,FImplParams> > &stencil) {
      _stencils.push_back(std::move(stencil));
    }

    void openViews(const Lattice<obj> *fields, int size){

      createCommBuffer(fields,size);

      this->reserve(_stencils.size());
      for (auto &stencil: _stencils) {
        this->_view.push_back(stencil->View(AcceleratorRead));
      }
      this->copyToDevice();
    }

    void createCommBuffer(const Lattice<obj> *fields, int nFields) {

      Vector<obj> buffer;

      GridBase *grid = fields[0].Grid();
      SimpleCompressor<obj> compressor;
      int comm_buf_size;
      obj *buf_p;

      int j = 0;
      bool multipleStencils = _stencils.size() > 1;
      if (multipleStencils) assert(_stencils.size() == nFields);

      for (int i = 0; i < nFields; i++) {
        const auto &field = fields[i];
        auto &stencil = _stencils[j];

        stencil->HaloExchange(field,compressor);

        comm_buf_size = stencil->_unified_buffer_size;
        _offset.push_back(buffer.size());

        buffer.resize(buffer.size()+comm_buf_size);
        buf_p = &buffer[_offset.back()];

        if (comm_buf_size > 0) {
          obj *comm_buf_p = stencil->CommBuf();
          accelerator_for (k, comm_buf_size, 1, {
            buf_p[k] = comm_buf_p[k];
          });
        }

        if (multipleStencils) j++;
      }
      _offset_device_size = _offset.size()*sizeof(Integer);
      _offset_device = (Integer *)acceleratorAllocDevice(_offset_device_size);
      acceleratorCopyToDevice(_offset.data(),_offset_device,_offset_device_size);

      _buffer_device_size = buffer.size()*sizeof(obj);
      _buffer_device = (obj *)acceleratorAllocDevice(_buffer_device_size);
      acceleratorCopyToDevice(buffer.data(),_buffer_device,_buffer_device_size);

  }
    virtual void closeViews() {
      for(int p=0;p<this->_view.size();p++)   this->_view[p].ViewClose();

      _offset.resize(0);
      _stencils.resize(0);

      if (this->_view_device_size > 0) {
        acceleratorFreeDevice(this->_view_device);
        acceleratorFreeDevice(_buffer_device);
        acceleratorFreeDevice(_offset_device);

        this->_view_device_size = 0;
        _offset_device_size = 0;
        _buffer_device_size = 0;
      }

    }
  };

  class A2ATaskBase {
  protected:
    A2AFieldView<vobj> &_left, &_right;

    std::vector<Integer> &_gamma_indices;
    Integer *_gamma_indices_device;

    std::vector<Coordinate> _i_coor_container;
    Coordinate *_i_coor_container_device;

    GridBase *_grid;

  public:
    A2ATaskBase(A2AFieldView<vobj> &left, A2AFieldView<vobj> &right, std::vector<Integer> &gammaIndices, GridBase *grid):
      _left(left),_right(right),_gamma_indices(gammaIndices),_grid(grid) {

      _i_coor_container.resize(grid->Nsimd(), Coordinate(grid->_ndimension));
      for(int p = 0; p < grid->Nsimd(); p++) {
        grid->iCoorFromIindex(_i_coor_container[p],p);
      }

      size_t size = _i_coor_container.size()*sizeof(Coordinate);
      _i_coor_container_device = (Coordinate *)acceleratorAllocDevice(size);
      acceleratorCopyToDevice(_i_coor_container.data(),_i_coor_container_device,size);

      size = _gamma_indices.size()*sizeof(Integer);
      _gamma_indices_device = (Integer *)acceleratorAllocDevice(size);
      acceleratorCopyToDevice(_gamma_indices.data(),_gamma_indices_device,size);
    }

    ~A2ATaskBase() {
      acceleratorFreeDevice(_i_coor_container_device);
      acceleratorFreeDevice(_gamma_indices_device);
    }
  };

  class A2ATaskHalfHalf: public A2ATaskBase {
  protected:
    bool _odd_shifts;
    int _cb;

    public:
      A2ATaskHalfHalf(A2AFieldView<vobj> &left, A2AFieldView<vobj> &right, std::vector<Integer> &gammaIndices, GridBase *grid, int cb = Even, bool oddShifts = false):
      A2ATaskBase(left,right,gammaIndices,grid), _cb(cb), _odd_shifts(oddShifts) {}

  };

  class A2ATaskHalfHalfLocalNoMom: public A2ATaskHalfHalf {
  protected:
    A2AFieldView<cobj> &_gamma;

  public:
    A2ATaskHalfHalfLocalNoMom(A2AFieldView<vobj> &left, A2AFieldView<vobj> &right, A2AFieldView<cobj> &gamma, std::vector<Integer> &gammaIndices, GridBase *grid, int cb = Even):
      A2ATaskHalfHalf(left,right,gammaIndices,grid,cb,false),_gamma(gamma)  {}

    double getFlops() {
      // One complex multiply takes 6 floating point ops (4 mult, 2 add) 
      // --> complex inner product is 3 complex mult, 2 complex add = 3*6 + 2*2 = 22 double precision floating ops

      // For each vector and at each lattice site:
      //  - one inner product
      //  - For each gamma and momentum
      //    - multiply by gamma phase
      //    - multiply by momentum phase
      //    - sum
      return (22.0+(6.0+2.0)*(this->_gamma_indices.size()));
    }

    void execute(Scalar_s *result_p, int orthogDir) {

      A2A_TASK_HALF_COMMON;

      ComplexView  *viewG_p    = this->_gamma.getView();

      assert(orthogDir == Tdir); // This kernel assumes lattice is coalesced over time slices
      assert(nBlocks == 1);

      A2A_KERNEL_COMPUTE_LOCAL_TIMEDIR_NOMOM;

      A2A_KERNEL_IO_NOMOM
    }
  };

  class A2ATaskHalfHalfOneLinkNoMom: public A2ATaskHalfHalf {
  protected:
    A2AFieldView<vColourMatrix>   &_links_left,&_links_right;
    A2AStencilView<vobj> &_right_stencil;
    A2AStencilView<vColourMatrix> &_link_stencil;

  public:
    A2ATaskHalfHalfOneLinkNoMom(A2AFieldView<vobj> &left, A2AFieldView<vobj> &right, A2AStencilView<vobj> &rightStencil, A2AFieldView<vColourMatrix> &linksLeft,
                          A2AFieldView<vColourMatrix> &linksRight, A2AStencilView<vColourMatrix> &linkStencil, 
                          std::vector<Integer> &gammaIndices, GridBase *grid, int cb = Even):
      A2ATaskHalfHalf(left,right,gammaIndices,grid,cb,true),_links_left(linksLeft),_links_right(linksRight),_link_stencil(linkStencil),_right_stencil(rightStencil) {}

    double getFlops() {
      // matrix*vector = 3 inner products
      // current code: innerProduct(left,link_ahead*shift_ahead+adj(link_behind)*shift_behind)
      //  = matrix*vector + matrix* vector --> inner product = 7 inner products and 1 complex sum

      // For each vector, each gamma, and at each lattice site:
      //  - one inner product
      //  - two su(3) matrix*vector ops
      //  - one complex sum
      return ((7*22.0+2.0)*this->_gamma_indices.size());
    }

    void execute(Scalar_s *result_p, int orthogDir) {

      A2A_TASK_HALF_COMMON;

      assert(orthogDir == Tdir); // This kernel assumes lattice is coalesced over time slices
      assert(nBlocks == 1);

      // Pointers for accelerator indexing
      GaugeView *viewGL_p   = this->_links_left.getView();
      GaugeView *viewGR_p   = this->_links_right.getView();
      FermStencilView  *stencilR_p = this->_right_stencil.getView(); // Gauge stencil for shifted links

      GaugeStencilView *stencilG_p = this->_link_stencil.getView(); // Gauge stencil for shifted links

      vobj          *bufRight_p = this->_right_stencil.getBuffer(); // buffer for shifted kets in halo region
      vColourMatrix *bufGauge_p = this->_link_stencil.getBuffer(); // buffer for shifted links in halo region

      Integer *offsetG_p = this->_link_stencil.getOffset(); // buffer offsets for shifted links in halo region
      int haloBuffRightSize = this->_right_stencil.getOffset(1);

      A2A_KERNEL_COMPUTE_ONELINK_TIMEDIR_NOMOM;

      A2A_KERNEL_IO_NOMOM;
    }
  };

public:
  GridBase *_grid, *_cb_grid;

  LatticeGaugeField *_U;
  const std::vector<StagGamma::SpinTastePair> &_gammas;
  const std::vector<ComplexField> &_mom;

  std::vector<ComplexField> _stag_phase_E,_stag_phase_O;
  std::vector<LatticeColourMatrix> _Umu_E,_Umu_O;

  std::vector<Integer> _gamma_indices_local,_gamma_indices_comm;
  std::vector<int> _shift_dirs, _shift_displacements;
  double _flops;

  A2AFieldView<vobj> _view_right_E, _view_right_O, _view_left_E, _view_left_O;
  A2AFieldView<cobj> _view_gamma_E,_view_gamma_O,   _view_mom;
  A2AFieldView<vColourMatrix>   _view_links_E, _view_links_O;
  A2AStencilView<vobj> _view_stencil_right_E, _view_stencil_right_O;
  A2AStencilView<vColourMatrix> _view_stencil_gauge_E, _view_stencil_gauge_O;

private:
  FermionField *_r_addr_E, *_l_addr_E;
public:
  A2AWorkerMILC() = delete;
  A2AWorkerMILC(GridBase *grid, const std::vector<StagGamma::SpinTastePair>& gammas, const std::vector<ComplexField> &mom, LatticeGaugeField* U = nullptr, GridBase *cbGrid = nullptr)
  : _grid(grid), _gammas(gammas), _mom(mom), _U(U), _cb_grid(cbGrid) {

    _l_addr_E = nullptr;
    _r_addr_E = nullptr;

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

    _view_mom.openViews(_mom.data(),_mom.size());

    if (_gamma_indices_comm.size() > 0) {
      double t0 = usecond();
      buildGaugeLinks();
      double t1 = usecond();
      std::cout << GridLogPerformance << " MesonField one link timings: build link fields:" << (t1-t0)/1000 << "ms" << std::endl;   
    }
  }

  ~A2AWorkerMILC() {
    _view_left_E.closeViews();
    _view_left_O.closeViews();
    _view_right_E.closeViews();
    _view_right_O.closeViews();
    _view_stencil_right_E.closeViews();
    _view_stencil_right_O.closeViews();
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
  void setFlops(double flops) {_flops = flops; }

  void buildLocalPhases() {
    int nGamma_local = _gamma_indices_local.size();

    _stag_phase_E.resize(nGamma_local,_cb_grid);
    _stag_phase_O.resize(nGamma_local,_cb_grid);

    { // Set up staggered phases
      StagGamma spinTaste;
      ComplexField temp(_grid);
      int mu;

      _view_gamma_E.reserve(nGamma_local);
      _view_gamma_O.reserve(nGamma_local);
      for (int i = 0; i < nGamma_local; i++) {
        mu = _gamma_indices_local[i];

        temp = 1.0;

        spinTaste.setSpinTaste(_gammas[mu]);
        spinTaste.applyPhase(temp,temp); // store spin-taste phase

        pickCheckerboard(Even,_stag_phase_E[i],temp);
        pickCheckerboard(Odd,_stag_phase_O[i],temp);

      }
      _view_gamma_E.openViews(_stag_phase_E.data(),nGamma_local);
      _view_gamma_O.openViews(_stag_phase_O.data(),nGamma_local);
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

      ptr = std::move(std::unique_ptr<GaugeStencil>(new GaugeStencil(_cb_grid, 1, Odd, {_shift_dirs[2*i]}, {-1})));
      _view_stencil_gauge_O.addStencil(ptr);
    }
    _view_stencil_gauge_E.openViews(_Umu_E.data(),nGamma_comm);
    _view_stencil_gauge_O.openViews(_Umu_O.data(),nGamma_comm);
    _view_links_E.openViews(_Umu_E.data(),nGamma_comm);
    _view_links_O.openViews(_Umu_O.data(),nGamma_comm);

    double t1=usecond();

    std::cout << GridLogPerformance << " MesonField one link timings: gauge comms:" << (t1-t0)/1000 << "ms" << std::endl;   

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

  int sizeL = mat.dimension(3);
  int sizeR = mat.dimension(4);
  bool checkerL = lhs_wi_E[0].Grid()->_isCheckerBoarded;
  bool checkerR = rhs_vj_E[0].Grid()->_isCheckerBoarded;

  if (checkerL) sizeL /= 2;
  if (checkerR) sizeR /= 2;

  size_t matSize = mat.size()*sizeof(Scalar_s);
  Scalar_s *matDevice = (Scalar_s *)acceleratorAllocDevice(matSize);

  accelerator_for(i, mat.size(), 1, {
    matDevice[i] = Zero();
  })

  double t0=usecond();

  if (_l_addr_E != lhs_wi_E) {
    _l_addr_E = const_cast<FermionField*>(lhs_wi_E);
    _view_left_E.closeViews();
    _view_left_O.closeViews();

    _view_left_E.openViews(lhs_wi_E,sizeL);
    _view_left_O.openViews(lhs_wi_O,sizeL);

  }

  if (_r_addr_E != rhs_vj_E) {
    _r_addr_E = const_cast<FermionField*>(rhs_vj_E);
    _view_right_E.closeViews();
    _view_right_O.closeViews();
    _view_stencil_right_E.closeViews();
    _view_stencil_right_O.closeViews();

    if (_gamma_indices_comm.size() > 0) {
      auto ptr = std::move(std::unique_ptr<FermStencil>(new FermStencil(_cb_grid, _shift_dirs.size(), Even,
                                                                _shift_dirs, _shift_displacements)));
      _view_stencil_right_E.addStencil(ptr);

      ptr = std::move(std::unique_ptr<FermStencil>(new FermStencil(_cb_grid, _shift_dirs.size(), Odd,
                                                                _shift_dirs, _shift_displacements)));
      _view_stencil_right_O.addStencil(ptr);

      _view_stencil_right_E.openViews(rhs_vj_E,sizeR);
      _view_stencil_right_O.openViews(rhs_vj_O,sizeR);
    }

    _view_right_E.openViews(rhs_vj_E,sizeR);
    _view_right_O.openViews(rhs_vj_O,sizeR);
  }
  double t1=usecond();
  std::cout << GridLogPerformance << " MesonField timings: left/right views+comms:" << (t1-t0)/1000 << "ms" << std::endl;   


  if (t_kernel) *t_kernel = -usecond();

  setFlops(0.0);
  // Run any nonlocal gamma operators
  if (_gamma_indices_comm.size() > 0) {
    if (checkerL && checkerR) {
      A2ATaskHalfHalfOneLinkNoMom task_e(_view_left_E, _view_right_O, _view_stencil_right_O, _view_links_E, _view_links_O,
                                    _view_stencil_gauge_O, _gamma_indices_comm, _cb_grid, Even);
      task_e.execute(matDevice,orthog_dir);

      A2ATaskHalfHalfOneLinkNoMom task_o(_view_left_O, _view_right_E, _view_stencil_right_E, _view_links_O, _view_links_E,
                                    _view_stencil_gauge_E, _gamma_indices_comm, _cb_grid, Odd);
      task_o.execute(matDevice,orthog_dir);

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
      A2ATaskHalfHalfLocalNoMom task_e(_view_left_E, _view_right_E, _view_gamma_E, _gamma_indices_local, _cb_grid, Even);
      task_e.execute(matDevice,orthog_dir);

      A2ATaskHalfHalfLocalNoMom task_o(_view_left_O, _view_right_O, _view_gamma_O, _gamma_indices_local, _cb_grid, Odd);
      task_o.execute(matDevice,orthog_dir);

      setFlops(getFlops()+task_e.getFlops());
    } else if (checkerL) {
    assert(0);
    } else if (checkerR) {
    assert(0);
    } else {
    assert(0);
    }
  }
  if (t_kernel) *t_kernel += usecond();

  acceleratorCopyFromDevice(matDevice,mat.data(),matSize);
  acceleratorFreeDevice(matDevice);
}

NAMESPACE_END(Grid);

#undef A2A_TASK_COMMON
#undef A2A_TASK_HALF_COMMON
#undef A2A_KERNEL_COMPUTE_ONELINK_TIMEDIR_NOMOM
#undef A2A_KERNEL_COMPUTE_LOCAL_TIMEDIR_NOMOM
#undef A2A_KERNEL_IO_NOMOM
