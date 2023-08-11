#pragma once
#include <Grid/Grid_Eigen_Tensor.h>

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
  template<typename Vtype, typename obj>
  class A2AViewBase {
  public:
    Vector<Vtype> _view;

  public:
    Vtype &operator[](size_t i) { return _view[i];}
    
    int size() {return _view.size();}
    void reserve(int size) { _view.reserve(size); }

    virtual void closeViews() {
      for(int p=0;p<_view.size();p++)   _view[p].ViewClose();
      _view.resize(0);
    }
  };

  template<typename obj>
  class A2AFieldView: public A2AViewBase<LatticeView<obj>,obj> {
  public:
    void addView(const Lattice<obj> &field) {
      this->_view.push_back(field.View(AcceleratorRead));
    }
  };

  template<typename obj>
  class A2AStencilView: public A2AViewBase<CartesianStencilView<obj,obj,FImplParams>, obj>{
  protected:
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
      this->reserve(_stencils.size());
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
    virtual void closeViews() {
      for(int p=0;p<this->_view.size();p++)   this->_view[p].ViewClose();
      this->_view.resize(0);
      _buffer.resize(0);
      _offset.resize(0);
      _stencils.resize(0);
    }
  };

  class A2ATaskBase {
  protected:
    A2AFieldView<vobj> &_left, &_right;
    Vector<Integer> &_gamma_indices;
    GridBase *_grid;
    Vector<Coordinate> _i_coor_container;

  public:
    A2ATaskBase(A2AFieldView<vobj> &left, A2AFieldView<vobj> &right, Vector<Integer> &gammaIndices, GridBase *grid):
      _left(left),_right(right),_gamma_indices(gammaIndices),_grid(grid) {

      _i_coor_container.resize(grid->Nsimd(), Coordinate(grid->_ndimension));
      for(int p = 0; p < grid->Nsimd(); p++) {
        grid->iCoorFromIindex(_i_coor_container[p],p);
      }
    }
  };

  class A2ATaskHalfHalf: public A2ATaskBase {
  protected:
    bool _odd_shifts;
    int _cb;

    public:
      A2ATaskHalfHalf(A2AFieldView<vobj> &left, A2AFieldView<vobj> &right, Vector<Integer> &gammaIndices, GridBase *grid, int cb = Even, bool oddShifts = false):
      A2ATaskBase(left,right,gammaIndices,grid), _cb(cb), _odd_shifts(oddShifts) {}

  };

  class A2ATaskHalfHalfLocalNoMom: public A2ATaskHalfHalf {
  protected:
    A2AFieldView<cobj> &_gamma;

  public:
    A2ATaskHalfHalfLocalNoMom(A2AFieldView<vobj> &left, A2AFieldView<vobj> &right, A2AFieldView<cobj> &gamma, Vector<Integer> &gammaIndices, GridBase *grid, int cb = Even):
      A2ATaskHalfHalf(left,right,gammaIndices,grid,cb,false),_gamma(gamma)  {}

    double getFlops();

    template <typename MatType>
    void execute(MatType &result, int orthogDir);
  };

  class A2ATaskHalfHalfOneLinkNoMom: public A2ATaskHalfHalf {
  protected:
    A2AFieldView<vColourMatrix>   &_links_left,&_links_right;
    A2AStencilView<vobj> &_right_stencil;
    A2AStencilView<vColourMatrix> &_link_stencil;

  public:
    A2ATaskHalfHalfOneLinkNoMom(A2AFieldView<vobj> &left, A2AFieldView<vobj> &right, A2AStencilView<vobj> &rightStencil, A2AFieldView<vColourMatrix> &linksLeft,
                          A2AFieldView<vColourMatrix> &linksRight, A2AStencilView<vColourMatrix> &linkStencil, 
                          Vector<Integer> &gammaIndices, GridBase *grid, int cb = Even):
      A2ATaskHalfHalf(left,right,gammaIndices,grid,cb,true),_links_left(linksLeft),_links_right(linksRight),_link_stencil(linkStencil),_right_stencil(rightStencil) {}

    double getFlops();

    template <typename MatType>
    void execute(MatType &result, int orthogDir);
  };

public:
  GridBase *_grid, *_cb_grid;

  LatticeGaugeField *_U;
  const std::vector<StagGamma::SpinTastePair> &_gammas;
  const std::vector<ComplexField> &_mom;

  std::vector<ComplexField> _stag_phase_E,_stag_phase_O;
  std::vector<LatticeColourMatrix> _Umu_E,_Umu_O;

  Vector<Integer> _gamma_indices_local,_gamma_indices_comm;
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
  A2AWorkerMILC(GridBase *grid, const std::vector<StagGamma::SpinTastePair>& gammas, const std::vector<ComplexField> &mom, LatticeGaugeField* U = nullptr, GridBase *cbGrid = nullptr);

  ~A2AWorkerMILC();

  double getFlops() {
    return _flops;
  }
  void setFlops(double flops) {_flops = flops; }

  void buildLocalPhases();
  void buildGaugeLinks();

  template <typename TensorType> // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  void StagMesonFieldNoGlobalSum(TensorType &mat,
   const FermionField *lhs_wi_E, const FermionField *lhs_wi_O,
   const FermionField *rhs_vj_E, const FermionField *rhs_vj_O,
   int orthog_dir, double *t_kernel = nullptr);
};

NAMESPACE_END(Grid);