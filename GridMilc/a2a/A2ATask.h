/*
 * GridMilc/a2a/A2ATask.h — part of GridMilc (https://github.com/paboyle/Grid)
 *
 * All-to-all staggered meson-field contraction tasks (local + one-link).
 * Header-only; lifted from HadronsMILC. Self-contained via Grid's QCD core
 * umbrella.
 *
 * GridMilc is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 2 (or, at your option,
 * any later version). See COPYING/LICENSE in the top-level distribution.
 */
#pragma once

#include <Grid/GridQCDcore.h>
#include <GridMilc/a2a/A2AView.h>
#include <GridMilc/spin/StagGamma.h>
#include <GridMilc/a2a/StencilGather5d.h>

#ifndef MF_SUM_ARRAY_MAX
#define MF_SUM_ARRAY_MAX 16
#endif

NAMESPACE_BEGIN(Grid);

#define A2A_TYPEDEFS                                                           \
  typedef typename FImpl::SiteSpinor vobj;                                     \
  typedef typename FImpl::ComplexField ComplexField;                           \
  typedef typename FImpl::FermionField FermionField;                           \
  typedef typename ComplexField::vector_object cobj;                           \
  typedef typename FImpl::ImplParams FImplParams;                              \
  typedef CartesianStencil<vColourMatrix, vColourMatrix, FImplParams>          \
      GaugeStencil;                                                            \
  typedef CartesianStencilView<vColourMatrix, vColourMatrix, FImplParams>      \
      GaugeStencilView;                                                        \
  typedef typename vobj::scalar_type scalar_type;                              \
  typedef typename vobj::vector_type vector_type;                              \
  typedef iSinglet<scalar_type> Scalar_s;                                      \
  typedef iSinglet<vector_type> Scalar_v;                                      \
  typedef decltype(coalescedRead(Scalar_v())) calcScalar;                      \
  typedef decltype(coalescedRead(vobj())) calcSpinor;                          \
  typedef decltype(coalescedRead(vColourMatrix())) calcColourMatrix;           \
  typedef typename FImpl::StencilImpl FermStencil;                             \
  typedef typename FImpl::StencilView FermStencilView;                         \
  typedef LatticeView<vColourMatrix> GaugeView;                                \
  typedef LatticeView<cobj> ComplexView;                                       \
  typedef LatticeView<vobj> FermView;                                          \
  typedef typename A2ATaskBase<FImpl>::ContractType ContractType;              \
  typedef std::function<void(scalar_type *, cobj *)> SimdFunc;                 \
  typedef std::function<void(cobj *, int, int)> VectorFunc;

#define COMMON_VARS                                                            \
  int sizeL = this->_left_view->size();                                        \
  int sizeR = this->_right_view->size();                                       \
                                                                               \
  int orthogDir = this->_orthog_dir;                                           \
  const size_t simdSize = this->_grid->Nsimd();                                \
  const int reducedOrthogDimSize = this->_grid->_rdimensions[orthogDir];       \
                                                                               \
  const int nBlocks = this->_grid->_slice_nblock[orthogDir];                   \
                                                                               \
  const int localSpatialVolume = this->_grid->_ostride[orthogDir];             \
                                                                               \
  FermView *viewL_p = this->_left_view->getView();                             \
  FermView *viewR_p = this->_right_view->getView();                            \
  int localOrthogDimSize = this->_grid->_ldimensions[orthogDir];               \
                                                                               \
  int Nt = this->_grid->GlobalDimensions()[orthogDir];                         \
                                                                               \
  int pd = this->_grid->_processors[orthogDir];                                \
  int pc = this->_grid->_processor_coor[orthogDir];                            \
                                                                               \
  int orthogSimdSize = this->_grid->_simd_layout[orthogDir];                   \
  Coordinate *icoor_p = this->_i_coor_container_device;                        \
  Integer *ocoor_p = this->_o_coor_map_device;                                 \
  bool cbEven = this->_cb_left == Even;                                        \
  bool oddShifts = this->_odd_shifts;

template <typename FImpl> class A2ATaskBase {
public:
  GRID_SERIALIZABLE_ENUM(ContractType, undef, Full, 0, RightHalf, 1, LeftHalf,
                         2, BothHalf, 3);

  A2A_TYPEDEFS;

protected:
  std::shared_ptr<A2AFieldView<vobj>> _left_view, _right_view;

  std::vector<Coordinate> _i_coor_container;
  Coordinate *_i_coor_container_device;

  std::vector<Integer> _o_coor_map;
  Integer *_o_coor_map_device;

  GridBase *_grid, *_full_grid;

  ContractType _contract_type;

  bool _odd_shifts;
  int _orthog_dir;
  const int _cb_left;

public:
  A2ATaskBase(GridBase *grid, int orthogDir, int cb = Even)
      : _grid(grid), _full_grid(grid), _cb_left(cb), _orthog_dir(orthogDir),
        _odd_shifts(false), _contract_type(ContractType::undef) {

    _i_coor_container.resize(grid->Nsimd(), Coordinate(grid->_ndimension));
    for (int p = 0; p < grid->Nsimd(); p++) {
      grid->iCoorFromIindex(_i_coor_container[p], p);
    }

    size_t size = _i_coor_container.size() * sizeof(Coordinate);
    _i_coor_container_device = (Coordinate *)acceleratorAllocDevice(size);
    acceleratorCopyToDevice(_i_coor_container.data(), _i_coor_container_device,
                            size);
  }

  virtual ~A2ATaskBase() {
    acceleratorFreeDevice(_i_coor_container_device);
    if (_o_coor_map.size() > 0)
      acceleratorFreeDevice(_o_coor_map_device);
  }

  virtual double getFlops() = 0;

  // Populates `_o_coor_map` property with full grid indices
  void generateCoorMap() {

    assert(_grid->CheckerBoarded(_orthog_dir) != 1);

    if (_o_coor_map.size() == 0) {
      _o_coor_map.resize(_grid->oSites(), 0);
      _o_coor_map_device = (Integer *)acceleratorAllocDevice(
          _o_coor_map.size() * sizeof(Integer));

      int nBlocks = _grid->_slice_nblock[_orthog_dir];
      int vecsPerSlicePerBlock = _grid->_slice_block[_orthog_dir];
      int blockStride = _grid->_slice_stride[_orthog_dir];
      int rtStride = _grid->_ostride[_orthog_dir];
      int cb = _cb_left;

      thread_for(ss, _full_grid->oSites(), {
        int cbos;
        Coordinate coor;

        _full_grid->oCoorFromOindex(coor, ss);
        cbos = _grid->CheckerBoard(coor);

        if (cbos == cb) {
          int ssh = _grid->oIndex(coor);
          _o_coor_map[ssh] = ss;
        }
      });
      acceleratorCopyToDevice(_o_coor_map.data(), _o_coor_map_device,
                              _o_coor_map.size() * sizeof(Integer));
    }
  }

  // Updates object to compute meson field with new `left` vectors
  // Updated properties:
  // `_contract_type` - Updated to reflect `left` checkerboard state
  // `_left_view`     - Allocates views from `left` pointer parameter
  // `_grid`          - Sets to checkerboarded grid if `_left_view`
  //                    or `_right_view` are checkerboarded, otherwise set to
  //                    full grid
  virtual void setLeft(const FermionField *left, int size) {

    bool checkerL = left[0].Grid()->_isCheckerBoarded;

    // Toggle LeftHalf bit
    if (_contract_type == ContractType::undef) {
      _contract_type = checkerL ? ContractType::LeftHalf : ContractType::Full;
    } else {
      if (checkerL)
        _contract_type = _contract_type | ContractType::LeftHalf;
      else
        _contract_type = _contract_type & ContractType::RightHalf;
    }

    switch (_contract_type) {
    case ContractType::LeftHalf:
    case ContractType::BothHalf:
    case ContractType::Full:
      _grid = left[0].Grid();
    default:
      break;
    }

    if (checkerL)
      generateCoorMap();

    _left_view = std::make_shared<A2AFieldView<vobj>>();
    _left_view->openViews(left, size);
  }

  // Updates object to compute meson field with new `right` vectors.
  // See corresponding `setLeft` method.
  virtual void setRight(const FermionField *right, int size) {

    bool checkerR = right[0].Grid()->_isCheckerBoarded;

    // Toggle RightHalf bit
    if (_contract_type == ContractType::undef) {
      _contract_type = checkerR ? ContractType::RightHalf : ContractType::Full;
    } else {
      if (checkerR)
        _contract_type = _contract_type | ContractType::RightHalf;
      else
        _contract_type = _contract_type & ContractType::LeftHalf;
    }

    switch (_contract_type) {
    case ContractType::RightHalf:
    case ContractType::BothHalf:
    case ContractType::Full:
      _grid = right[0].Grid();
    default:
      break;
    }

    if (checkerR)
      generateCoorMap();

    _right_view = std::make_shared<A2AFieldView<vobj>>();
    _right_view->openViews(right, size);
  }

  // Updates object to compute meson field with `other._left_view` vectors.
  // Should only be used for mixed cb/full calculation.
  virtual void setLeft(A2ATaskBase<FImpl> &other) {
    assert(!(other.getType() & ContractType::LeftHalf));
    _left_view = other.getLeftView();

    _contract_type = other.getType();
  }

  virtual void setRight(A2ATaskBase<FImpl> &other) {
    assert(!(other.getType() & ContractType::RightHalf));
    _right_view = other.getRightView();

    _contract_type = other.getType();
  }

  std::shared_ptr<A2AFieldView<vobj>> getLeftView() { return _left_view; }
  std::shared_ptr<A2AFieldView<vobj>> getRightView() { return _right_view; }
  ContractType getType() { return _contract_type; }

  virtual int getNgamma() = 0;
  virtual void vectorSumHalf(cobj *a, int b, int c) = 0;
  virtual void vectorSumFull(cobj *a, int b, int c) = 0;
  virtual void vectorSumMixed(cobj *a, int b, int c) = 0;

  virtual void execute(scalar_type *result_p) {

    COMMON_VARS;

    VectorFunc vectorSum;
    SimdFunc simdSum;

    int multFact;

    switch (this->_contract_type) {
    case ContractType::BothHalf:
      vectorSum = [this](cobj *a, int b, int c) {
        this->vectorSumHalf(a, b, c);
      };
      simdSum = [this](scalar_type *a, cobj *b) { this->simdSumHalf(a, b); };
      multFact = 4;
      break;
    case ContractType::LeftHalf:
    case ContractType::RightHalf:
      vectorSum = [this](cobj *a, int b, int c) {
        this->vectorSumMixed(a, b, c);
      };
      simdSum = [this](scalar_type *a, cobj *b) { this->simdSumMixed(a, b); };
      multFact = 2;
      break;
    case ContractType::Full:
      vectorSum = [this](cobj *a, int b, int c) {
        this->vectorSumFull(a, b, c);
      };
      simdSum = [this](scalar_type *a, cobj *b) { this->simdSumFull(a, b); };
      multFact = 1;
      break;
    }

    int nGamma = this->getNgamma();
    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int MFrvol = gammaStride * nGamma;

    cobj *shm_p = (cobj *)acceleratorAllocDevice(MFrvol * sizeof(cobj));

    // Loop over gammas in batches of MF_SUM_ARRAY_MAX
    for (int mu = 0; mu < nGamma; mu += MF_SUM_ARRAY_MAX) {

      int nGammaBlock = std::min(nGamma - mu, MF_SUM_ARRAY_MAX);

      vectorSum(shm_p + mu * gammaStride, mu, nGammaBlock);
    }

    for (int mu = 0; mu < nGamma; mu++) {
      simdSum(result_p + mu * multFact * sizeR * sizeL * Nt,
              shm_p + mu * gammaStride);
    }

    acceleratorFreeDevice(shm_p);
  }

  // Sums over SIMD vectorized results stored in `shm` parameter.
  // Case: Neither `_left_view` nor `_right_view` are checkerboarded.
  void simdSumFull(scalar_type *result, cobj *shm) {

    COMMON_VARS;

    auto shm_p = shm;
    auto result_p = result;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, 1, {
      ExtractBuffer<Scalar_s> extracted(simdSize);
      scalar_type temp;

      int shmem_idx = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        extract(shm_p[shmem_idx + rt], extracted);

        for (int simdOffset = 0; simdOffset < orthogSimdSize; simdOffset++) {

          temp = scalar_type(0.0);
          for (int idx = 0; idx < simdSize; idx++) {
            if (icoor_p[idx][orthogDir] == simdOffset) {
              temp += TensorRemove(extracted[idx]);
            }
            acceleratorSynchronise();
          }

          // Calculate local time from reduced time
          int lt = rt + simdOffset * reducedOrthogDimSize;
          int gt = lt + pc * localOrthogDimSize;

          int ij_dx = r_index + sizeR * (l_index + sizeL * gt);

          result_p[ij_dx] = temp;
        }
      }
    });
  }

  // Sums over SIMD vectorized results stored in `shm` parameter.
  // Case: Either `_left_view` or `_right_view` is checkerboarded, not both.
  void simdSumMixed(scalar_type *result, cobj *shm) {

    COMMON_VARS;

    auto shm_p = shm;
    auto result_p = result;

    bool fullLeft = this->_contract_type == ContractType::RightHalf;

    int multRight = fullLeft ? 2 : 1;
    int multLeft = fullLeft ? 1 : 2;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, 1, {
      ExtractBuffer<Scalar_s> extracted(simdSize);
      scalar_type temp;

      int shmem_idx = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        extract(shm_p[shmem_idx + rt], extracted);

        for (int simdOffset = 0; simdOffset < orthogSimdSize; simdOffset++) {

          temp = scalar_type(0.0);
          for (int idx = 0; idx < simdSize; idx++) {
            if (icoor_p[idx][orthogDir] == simdOffset) {
              temp += TensorRemove(extracted[idx]);
            }
            acceleratorSynchronise();
          }

          // Calculate local time from reduced time
          int lt = rt + simdOffset * reducedOrthogDimSize;
          int gt = lt + pc * localOrthogDimSize;

          int ij_dx =
              multRight * (r_index + sizeR * multLeft * (l_index + sizeL * gt));

          result_p[ij_dx] += temp;

          if (fullLeft) {
            if (cbEven && oddShifts) {
              result_p[ij_dx + 1] -= temp;
            } else if (cbEven) {
              result_p[ij_dx + 1] += temp;
            } else if (oddShifts) {
              result_p[ij_dx + 1] += temp;
            } else {
              result_p[ij_dx + 1] -= temp;
            }
            acceleratorSynchronise();
          } else {
            if (cbEven && oddShifts) {
              result_p[ij_dx + sizeR] += temp;
            } else if (cbEven) {
              result_p[ij_dx + sizeR] += temp;
            } else if (oddShifts) {
              result_p[ij_dx + sizeR] -= temp;
            } else {
              result_p[ij_dx + sizeR] -= temp;
            }
            acceleratorSynchronise();
          }
          acceleratorSynchronise();
        }
      }
    });
  }

  // Sums over SIMD vectorized results stored in `shm` parameter.
  // Case: Both `_left_view` and `_right_view` are checkerboarded.
  void simdSumHalf(scalar_type *result, cobj *shm) {

    COMMON_VARS;

    auto shm_p = shm;
    auto result_p = result;

    int sizeROut = 2 * sizeR;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, 1, {
      ExtractBuffer<Scalar_s> extracted(simdSize);
      scalar_type temp;

      int shmem_idx = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        extract(shm_p[shmem_idx + rt], extracted);

        for (int simdOffset = 0; simdOffset < orthogSimdSize; simdOffset++) {

          temp = scalar_type(0.0);
          for (int idx = 0; idx < simdSize; idx++) {
            if (icoor_p[idx][orthogDir] == simdOffset) {
              temp += TensorRemove(extracted[idx]);
            }
            acceleratorSynchronise();
          }

          // Calculate local time and global time from reduced time
          int lt = rt + simdOffset * reducedOrthogDimSize;
          int gt = lt + pc * localOrthogDimSize;

          int ij_dx = 2 * (r_index + sizeR * 2 * (l_index + sizeL * gt));

          result_p[ij_dx] += temp;

          if (cbEven && oddShifts) {
            result_p[ij_dx + 1] -= temp;
            result_p[ij_dx + sizeROut + 1] -= temp;
            result_p[ij_dx + sizeROut] += temp;
          } else if (cbEven) {
            result_p[ij_dx + 1] += temp;
            result_p[ij_dx + sizeROut + 1] += temp;
            result_p[ij_dx + sizeROut] += temp;

          } else if (oddShifts) {
            result_p[ij_dx + 1] += temp;
            result_p[ij_dx + sizeROut + 1] -= temp;
            result_p[ij_dx + sizeROut] -= temp;
          } else {
            result_p[ij_dx + 1] -= temp;
            result_p[ij_dx + sizeROut + 1] += temp;
            result_p[ij_dx + sizeROut] -= temp;
          }
          acceleratorSynchronise();
        }
      }
    });
  }
};

template <typename FImpl> class A2ATaskLocal : public A2ATaskBase<FImpl> {
public:
  A2A_TYPEDEFS;

protected:
  std::vector<StagGamma::SpinTastePair> _gammas;
  std::vector<ComplexField> _phase;
  std::shared_ptr<A2AFieldView<cobj>> _phase_view;

public:
  A2ATaskLocal(GridBase *grid, int orthogDir, A2ATaskLocal<FImpl> &other,
               const std::vector<StagGamma::SpinTastePair> &gammas = {},
               int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb), _gammas(gammas) {
    _phase_view = other.getPhaseView();
  }

  A2ATaskLocal(GridBase *grid, int orthogDir,
               const std::vector<StagGamma::SpinTastePair> &gammas,
               int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb), _gammas(gammas) {

    int nGamma = _gammas.size();

    _phase.resize(nGamma, this->_full_grid);

    ComplexField temp(this->_full_grid);
    temp = 1.0;
    StagGamma spinTaste;

    _phase_view = std::make_shared<A2AFieldView<cobj>>();
    _phase_view->reserve(nGamma);

    for (int mu = 0; mu < nGamma; mu++) {

      spinTaste.setSpinTaste(_gammas[mu]);

      spinTaste.applyCoeffsAndPhase(_phase[mu], temp); // store spin-taste phase
    }
    _phase_view->openViews(_phase.data(), nGamma);
  }

  A2ATaskLocal(GridBase *grid, int orthogDir,
               const std::vector<ComplexField> &gammas, int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb) {

    int nGamma = gammas.size();

    _phase_view = std::make_shared<A2AFieldView<cobj>>();
    _phase_view->reserve(nGamma);
    _phase_view->openViews(gammas.data(), nGamma);
  }

  virtual ~A2ATaskLocal() {
    if (_phase_view)
      _phase_view->closeViews();
  }

  std::shared_ptr<A2AFieldView<cobj>> getPhaseView() { return _phase_view; }

  virtual int getNgamma() { return _phase_view->size(); }

  virtual double getFlops() {
    // One complex multiply takes 6 floating point ops (4 mult, 2 add)
    // --> complex inner product is 3 complex mult, 2 complex add = 3*6 + 2*2 =
    // 22 double precision floating ops

    // For each vector and at each lattice site:
    //  - one inner product
    //  - For each gamma and momentum
    //    - multiply by gamma phase
    //    - sum
    return (22.0 + (6.0 + 2.0) * (this->getNgamma()));
  }

  virtual void vectorSumHalf(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    ComplexView *viewG_p = this->_phase_view->getView() + mu_offset;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar gamma_phase, temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {

          ss = rt * localSpatialVolume + so;

          temp_site = innerProduct(coalescedRead(viewL_p[l_index][ss]),
                                   coalescedRead(viewR_p[r_index][ss]));

          for (int mu = 0; mu < nGamma; mu++) {
            gamma_phase = coalescedRead(viewG_p[mu][ocoor_p[ss]]);
            sum[mu] += gamma_phase * temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }

  virtual void vectorSumFull(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    ComplexView *viewG_p = this->_phase_view->getView() + mu_offset;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar gamma_phase, temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {

          ss = rt * localSpatialVolume + so;

          temp_site = innerProduct(coalescedRead(viewL_p[l_index][ss]),
                                   coalescedRead(viewR_p[r_index][ss]));

          for (int mu = 0; mu < nGamma; mu++) {
            gamma_phase = coalescedRead(viewG_p[mu][ss]);
            sum[mu] += gamma_phase * temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }

  virtual void vectorSumMixed(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    ComplexView *viewG_p = this->_phase_view->getView() + mu_offset;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    bool checkerL = this->_contract_type == ContractType::LeftHalf;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar gamma_phase, temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {

          ss = rt * localSpatialVolume + so;

          if (checkerL) {
            temp_site =
                innerProduct(coalescedRead(viewL_p[l_index][ss]),
                             coalescedRead(viewR_p[r_index][ocoor_p[ss]]));
          } else {
            temp_site =
                innerProduct(coalescedRead(viewL_p[l_index][ocoor_p[ss]]),
                             coalescedRead(viewR_p[r_index][ss]));
          }
          acceleratorSynchronise();

          for (int mu = 0; mu < nGamma; mu++) {
            gamma_phase = coalescedRead(viewG_p[mu][ocoor_p[ss]]);
            sum[mu] += gamma_phase * temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }
};

template <typename FImpl> class A2ATaskOnelink : public A2ATaskBase<FImpl> {
public:
  A2A_TYPEDEFS;

protected:
  const std::vector<StagGamma::SpinTastePair> &_gammas = {};
  std::vector<int> _shift_dirs, _shift_displacements;
  LatticeGaugeField *_U;

  std::vector<LatticeColourMatrix> _link, _link_shifted;
  std::shared_ptr<A2AFieldView<vColourMatrix>> _link_view, _link_shifted_view;
  std::shared_ptr<A2AStencilView<vobj, FImplParams>> _right_stencil_view;

public:
  A2ATaskOnelink(GridBase *grid, int orthogDir,
                 const std::vector<StagGamma::SpinTastePair> &gammas,
                 int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb), _gammas(gammas) {

    this->_odd_shifts = true;

    StagGamma spinTaste;

    for (int i = 0; i < _gammas.size(); i++) {

      spinTaste.setSpinTaste(_gammas[i]);

      // Harden: A2ATaskOnelink is correct ONLY for popcount==1 (a single
      // covariant hop). The original code silently truncated popcount>=2
      // gammas to their first active direction (the break below), producing
      // wrong results with no error. Route popcount>=2 to A2ATaskSpinTaste.
      int pc = StagGamma::popcountShift(spinTaste._spin, spinTaste._taste);
      if (pc != 1) {
        std::cerr << "A2ATaskOnelink requires popcount(spin^taste)==1, got "
                  << pc << " for gamma " << StagGamma::GetName(_gammas[i])
                  << "; use A2ATaskSpinTaste for popcount>=2" << std::endl;
        GridAbort();
      }

      int shift = (spinTaste._spin ^ spinTaste._taste);
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
    }
  }

  A2ATaskOnelink(GridBase *grid, int orthogDir, A2ATaskOnelink<FImpl> &other,
                 const std::vector<StagGamma::SpinTastePair> &gammas,
                 LatticeGaugeField *U, int cb = Even)
      : A2ATaskOnelink<FImpl>(grid, orthogDir, gammas, cb) {
    _U = U;
    _link_view = other.getLinkView();
    _link_shifted_view = other.getLinkView(true);
  }

  A2ATaskOnelink(GridBase *grid, int orthogDir,
                 const std::vector<StagGamma::SpinTastePair> &gammas,
                 LatticeGaugeField *U, int cb = Even)
      : A2ATaskOnelink<FImpl>(grid, orthogDir, gammas, cb) {

    _U = U;

    double t0 = usecond();

    int nGamma = _gammas.size();

    _link.resize(nGamma, _U->Grid());
    _link_shifted.resize(nGamma, _U->Grid());

    StagGamma spinTaste;

    for (int i = 0; i < nGamma; i++) {

      spinTaste.setSpinTaste(_gammas[i]);

      _link[i] = PeekIndex<LorentzIndex>(
          *_U,
          _shift_dirs[2 * i]); // Store full lattice links in shift direction

      _link_shifted[i] = Cshift(_link[i], _shift_dirs[2 * i], -1);

      spinTaste.applyCoeffsAndPhase(_link[i], _link[i]); // store spin-taste phase
      spinTaste.applyCoeffsAndPhase(_link_shifted[i],
                           _link_shifted[i]); // store spin-taste phase
    }
    _link_view = std::make_shared<A2AFieldView<vColourMatrix>>();
    _link_shifted_view = std::make_shared<A2AFieldView<vColourMatrix>>();

    _link_view->openViews(_link.data(), nGamma);
    _link_shifted_view->openViews(_link_shifted.data(), nGamma);

    double t1 = usecond();
    std::cout << GridLogPerformance
              << " MesonField onelink timings: build link fields + comms: "
              << (t1 - t0) / 1000 << " ms" << std::endl;
  }

  // Updates object to compute meson field with new `right` vectors.
  // See corresponding `setLeft` method.
  virtual void setRight(const FermionField *right, int size) {

    bool checkerR = right[0].Grid()->_isCheckerBoarded;

    // Toggle RightHalf bit
    if (this->_contract_type == ContractType::undef) {
      this->_contract_type =
          checkerR ? ContractType::RightHalf : ContractType::Full;
    } else {
      if (checkerR)
        this->_contract_type = this->_contract_type | ContractType::RightHalf;
      else
        this->_contract_type = this->_contract_type & ContractType::LeftHalf;
    }

    int cb = (this->_cb_left == Odd) ? Even : Odd;

    switch (this->_contract_type) {
    case ContractType::Full:
      cb = Even;
    case ContractType::BothHalf:
    case ContractType::RightHalf:
      this->_grid = right[0].Grid();
      break;
    case ContractType::LeftHalf:
      cb = Even;
    default:
      break;
    }

    if (checkerR)
      this->generateCoorMap();

    auto ptr = std::make_unique<FermStencil>(
        this->_grid, _shift_dirs.size(), cb, _shift_dirs, _shift_displacements);

    _right_stencil_view = std::make_shared<A2AStencilView<vobj, FImplParams>>();

    _right_stencil_view->addStencil(ptr);
    _right_stencil_view->openViews(right, size);

    this->_right_view = std::make_shared<A2AFieldView<vobj>>();
    this->_right_view->openViews(right, size);
  }

  virtual void setRight(A2ATaskBase<FImpl> &other) {
    assert(!(other.getType() & ContractType::RightHalf));
    this->_right_view = other.getRightView();
    _right_stencil_view =
        dynamic_cast<A2ATaskOnelink<FImpl> &>(other).getRightStencilView();

    this->_contract_type = other.getType();
  }

  virtual ~A2ATaskOnelink() {
    if (_right_stencil_view)
      _right_stencil_view->closeViews();
    if (_link_view)
      _link_view->closeViews();
    if (_link_shifted_view)
      _link_shifted_view->closeViews();
  }

  std::shared_ptr<A2AStencilView<vobj, FImplParams>> getRightStencilView() {
    return _right_stencil_view;
  }

  std::shared_ptr<A2AFieldView<vColourMatrix>>
  getLinkView(bool shifted = false) {
    if (shifted)
      return _link_shifted_view;
    else
      return _link_view;
  }

  virtual int getNgamma() { return _link_view->size(); }

  virtual double getFlops() {
    // matrix*vector = 3 inner products
    // current code:
    // innerProduct(left,link_ahead*shift_ahead+adj(link_behind)*shift_behind)
    //  = matrix*vector + matrix* vector --> inner product = 7 inner products
    //  and 1 complex sum

    // For each vector, each gamma, and at each lattice site:
    //  - one inner product
    //  - two su(3) matrix*vector ops
    //  - one complex sum
    return ((7 * 22.0 + 2.0) * this->getNgamma());
  }

  virtual void vectorSumHalf(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    // Pointers for accelerator indexing
    GaugeView *viewGL_p = this->_link_view->getView() + mu_offset;
    GaugeView *viewGR_p = this->_link_shifted_view->getView() + mu_offset;

    FermStencilView *stencilR_p =
        this->_right_stencil_view->getView(); // ket stencil for shifted links
    vobj *bufRight_p =
        this->_right_stencil_view
            ->getBuffer(); // buffer for shifted kets in halo region

    int haloBuffRightSize = this->_right_stencil_view->getOffset(1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      calcColourMatrix link_ahead, link_behind;
      calcSpinor left, shift_ahead, shift_behind;
      calcScalar sum[MF_SUM_ARRAY_MAX];

      StencilEntry *SE;
      int ptype, ss;
      int shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {
          ss = rt * localSpatialVolume + so;

          left = coalescedRead(viewL_p[l_index][ss]);

          for (int mu = 0; mu < nGamma; mu++) {
            int rightShiftOffset = 2 * (mu + mu_offset);
            link_ahead = coalescedRead(viewGL_p[mu][ocoor_p[ss]]);

            SE = stencilR_p->GetEntry(ptype, rightShiftOffset, ss);
            if (SE->_is_local) {
              shift_ahead = coalescedReadPermute(viewR_p[r_index][SE->_offset],
                                                 ptype, SE->_permute);
            } else {
              shift_ahead = coalescedRead(
                  bufRight_p[r_index * haloBuffRightSize + SE->_offset]);
            }
            acceleratorSynchronise();

            SE = stencilR_p->GetEntry(ptype, rightShiftOffset + 1, ss);
            if (SE->_is_local) {
              shift_behind = coalescedReadPermute(viewR_p[r_index][SE->_offset],
                                                  ptype, SE->_permute);
            } else {
              shift_behind = coalescedRead(
                  bufRight_p[r_index * haloBuffRightSize + SE->_offset]);
            }
            acceleratorSynchronise();

            link_behind = adj(coalescedRead(viewGR_p[mu][ocoor_p[ss]]));

            sum[mu] += innerProduct(left, link_ahead * shift_ahead +
                                              link_behind * shift_behind);
          }
          for (int mu = 0; mu < nGamma; mu++) {
            int shmem_idx = rt + shmem_base + mu * gammaStride;
            coalescedWrite(shm_p[shmem_idx], sum[mu]);
          }
        }
      }
    });
  }

  virtual void vectorSumFull(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    // Pointers for accelerator indexing
    GaugeView *viewGL_p = this->_link_view->getView() + mu_offset;
    GaugeView *viewGR_p = this->_link_shifted_view->getView() + mu_offset;

    FermStencilView *stencilR_p =
        this->_right_stencil_view->getView(); // ket stencil for shifted links
    vobj *bufRight_p =
        this->_right_stencil_view
            ->getBuffer(); // buffer for shifted kets in halo region

    int haloBuffRightSize = this->_right_stencil_view->getOffset(1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      calcColourMatrix link_ahead, link_behind;
      calcSpinor left, shift_ahead, shift_behind;
      calcScalar sum[MF_SUM_ARRAY_MAX];

      StencilEntry *SE;
      int ptype, ss;
      int shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {
          ss = rt * localSpatialVolume + so;

          left = coalescedRead(viewL_p[l_index][ss]);

          for (int mu = 0; mu < nGamma; mu++) {
            int rightShiftOffset = 2 * (mu + mu_offset);
            link_ahead = coalescedRead(viewGL_p[mu][ss]);

            SE = stencilR_p->GetEntry(ptype, rightShiftOffset, ss);
            if (SE->_is_local) {
              shift_ahead = coalescedReadPermute(viewR_p[r_index][SE->_offset],
                                                 ptype, SE->_permute);
            } else {
              shift_ahead = coalescedRead(
                  bufRight_p[r_index * haloBuffRightSize + SE->_offset]);
            }
            acceleratorSynchronise();

            SE = stencilR_p->GetEntry(ptype, rightShiftOffset + 1, ss);
            if (SE->_is_local) {
              shift_behind = coalescedReadPermute(viewR_p[r_index][SE->_offset],
                                                  ptype, SE->_permute);
            } else {
              shift_behind = coalescedRead(
                  bufRight_p[r_index * haloBuffRightSize + SE->_offset]);
            }
            acceleratorSynchronise();

            link_behind = adj(coalescedRead(viewGR_p[mu][ss]));

            sum[mu] += innerProduct(left, link_ahead * shift_ahead +
                                              link_behind * shift_behind);
          }
          for (int mu = 0; mu < nGamma; mu++) {
            int shmem_idx = rt + shmem_base + mu * gammaStride;
            coalescedWrite(shm_p[shmem_idx], sum[mu]);
          }
        }
      }
    });
  }

  virtual void vectorSumMixed(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    // Pointers for accelerator indexing
    GaugeView *viewGL_p = this->_link_view->getView() + mu_offset;
    GaugeView *viewGR_p = this->_link_shifted_view->getView() + mu_offset;

    FermStencilView *stencilR_p =
        this->_right_stencil_view->getView(); // ket stencil for shifted links
    vobj *bufRight_p =
        this->_right_stencil_view
            ->getBuffer(); // buffer for shifted kets in halo region

    int haloBuffRightSize = this->_right_stencil_view->getOffset(1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    bool checkerL = this->_contract_type == ContractType::LeftHalf;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      calcColourMatrix link_ahead, link_behind;
      calcSpinor left, shift_ahead, shift_behind;
      calcScalar sum[MF_SUM_ARRAY_MAX];

      StencilEntry *SE;
      int ptype, ss, ss_left, ss_right, ss_link;
      int shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {
          ss = rt * localSpatialVolume + so;

          ss_link = ocoor_p[ss];
          if (checkerL) {
            ss_left = ss;
            ss_right = ss_link;
          } else {
            ss_left = ss_link;
            ss_right = ss;
          }
          acceleratorSynchronise();

          left = coalescedRead(viewL_p[l_index][ss_left]);

          for (int mu = 0; mu < nGamma; mu++) {
            int rightShiftOffset = 2 * (mu + mu_offset);
            link_ahead = coalescedRead(viewGL_p[mu][ss_link]);

            SE = stencilR_p->GetEntry(ptype, rightShiftOffset, ss_right);
            if (SE->_is_local) {
              shift_ahead = coalescedReadPermute(viewR_p[r_index][SE->_offset],
                                                 ptype, SE->_permute);
            } else {
              shift_ahead = coalescedRead(
                  bufRight_p[r_index * haloBuffRightSize + SE->_offset]);
            }
            acceleratorSynchronise();

            SE = stencilR_p->GetEntry(ptype, rightShiftOffset + 1, ss_right);
            if (SE->_is_local) {
              shift_behind = coalescedReadPermute(viewR_p[r_index][SE->_offset],
                                                  ptype, SE->_permute);
            } else {
              shift_behind = coalescedRead(
                  bufRight_p[r_index * haloBuffRightSize + SE->_offset]);
            }
            acceleratorSynchronise();

            link_behind = adj(coalescedRead(viewGR_p[mu][ss_link]));

            sum[mu] += innerProduct(left, link_ahead * shift_ahead +
                                              link_behind * shift_behind);
          }
          for (int mu = 0; mu < nGamma; mu++) {
            int shmem_idx = rt + shmem_base + mu * gammaStride;
            coalescedWrite(shm_p[shmem_idx], sum[mu]);
          }
        }
      }
    });
  }
};
template <typename FImpl> class A2ATaskSpinTaste : public A2ATaskBase<FImpl> {
public:
  A2A_TYPEDEFS;

protected:
  std::vector<StagGamma::SpinTastePair> _gammas;
  LatticeGaugeField *_U;
  std::vector<FermionField> _transformed;
  std::shared_ptr<A2AFieldView<vobj>> _transformed_view;

public:
  A2ATaskSpinTaste(GridBase *grid, int orthogDir,
                   const std::vector<StagGamma::SpinTastePair> &gammas,
                   LatticeGaugeField *U, int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb), _gammas(gammas), _U(U) {

    // Validate uniform popcount. A2ATaskSpinTaste is a GENERAL primitive:
    // applyGamma + local inner product is correct for every popcount 0-4
    // (cross-checked against Local at pc0 and Onelink at pc1 in Test_a2a).
    // MesonField routes pc0->Local and pc1->Onelink only as a performance
    // optimization, NOT a correctness limit, so we do NOT abort on pc<2.
    // The uniformity check IS required: _odd_shifts is a single per-task
    // value derived from the (common) popcount parity, so mixed popcounts
    // would misroute Even/Odd right vectors. NOTE: _odd_shifts is bound into
    // COMMON_VARS (oddShifts) but is UNUSED by the SpinTaste kernels — the
    // E/O right-vector routing is governed by the WORKER's _odd_shifts
    // (A2AWorkerSpinTaste). It is retained here for COMMON_VARS parity and
    // because its value is correct if a future kernel reads it.
    if (!_gammas.empty()) {
      StagGamma spinTaste;
      spinTaste.setSpinTaste(_gammas[0]);
      int pc = StagGamma::popcountShift(spinTaste._spin, spinTaste._taste);
      this->_odd_shifts = (pc % 2 == 1);
      for (int i = 1; i < (int)_gammas.size(); i++) {
        spinTaste.setSpinTaste(_gammas[i]);
        int pci = StagGamma::popcountShift(spinTaste._spin, spinTaste._taste);
        if (pci != pc) {
          std::cerr
              << "A2ATaskSpinTaste requires uniform popcount; gamma 0 has "
                 "popcount "
              << pc << " but gamma " << i << " has popcount " << pci
              << std::endl;
          GridAbort();
        }
      }
    }
  }

  // _transformed_view is a shared_ptr; its destructor calls closeViews().
  // No explicit destructor body needed (matches A2ATaskLocal pattern).

  std::shared_ptr<A2AFieldView<vobj>> getTransformedView() {
    return _transformed_view;
  }

  // Construction-time nGamma is authoritative for SpinTaste: the transformed
  // view always holds exactly _gammas.size()*sizeRight entries (one block per
  // gamma), opened in setRight. Unlike Local/Onelink (which derive nGamma
  // from their phase/link view size), SpinTaste's view is gamma-indexed in a
  // fixed 1:1 correspondence with _gammas, so _gammas.size() cannot diverge.
  virtual int getNgamma() { return _gammas.size(); }

  virtual double getFlops() {
    // One inner product per (gamma, left vector, right vector, site).
    // No per-gamma phase multiply (phase is baked into the transformed
    // vectors by applyGamma). innerProduct = 22 double-precision ops.
    return (22.0 * this->getNgamma());
  }

  // Pre-transform right vectors via applyGamma, storing the result in
  // _transformed_view. The raw right vectors are kept in _right_view for
  // COMMON_VARS geometry queries (sizeR etc.).
  virtual void setRight(const FermionField *right, int size) {

    bool checkerR = right[0].Grid()->_isCheckerBoarded;

    // Contract-type logic (replicated from A2ATaskBase::setRight / Onelink)
    if (this->_contract_type == ContractType::undef) {
      this->_contract_type =
          checkerR ? ContractType::RightHalf : ContractType::Full;
    } else {
      if (checkerR)
        this->_contract_type =
            this->_contract_type | ContractType::RightHalf;
      else
        this->_contract_type =
            this->_contract_type & ContractType::LeftHalf;
    }

    switch (this->_contract_type) {
    case ContractType::RightHalf:
    case ContractType::BothHalf:
    case ContractType::Full:
      this->_grid = right[0].Grid();
    default:
      break;
    }

    if (checkerR)
      this->generateCoorMap();

    // Raw right view for COMMON_VARS (sizeR, viewR_p — unused in kernel)
    this->_right_view = std::make_shared<A2AFieldView<vobj>>();
    this->_right_view->openViews(right, size);

    // Pre-transform: psi_{j,gamma}(x) = applyGamma(gamma, v_j)(x).
    // applyGamma output is on right[j].Grid() (same GridBase* for Even/Odd
    // CB) with Checkerboard() flipped for odd popcount.  When _odd_shifts
    // is set, StagMesonField routes rhs_vj_O to the Even task; applyGamma
    // flips it back to Even-CB, matching the left vectors.  For even
    // popcount (_odd_shifts=false) the input is rhs_vj_E and the CB is
    // preserved.
    int nGamma = _gammas.size();
    // Clear transformed vectors retained from a previous setRight call. The
    // worker calls setRight once per j-block in production; without this,
    // _transformed would accumulate nGamma*size dead FermionFields per call —
    // a memory leak (applyGamma overwrites the live [0,nGamma*size) range so
    // results are correct, but the tail grows unbounded). clear() frees the
    // old elements while retaining capacity for reuse.
    _transformed.clear();
    _transformed.reserve(nGamma * size);
    for (int i = 0; i < nGamma * size; i++)
      _transformed.emplace_back(right[0].Grid());

    // _U is required for popcount>=1 (covariant shift). Fail early with a
    // clear message rather than dereferencing a null pointer in setGaugeField
    // (applyGamma asserts U!=nullptr, but only when shift!=0 — too late).
    if (_U == nullptr) {
      std::cerr << "A2ATaskSpinTaste::setRight: null gauge field (_U)"
                << std::endl;
      GridAbort();
    }
    StagGamma spinTaste;
    spinTaste.setGaugeField(*_U);
    for (int mu = 0; mu < nGamma; mu++) {
      spinTaste.setSpinTaste(_gammas[mu]);
      for (int j = 0; j < size; j++) {
        spinTaste.applyGamma(_transformed[mu * size + j], right[j]);
      }
    }

    _transformed_view = std::make_shared<A2AFieldView<vobj>>();
    _transformed_view->openViews(_transformed.data(), nGamma * size);
  }

  // Share transformed view from another task (Full right + CB left case).
  virtual void setRight(A2ATaskBase<FImpl> &other) {
    assert(!(other.getType() & ContractType::RightHalf));
    this->_right_view = other.getRightView();
    _transformed_view =
        dynamic_cast<A2ATaskSpinTaste<FImpl> &>(other).getTransformedView();
    this->_contract_type = other.getType();
  }

  // ---- Local inner-product kernels (no phase; phase is baked in) ----
  // NOTE: temp_site depends on gamma because viewT_p[r_gamma] changes per
  // gamma (the transformed right vector is gamma-specific). This is NOT the
  // same as A2ATaskLocal where the right vector is shared and only the phase
  // varies — here the inner product MUST be recomputed per gamma.

  virtual void vectorSumHalf(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    FermView *viewT_p = this->_transformed_view->getView();

    assert(orthogDir == Tdir);
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {
          ss = rt * localSpatialVolume + so;

          for (int mu = 0; mu < nGamma; mu++) {
            int r_gamma = (mu_offset + mu) * sizeR + r_index;
            temp_site = innerProduct(
                coalescedRead(viewL_p[l_index][ss]),
                coalescedRead(viewT_p[r_gamma][ss]));
            sum[mu] += temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }

  virtual void vectorSumFull(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    FermView *viewT_p = this->_transformed_view->getView();

    assert(orthogDir == Tdir);
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {
          ss = rt * localSpatialVolume + so;

          for (int mu = 0; mu < nGamma; mu++) {
            int r_gamma = (mu_offset + mu) * sizeR + r_index;
            temp_site = innerProduct(
                coalescedRead(viewL_p[l_index][ss]),
                coalescedRead(viewT_p[r_gamma][ss]));
            sum[mu] += temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }

  virtual void vectorSumMixed(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    FermView *viewT_p = this->_transformed_view->getView();

    assert(orthogDir == Tdir);
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    bool checkerL = this->_contract_type == ContractType::LeftHalf;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {
          ss = rt * localSpatialVolume + so;

          for (int mu = 0; mu < nGamma; mu++) {
            int r_gamma = (mu_offset + mu) * sizeR + r_index;
            if (checkerL) {
              temp_site = innerProduct(
                  coalescedRead(viewL_p[l_index][ss]),
                  coalescedRead(viewT_p[r_gamma][ocoor_p[ss]]));
            } else {
              temp_site = innerProduct(
                  coalescedRead(viewL_p[l_index][ocoor_p[ss]]),
                  coalescedRead(viewT_p[r_gamma][ss]));
            }
            acceleratorSynchronise();
            sum[mu] += temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }
};
///////////////////////////////////////////////////////////////////////////////
// spinTasteEndpoints: enumerate the 2^nActive sign-combination endpoints for a
// spin-taste (the lattice displacements s_ep whose covariant transports W_s the
// gauge chain builds). Pure host-side combinatorics; cheap. Each endpoint is
// one of the 2^popcount(spin^taste) +/-1 sign combos over the active directions
// (gmu[] order, X/Y/Z/T).
///////////////////////////////////////////////////////////////////////////////
// NOTE: these helpers are NOT templated on FImpl (they work with global
// colour-matrix types only), so they do NOT invoke A2A_TYPEDEFS. They sit
// inside the A2A_TYPEDEFS macro's textual scope without expanding it.
inline std::vector<Coordinate>
spinTasteEndpoints(const StagGamma &spinTaste) {
  int shift = spinTaste._spin ^ spinTaste._taste;
  std::array<int, 4> dirs;
  int nActive = 0;
  for (int j = 0; j < 4; j++)
    if (static_cast<int>(StagGamma::gmu[j]) & shift)
      dirs[nActive++] = j;

  std::vector<Coordinate> endpoints;
  for (int signs = 0; signs < (1 << nActive); signs++) {
    Coordinate s(4, 0);
    for (int i = 0; i < nActive; i++)
      s[dirs[i]] = (signs & (1 << i)) ? +1 : -1;
    endpoints.push_back(s);
  }
  return endpoints;
}

///////////////////////////////////////////////////////////////////////////////
// spinTasteGaugeChain: build W_s (symmetric gauge transporter + Follana phase)
// for a single spin-taste on the 5D grid and RETURN the 2^nActive endpoint
// transporters.
//
// W_s[ep](x) = phase(x) * scaling * Sum_over_orderings U-path(x -> x+s_ep),
// reproducing StagGamma::applyGamma's internal covariant structure. The chain
// is built ENTIRELY in 4D (U does not depend on RHS, so there is no need to
// replicate it across dim-5 during construction) and promoted to 5D only once
// per endpoint at the end. The Follana phase/scaling is applied in 4D via
// applyCoeffsAndPhase (grid-agnostic, local).
//
// Self-contained: calls spinTasteEndpoints for the sign combos (the endpoint
// enumeration logic lives in one place). The task move-merges the returned W_s
// into its flattened _wsFlat — no deep copy (each Lattice is moved, a pointer
// swap). const U: the build only reads U (PeekIndex per Lorentz component);
// Phase 3 passes a non-const LatticeGaugeField* which binds to this const*
// param without issue.
///////////////////////////////////////////////////////////////////////////////
inline std::vector<LatticeColourMatrix>
spinTasteGaugeChain(const LatticeGaugeField *U, const StagGamma &spinTaste,
                    GridCartesian *grid5d, GridCartesian *grid4d) {
  static constexpr int Nd = 4;

  std::vector<Coordinate> endpoints = spinTasteEndpoints(spinTaste);
  int nEndpoints = (int)endpoints.size();

  // Active directions (recomputed; cheap) for the permutation ordering. The
  // endpoint enumeration above derived the same dirs/nActive; recomputing here
  // keeps the two helpers independent and costs only a 4-iteration host loop.
  int shift = spinTaste._spin ^ spinTaste._taste;
  std::array<int, 4> dirs;
  int nActive = 0;
  for (int j = 0; j < 4; j++)
    if (static_cast<int>(StagGamma::gmu[j]) & shift)
      dirs[nActive++] = j;

  // Per-direction gauge links in 4D — peeked directly from *U (matches
  // A2ATaskOnelink's PeekIndex<LorentzIndex>(*_U, ...) pattern). W_s is a pure
  // gauge transporter (RHS-independent), so the covariant chain is built
  // ENTIRELY in 4D and promoted to 5D only once per endpoint at the end.
  // Building in 4D avoids Nsimd-fold replicated work/memory during the
  // 2^nActive x nActive! Cshift chain construction (the dominant setup cost).
  // The hops use the literal CovShiftForward/Backward forms
  // (Grid/qcd/utils/CovariantCshift.h): each carries exactly one halo Cshift,
  // inherent to the covariant chain — so no pre-shifted adjoint copy is hoisted
  // (hoisting it would not reduce the halo count, only obscure the asymmetry).
  std::vector<LatticeColourMatrix> Udir4d;
  Udir4d.reserve(Nd);
  for (int mu = 0; mu < Nd; mu++) {
    Udir4d.emplace_back(grid4d);
    Udir4d[mu] = PeekIndex<LorentzIndex>(*U, mu);
  }

  // Single 4D accumulator (reused per endpoint) + 4D chain scratch. Lattice has
  // no default ctor (requires a GridBase*), so they are constructed on grid4d.
  LatticeColourMatrix accumChain(grid4d), chain(grid4d);
  std::array<int, 4> perm;

  StagGamma st;
  st.setSpinTaste(spinTaste._spin, spinTaste._taste);

  std::vector<LatticeColourMatrix> Ws;
  Ws.reserve(nEndpoints);
  for (int ep = 0; ep < nEndpoints; ep++) {
    const Coordinate &s = endpoints[ep];

    // Symmetrized sum over all nActive! orderings (matches applyGamma) in 4D.
    // This is a one-time setup cost amortized across all (l,r) contractions.
    // A future optimization (design Future Optimization F2, v3.3b refinement)
    // is to NOT materialize W_s at all: pad the gauge field U and apply the
    // chain on-the-fly per site in the kernel, removing this 2^nActive x
    // nActive! Cshift materialization cost entirely.
    accumChain = Zero();
    for (int i = 0; i < nActive; i++)
      perm[i] = i;
    do {
      chain = 1.0;
      for (int i = 0; i < nActive; i++) {
        int dir = dirs[perm[i]];
        int sign = s[dir];
        if (sign > 0) {
          // CovShiftForward: Out(x) = U(x) * chain(x+dir) = U * Cshift(chain,+1)
          chain = Udir4d[dir] * Cshift(chain, dir, +1);
        } else {
          // CovShiftBackward: Out(x) = adj(U)(x-dir) * chain(x-dir)
          //                        = Cshift( adj(U) * chain, -1 )
          chain = Cshift(adj(Udir4d[dir]) * chain, dir, -1);
        }
      }
      accumChain = accumChain + chain;
    } while (std::next_permutation(perm.begin(), perm.begin() + nActive));

    // Apply Follana phase + (1/2)^n / n! scaling in 4D — applyCoeffsAndPhase is
    // local and grid-agnostic, so the S3 spatial-only caveat is moot here
    // (there is no dim-5 yet). Then promote the finished W_s to 5D (replicated
    // dim-5) once per endpoint. (applyCoeffsAndPhase is in-place safe: it reads
    // rhs into a temp before overwriting lhs.)
    st.applyCoeffsAndPhase(accumChain, accumChain);
    Ws.emplace_back(grid5d);
    promoteField5d(Ws[ep], accumChain, grid4d, grid5d);
  }
  return Ws;
}

///////////////////////////////////////////////////////////////////////////////
// A2ATaskSpinTasteStencil: 5D stencil-based spin-taste meson-field task.
//
// SIMD lanes (dim-5) = RHS vectors (batched Nsimd at a time). Spatial dims are
// simd=1, so gathers are trivial per-site offsets. L/W live on the 5D grid
// (replicated dim-5); ψ is padded once via PaddedCell::Exchange.
//
// Because dim-5 is the only SIMD-split dimension, the inherited simdSumFull
// (which reduces over time-SIMD lanes) is WRONG here; execute() is overridden
// to assemble the output mat directly.
///////////////////////////////////////////////////////////////////////////////
template <typename FImpl>
class A2ATaskSpinTasteStencil : public A2ATaskBase<FImpl> {
public:
  A2A_TYPEDEFS;

protected:
  std::vector<StagGamma::SpinTastePair> _gammas;
  LatticeGaugeField *_U;
  GridCartesian *_fullGrid; // 4D full grid (E/O joined)
  GridCartesian *_grid5d;   // 5D grid (simd={1,1,1,1,Nsimd})

  std::unique_ptr<PaddedCell> _cell5d;
  GridCartesian *_paddedGrid5d;

  // Flattened W_s on the 5D grid + its A2AFieldView (device-safe array).
  // Assembled by move-merging each gamma's spinTasteGaugeChain result (no deep
  // copy); no per-gamma chain object is retained.
  std::vector<LatticeColourMatrix> _wsFlat;
  std::shared_ptr<A2AFieldView<vColourMatrix>> _wsView;

  // Flattened per-(gamma,ep,osite) padded offset (deviceVector).
  deviceVector<int> _allOffsets;
  deviceVector<int> _nEpDev;    // nEndpoints per gamma
  deviceVector<int> _epStartDev;// cumulative global endpoint start per gamma
  int _nEpTotal;
  int _osites; // 5D spatial oSites (== grid5d oSites)

  // LHS promoted to 5D + view.
  std::vector<Lattice<vobj>> _lhs5d;
  std::shared_ptr<A2AFieldView<vobj>> _lhsView;
  int _sizeL;

  // Padded RHS 5D batches + view.
  std::vector<Lattice<vobj>> _paddedRight5d;
  std::shared_ptr<A2AFieldView<vobj>> _paddedRhsView;
  int _sizeR;
  int _nBatches;
  int _Nsimd;

public:
  A2ATaskSpinTasteStencil(GridCartesian *fullGrid, int orthogDir,
                          const std::vector<StagGamma::SpinTastePair> &gammas,
                          LatticeGaugeField *U, int cb = Even)
      : A2ATaskBase<FImpl>(fullGrid, orthogDir, cb),
        _gammas(gammas), _U(U), _fullGrid(fullGrid) {

    _grid5d = createGrid5d(fullGrid);
    _Nsimd = fullGrid->Nsimd();
    _osites = _grid5d->oSites();

    _cell5d = std::make_unique<PaddedCell>(1, _grid5d);
    _paddedGrid5d = _cell5d->grids.back();

    // Build W_s + offsets per gamma. spinTasteGaugeChain returns each gamma's
    // W_s, which are move-merged into wsFlatHost (no deep copy) — wsFlatHost
    // becomes the authoritative _wsFlat. spinTasteEndpoints gives the endpoint
    // list used both for nEndpoints and buildPaddedOffset5d.
    std::vector<LatticeColourMatrix> wsFlatHost;
    std::vector<int> allOffsetsHost;
    std::vector<int> nEpHost, epStartHost;
    _nEpTotal = 0;
    int epCum = 0;
    StagGamma spinTaste;

    for (int g = 0; g < (int)gammas.size(); g++) {
      spinTaste.setSpinTaste(gammas[g]);

      // Two pure helpers: enumerate this gamma's endpoints, build its W_s.
      std::vector<Coordinate> endpoints = spinTasteEndpoints(spinTaste);
      std::vector<LatticeColourMatrix> ws =
          spinTasteGaugeChain(U, spinTaste, _grid5d, fullGrid);

      nEpHost.push_back((int)endpoints.size());
      epStartHost.push_back(epCum);

      // Flatten this gamma's offsets into the global host table. buildPaddedOffset5d
      // returns a host std::vector<int> directly (no device round-trip); the
      // single upload into _allOffsets happens once, after the loop (S1).
      std::vector<int> epOffsets =
          buildPaddedOffset5d(_grid5d, _paddedGrid5d, 1, endpoints);
      allOffsetsHost.insert(allOffsetsHost.end(),
                            epOffsets.begin(), epOffsets.end());

      // Move-merge this gamma's W_s into the flat store (no deep copy: each
      // Lattice is moved, a pointer swap). wsFlatHost -> _wsFlat below.
      wsFlatHost.insert(wsFlatHost.end(),
                        std::make_move_iterator(ws.begin()),
                        std::make_move_iterator(ws.end()));

      epCum += (int)endpoints.size();
      _nEpTotal += (int)endpoints.size();
    }

    // wsFlatHost is the authoritative W_s store (built in place above); move it
    // into the member and open its view.
    _wsFlat = std::move(wsFlatHost);
    _wsView = std::make_shared<A2AFieldView<vColourMatrix>>();
    _wsView->openViews(_wsFlat.data(), _nEpTotal);

    // Upload bookkeeping deviceVectors.
    // acceleratorCopyToDevice requires 8-byte-aligned byte counts
    // (thread_bcopy assert). Pad to next even element count.
    auto upload = [](const std::vector<int> &h) {
      size_t nPad = ((h.size() + 1) / 2) * 2;
      std::vector<int> tmp(nPad, 0);
      std::copy(h.begin(), h.end(), tmp.begin());
      deviceVector<int> d(nPad);
      acceleratorCopyToDevice((void*)tmp.data(), (void*)d.data(), nPad * sizeof(int));
      return d;
    };
    _allOffsets = upload(allOffsetsHost);
    _nEpDev = upload(nEpHost);
    _epStartDev = upload(epStartHost);
  }

  // Reset _cell5d BEFORE deleting _grid5d: PaddedCell::~PaddedCell -> DeleteGrids
  // reads unpadded_grid (= _grid5d) members, so _grid5d must still be alive while
  // the cell is destroyed (else use-after-free).
  virtual ~A2ATaskSpinTasteStencil() {
    _cell5d.reset();
    delete _grid5d;
  }

  virtual int getNgamma() { return (int)_gammas.size(); }
  virtual double getFlops() { return 22.0 * _nEpTotal; }
  // The worker reads the 4D full grid (to size full-grid input temporaries);
  // _fullGrid is protected, so expose it (B2).
  virtual GridCartesian *getFullGrid() const { return _fullGrid; }

  // setLeft: promote each full-grid LHS vector directly to the 5D grid
  // (replicated dim-5). Does NOT call A2ATaskBase::setLeft — that opens
  // _left_view on the input and toggles _contract_type/_grid, none of which the
  // custom 5D execute() uses (it indexes _lhs5d/_lhsView instead), and avoids
  // the dangling-_left_view lifecycle hazard. grid4d is GridBase* (left[0].Grid())
  // — promoteField5d reads only GridBase-accessible members (B4/C1).
  virtual void setLeft(const FermionField *left, int size) {
    _sizeL = size;
    GridBase *grid4d = left[0].Grid();
    _lhs5d.clear(); _lhs5d.reserve(size);
    for (int l = 0; l < size; l++) {
      _lhs5d.emplace_back(_grid5d);
      promoteField5d(_lhs5d.back(), left[l], grid4d, _grid5d);
    }
    _lhsView = std::make_shared<A2AFieldView<vobj>>();
    _lhsView->openViews(_lhs5d.data(), size);
  }

  // setRight: pack full-grid RHS into Nsimd-sized 5D batches, pad via Exchange.
  // As with setLeft, does not call the base (B5/C1: grid4d is GridBase*).
  virtual void setRight(const FermionField *right, int size) {
    _sizeR = size;
    _nBatches = (size + _Nsimd - 1) / _Nsimd;

    _paddedRight5d.clear(); _paddedRight5d.reserve(_nBatches);
    GridBase *grid4d = right[0].Grid();
    for (int b = 0; b < _nBatches; b++) {
      int nVec = std::min(_Nsimd, size - b * _Nsimd);
      Lattice<vobj> rhs5d(_grid5d);
      rhs5d = Zero(); // zero unused lanes of the (possible) partial last batch
      packRhs5d(rhs5d, right + b * _Nsimd, nVec, grid4d, _grid5d);
      _paddedRight5d.emplace_back(_cell5d->Exchange(rhs5d));
    }
    _paddedRhsView = std::make_shared<A2AFieldView<vobj>>();
    _paddedRhsView->openViews(_paddedRight5d.data(), _nBatches);
  }

  //....................................................................
  // execute (OVERRIDE): the inherited simdSumFull is wrong for the 5D model
  // (it reduces over time-SIMD lanes; here dim-5 is RHS, not time). We run the
  // fused kernel into a cobj scratch buffer indexed by (mu, r_batch, l, rt),
  // then assemble the output mat by extracting each dim-5 lane to its RHS slot.
  //....................................................................
  virtual void execute(scalar_type *result_p) override {
    int nGamma = getNgamma();
    int orthogDir = this->_orthog_dir;
    int localSpatialVolume = _grid5d->_ostride[orthogDir];
    // In the 5D grid (simd={1,1,1,1,Nsimd}) every spatial dim — including the
    // orthog dir — has simd=1, so _ldimensions[orthogDir] == _rdimensions[orthogDir]:
    // there is no separate "reduced" orthog size here (unlike the 4D time-SIMD
    // base class). Use localOrthogDimSize throughout.
    int localOrthogDimSize = _grid5d->_ldimensions[orthogDir];
    int pc = _grid5d->_processor_coor[orthogDir];
    int Nt = _grid5d->GlobalDimensions()[orthogDir];

    // shm_p holds cobj (SIMD) elements: one shm_idx already encodes the Nsimd
    // RHS lanes (coalescedRead/Write handle the lanes), so the stride is over
    // (r_batch, l, rt) only — no _Nsimd factor (else scratch over-allocated xNsimd) (C4).
    int gammaStride = _nBatches * _sizeL * localOrthogDimSize;

    cobj *shm_p =
        (cobj *)acceleratorAllocDevice(gammaStride * nGamma * sizeof(cobj));
    accelerator_for(idx, gammaStride * nGamma, 1, { shm_p[idx] = Zero(); });

    // Gamma batching (MF_SUM_ARRAY_MAX gammas per kernel launch).
    for (int mu = 0; mu < nGamma; mu += MF_SUM_ARRAY_MAX) {
      int nGammaBlock = std::min(nGamma - mu, MF_SUM_ARRAY_MAX);
      vectorSumFull5d(shm_p, mu, nGammaBlock, localOrthogDimSize,
                      localSpatialVolume, gammaStride);
    }

    // Assemble: extract each dim-5 lane -> its RHS slot r, with gt = rt + pc*localT.
    // Per-lane scalar work (each lane -> one RHS slot r), so use the portable
    // acceleratorSIMTlane pattern (matches packRhs5d): on GPU one SIMT thread per
    // lane keeps the lanes busy (C8); on CPU a scalar lane loop.
    // NOTE: do NOT use accelerator_for2d here. Its macro injects a third lambda
    // param literally named `lane` (Accelerator.h:141), which collides with any
    // user-named second loop var, and its nsimd slot claims SIMD vectorization
    // this scalar extractLane body does not do. The 1d accelerator_for +
    // acceleratorSIMTlane is the idiomatic per-lane form.
    // Copy member variables to locals before the GPU lambda: on GPU the lambda
    // captures `this` by value but `this` is a CPU pointer — any `this->member`
    // dereference inside the kernel causes an illegal memory access.
    auto shm_p_a = shm_p;
    auto result_p_a = result_p;
    int sizeL_a    = _sizeL;
    int sizeR_a    = _sizeR;
    int nBatches_a = _nBatches;
    int Nsimd = _Nsimd;
    int nOuter = nGamma * sizeL_a * nBatches_a;
    accelerator_for(idx, nOuter, Nsimd, {
      int mu = idx / (sizeL_a * nBatches_a);
      int rem = idx % (sizeL_a * nBatches_a);
      int l = rem / nBatches_a;
      int r_batch = rem % nBatches_a;
#ifdef GRID_SIMT
      {
        int lane = acceleratorSIMTlane(Nsimd);
#else
      for (int lane = 0; lane < Nsimd; lane++) {
#endif
        int r = r_batch * Nsimd + lane;
        if (r < sizeR_a) {
          for (int rt = 0; rt < localOrthogDimSize; rt++) {
            int shm_idx = rt + localOrthogDimSize * (l + sizeL_a * r_batch) +
                          mu * gammaStride;
            cobj vals = shm_p_a[shm_idx];
            int gt = rt + pc * localOrthogDimSize; // T is simd=1: no time-SIMD reduction
            int mat_idx = mu * sizeR_a * sizeL_a * Nt +
                          r + sizeR_a * (l + sizeL_a * gt);
            result_p_a[mat_idx] = TensorRemove(extractLane(lane, vals));
          }
        }
#ifdef GRID_SIMT
      }
#else
      }
#endif
    });

    acceleratorFreeDevice(shm_p);
  }

  //....................................................................
  // vectorSumFull5d: the fused kernel. Device-safe: all field access via
  // A2AFieldView device pointers; bookkeeping via deviceVector<int>::data().
  // accelerator_for2d is portable (vectorized on CPU, SIMT on GPU).
  //....................................................................
  void vectorSumFull5d(cobj *shm_p, int mu_offset, int nGamma,
                       int localOrthogDimSize, int localSpatialVolume,
                       int gammaStride) {
    int sizeL = _sizeL;
    int nBatches = _nBatches;
    int Nsimd = _Nsimd;
    int osites = _osites;

    GaugeView *wsView_p = _wsView->getView();          // [globalEp][ss]
    FermView *lhsView_p = _lhsView->getView();         // [l][ss]
    FermView *rhsView_p = _paddedRhsView->getView();   // [r_batch][paddedSS]

    auto offsets_p = _allOffsets.data();   // [(epStart[g]+ep)*osites + ss]
    auto nEp_p = _nEpDev.data();
    auto epStart_p = _epStartDev.data();

    accelerator_for2d(l_index, sizeL, r_batch, nBatches, Nsimd, {
      calcScalar sum[MF_SUM_ARRAY_MAX];
      for (int rt = 0; rt < localOrthogDimSize; rt++) {
        for (int mu = 0; mu < nGamma; mu++)
          sum[mu] = Zero();

        for (int so = 0; so < localSpatialVolume; so++) {
          int ss = rt * localSpatialVolume + so;
          auto L = coalescedRead(lhsView_p[l_index][ss]); // broadcast (replicated)

          for (int mu = 0; mu < nGamma; mu++) {
            int g = mu_offset + mu;
            int nEp = nEp_p[g];
            int epStart = epStart_p[g];
            for (int ep = 0; ep < nEp; ep++) {
              int paddedSS = offsets_p[(epStart + ep) * osites + ss];
              auto psi = coalescedRead(rhsView_p[r_batch][paddedSS]); // per-lane RHS
              auto W = coalescedRead(wsView_p[epStart + ep][ss]);     // broadcast
              sum[mu] = sum[mu] + innerProduct(L, W * psi);
            }
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shm_idx = rt + localOrthogDimSize * (l_index + sizeL * r_batch) +
                        (mu_offset + mu) * gammaStride;
          coalescedWrite(shm_p[shm_idx], sum[mu]);
        }
      }
    });
  }

  virtual void vectorSumFull(cobj *, int, int) {
    assert(0 && "use execute(); 5D task overrides execute()");
  }
  virtual void vectorSumHalf(cobj *, int, int) {
    assert(0 && "stencil task is full-grid only");
  }
  virtual void vectorSumMixed(cobj *, int, int) {
    assert(0 && "stencil task is full-grid only");
  }
};

#undef A2A_TYPEDEFS
#undef COMMON_VARS

NAMESPACE_END(Grid);
