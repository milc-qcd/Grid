/*
 * GridMilc/a2a/A2AWorker.h — part of GridMilc (https://github.com/paboyle/Grid)
 *
 * All-to-all staggered meson-field worker: top-level StagMesonField entry
 * point driving the local / one-link tasks. Header-only; lifted from
 * HadronsMILC. Self-contained via Grid's QCD core umbrella + Eigen Tensor.
 *
 * GridMilc is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 2 (or, at your option,
 * any later version). See COPYING/LICENSE in the top-level distribution.
 */
#pragma once

#include <Grid/GridQCDcore.h>
#include <Grid/Grid_Eigen_Tensor.h>
#include <GridMilc/a2a/A2ATask.h>
#include <GridMilc/spin/StagGamma.h>

NAMESPACE_BEGIN(Grid);

template <typename FImpl> class A2AWorkerBase {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;

public:
  GridBase *_grid, *_cb_grid;

  double _flops, _t_kernel, _t_gsum;

  A2ATaskBase<FImpl> *_task_e, *_task_o;

  const FermionField *_l_addr, *_r_addr;

  scalar_type *_cache_device;
  size_t _cache_bytes = 0;

  bool _odd_shifts{false};

public:
  A2AWorkerBase() = delete;
  A2AWorkerBase(GridBase *grid)
      : _grid(grid), _l_addr(nullptr), _r_addr(nullptr) {}

  void resetCache() { _l_addr = nullptr; _r_addr = nullptr; }

  virtual ~A2AWorkerBase() {
    if (_cache_bytes != 0) {
      acceleratorFreeDevice(_cache_device);
    }
    delete _task_e;
    delete _task_o;
  }

public:
  template <typename TensorType> // output: rank 5 tensor, e.g.
                                 // Eigen::Tensor<ComplexD, 5>
  void StagMesonField(TensorType &mat, const FermionField *lhs_wi_E,
                      const FermionField *lhs_wi_O,
                      const FermionField *rhs_vj_E,
                      const FermionField *rhs_vj_O);

  void setFlops(double flops) { _flops = flops; }
  double getFlops() { return _flops; }
};

template <typename FImpl> class A2AWorkerLocal : public A2AWorkerBase<FImpl> {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;

public:
  A2AWorkerLocal() = delete;
  A2AWorkerLocal(GridBase *grid, const std::vector<ComplexField> &mom,
                 const std::vector<StagGamma::SpinTastePair> &gammas,
                 int orthogDir)
      : A2AWorkerBase<FImpl>(grid) {
    this->_odd_shifts = false;
    if (mom.size()) {
      assert(0);
    } else {
      this->_task_e = new A2ATaskLocal<FImpl>(grid, orthogDir, gammas, Even);
      this->_task_o = new A2ATaskLocal<FImpl>(
          grid, orthogDir, dynamic_cast<A2ATaskLocal<FImpl> &>(*this->_task_e),
          gammas, Odd);
    }
  }
  A2AWorkerLocal(GridBase *grid, const std::vector<ComplexField> &mom,
                 const std::vector<ComplexField> &Amu, int orthogDir)
      : A2AWorkerBase<FImpl>(grid) {
    this->_odd_shifts = false;
    if (mom.size()) {
      assert(0);
    } else {
      this->_task_e = new A2ATaskLocal<FImpl>(grid, orthogDir, Amu, Even);
      this->_task_o = new A2ATaskLocal<FImpl>(
          grid, orthogDir, dynamic_cast<A2ATaskLocal<FImpl> &>(*this->_task_e),
          {}, Odd);
    }
  }
};

template <typename FImpl> class A2AWorkerOnelink : public A2AWorkerBase<FImpl> {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;

public:
  A2AWorkerOnelink() = delete;
  A2AWorkerOnelink(GridBase *grid, const std::vector<ComplexField> &mom,
                   const std::vector<StagGamma::SpinTastePair> &gammas,
                   LatticeGaugeField *U, int orthogDir)
      : A2AWorkerBase<FImpl>(grid) {
    this->_odd_shifts = true;
    if (mom.size()) {
      assert(0);
    } else {
      this->_task_e =
          new A2ATaskOnelink<FImpl>(grid, orthogDir, gammas, U, Even);
      this->_task_o = new A2ATaskOnelink<FImpl>(
          grid, orthogDir,
          dynamic_cast<A2ATaskOnelink<FImpl> &>(*this->_task_e), gammas, U,
          Odd);
    }
  }
  A2AWorkerOnelink(GridBase *grid, const std::vector<ComplexField> &mom,
                   const std::vector<ComplexField> &Amu, LatticeGaugeField *U,
                   int orthogDir)
      : A2AWorkerBase<FImpl>(grid) {
    // A2A onelink EM not implemented yet
    assert(0);
    this->_odd_shifts = true;
    // if(mom.size()) {
    // assert(0);
    // } else {
    // this->_task_e = new A2ATaskOnelink<FImpl>(grid,orthogDir,Amu,Even,U);
    // this->_task_o = new
    // A2ATaskOnelink<FImpl>(grid,orthogDir,dynamic_cast<A2ATaskOnelink<FImpl>
    // &>(*this->_task_e),{},Odd,U);
    // }
  }
};

template <typename FImpl>
class A2AWorkerSpinTaste : public A2AWorkerBase<FImpl> {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;

public:
  A2AWorkerSpinTaste() = delete;
  A2AWorkerSpinTaste(GridBase *grid, const std::vector<ComplexField> &mom,
                     const std::vector<StagGamma::SpinTastePair> &gammas,
                     LatticeGaugeField *U, int orthogDir)
      : A2AWorkerBase<FImpl>(grid) {
    if (mom.size()) {
      assert(0); // momentum projection not implemented (same as Onelink)
    }

    // Uniform-popcount validation + _odd_shifts. _odd_shifts is the LIVE
    // Even/Odd right-vector routing flag (used in StagMesonField), so the
    // uniformity check must live HERE (where the live consumer is), not rely
    // on the task ctor firing first. A2ATaskSpinTaste ctor also checks, but
    // duplicating here makes the worker self-validating.
    if (!gammas.empty()) {
      StagGamma spinTaste;
      spinTaste.setSpinTaste(gammas[0]);
      int pc = StagGamma::popcountShift(spinTaste._spin, spinTaste._taste);
      this->_odd_shifts = (pc % 2 == 1);
      for (int i = 1; i < (int)gammas.size(); i++) {
        spinTaste.setSpinTaste(gammas[i]);
        int pci = StagGamma::popcountShift(spinTaste._spin, spinTaste._taste);
        if (pci != pc) {
          std::cerr
              << "A2AWorkerSpinTaste requires uniform popcount; gamma 0 has "
                 "popcount "
              << pc << " but gamma " << i << " has popcount " << pci
              << std::endl;
          GridAbort();
        }
      }
    }

    this->_task_e =
        new A2ATaskSpinTaste<FImpl>(grid, orthogDir, gammas, U, Even);
    this->_task_o =
        new A2ATaskSpinTaste<FImpl>(grid, orthogDir, gammas, U, Odd);
  }
};

///////////////////////////////////////////////////////////////////////////////
// A2AWorkerSpinTasteStencil: full-grid stencil worker (full-grid only in v3.3).
// The caller passes full-grid LHS/RHS arrays (no E/O splitting); the task
// promotes them directly to the 5D grid. Mirrors the A2AWorkerLocal/Onelink
// lifecycle: setLeft/setRight re-run only when the input address changes
// (address-cache via the inherited _l_addr/_r_addr), so a worker kept alive
// across many calls skips redundant 5D re-promotion.
///////////////////////////////////////////////////////////////////////////////
template <class FImpl>
class A2AWorkerSpinTasteStencil : public A2AWorkerBase<FImpl> {
public:
  // NOTE (B3): A2A_TYPEDEFS is #undef'd at the end of A2ATask.h before this
  // header is parsed — use the four explicit local typedefs, matching
  // A2AWorkerLocal/Onelink/SpinTaste (A2AWorker.h:88-91).
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;

  A2ATaskSpinTasteStencil<FImpl> *_stencil_task;

  A2AWorkerSpinTasteStencil(GridBase *grid,
                            const std::vector<ComplexField> &mom,
                            const std::vector<StagGamma::SpinTastePair> &gammas,
                            LatticeGaugeField *U, int orthogDir)
      : A2AWorkerBase<FImpl>(grid) {
    // Momentum projection is not implemented for the stencil path (same as the
    // sibling A2AWorkerSpinTaste). MesonField passes `ph` unconditionally, so
    // this guards useStencil + non-zero momentum from silently producing
    // unprojected (wrong) results. (C3 review fix; design follow-up:
    // .rpiv/artifacts/designs/2026-07-25_16-04-12_opt-a2a-stencil-spin-taste-v3.md)
    if (mom.size()) {
      assert(0 && "A2AWorkerSpinTasteStencil: momentum projection not implemented");
    }
    GridCartesian *fullGrid = dynamic_cast<GridCartesian *>(grid);
    assert(fullGrid != nullptr);
    _stencil_task = new A2ATaskSpinTasteStencil<FImpl>(
        fullGrid, orthogDir, gammas, U, Even);
    this->_task_e = _stencil_task;
    this->_task_o = nullptr;
  }

  // NOTE (B1): NO custom destructor. ~A2AWorkerBase does `delete _task_e`
  // (== _stencil_task); a custom body that also deletes _stencil_task would
  // double-free. A2AWorkerSpinTaste has no custom dtor for the same reason.
  // (getFullGrid() on the task exposes the protected _fullGrid if ever needed.)

  template <typename TensorType>
  void StagMesonFieldStencil(TensorType &mat,
                             const FermionField *lhs,
                             const FermionField *rhs,
                             int sizeL, int sizeR) {
    // Device cache for the output mat.
    if (this->_cache_bytes < mat.size() * sizeof(scalar_type)) {
      if (this->_cache_bytes != 0)
        acceleratorFreeDevice(this->_cache_device);
      this->_cache_bytes = mat.size() * sizeof(scalar_type);
      this->_cache_device =
          (scalar_type *)acceleratorAllocDevice(this->_cache_bytes);
    }
    scalar_type *matDevice = this->_cache_device;
    accelerator_for(idx, mat.size(), 1, { matDevice[idx] = 0.0; });

    // Address-cache (C1): re-run the 5D promotion only when the input pointer
    // changes — mirrors the inherited StagMesonField's _l_addr/_r_addr gating.
    // Feed the full-grid inputs directly to the task (NO sizeL/sizeR 4D
    // full-grid temporaries); the task owns the 4D->5D promotion.
    if (this->_l_addr != lhs) {
      this->_l_addr = lhs;
      _stencil_task->setLeft(lhs, sizeL);
    }
    if (this->_r_addr != rhs) {
      this->_r_addr = rhs;
      _stencil_task->setRight(rhs, sizeR);
    }
    _stencil_task->execute(matDevice);

    acceleratorCopyFromDevice(matDevice, mat.data(),
                              mat.size() * sizeof(scalar_type));
    // Multi-rank reduction (B6): each rank summed only its local spatial sites;
    // the inherited StagMesonField does this GlobalSumVector — the stencil path
    // must too, or multi-rank mat values are wrong (breaks the BLOCKING checkpoint).
    this->_grid->GlobalSumVector(mat.data(), mat.size());
  }
};

template <class FImpl>
template <typename TensorType>
void A2AWorkerBase<FImpl>::StagMesonField(TensorType &mat,
                                          const FermionField *lhs_wi_E,
                                          const FermionField *lhs_wi_O,
                                          const FermionField *rhs_vj_E,
                                          const FermionField *rhs_vj_O) {
  if (_cache_bytes < mat.size() * sizeof(scalar_type)) {
    if (_cache_bytes != 0) {
      acceleratorFreeDevice(_cache_device);
    }
    _cache_bytes = mat.size() * sizeof(scalar_type);
    std::cout << GridLogPerformance << "cache bytes: " << _cache_bytes
              << std::endl;
    _cache_device = (scalar_type *)acceleratorAllocDevice(_cache_bytes);
  }
  scalar_type *matDevice = _cache_device;
  accelerator_for(idx, mat.size(), 1, { matDevice[idx] = 0.0; });

  int sizeL = mat.dimension(3);
  int sizeR = mat.dimension(4);
  bool checkerL = lhs_wi_E[0].Grid()->_isCheckerBoarded;
  bool checkerR = rhs_vj_E[0].Grid()->_isCheckerBoarded;

  if (checkerL)
    sizeL /= 2;
  if (checkerR)
    sizeR /= 2;

  // scalar_type *matDevice = mat.data();

  if (_l_addr != lhs_wi_E) {
    _l_addr = lhs_wi_E;

    _task_e->setLeft(lhs_wi_E, sizeL);
    if (checkerL)
      _task_o->setLeft(lhs_wi_O, sizeL);
    else if (checkerR)
      _task_o->setLeft(*_task_e);
  }

  if (_r_addr != rhs_vj_E) {
    _r_addr = rhs_vj_E;

    if (checkerR) {
      if (_odd_shifts) {
        _task_e->setRight(rhs_vj_O, sizeR);
        _task_o->setRight(rhs_vj_E, sizeR);
      } else {
        _task_e->setRight(rhs_vj_E, sizeR);
        _task_o->setRight(rhs_vj_O, sizeR);
      }
    } else {
      _task_e->setRight(rhs_vj_E, sizeR);
      if (checkerL) {
        _task_o->setRight(*_task_e);
      }
    }
  }

  _t_kernel = -usecond();
  if (!(checkerL || checkerR)) {
    _task_e->execute(matDevice);
  } else {
    _task_e->execute(matDevice);
    _task_o->execute(matDevice);
  }

  setFlops(_task_e->getFlops());
  _t_kernel += usecond();

  int resultStride = mat.dimension(2) * mat.dimension(3) * mat.dimension(4);
  int nGamma = mat.dimension(0) * mat.dimension(1);

  scalar_type *matHost = mat.data();
  size_t matBytes = mat.size() * sizeof(scalar_type);
  acceleratorCopyFromDevice(matDevice, matHost, matBytes);

  _t_gsum = -usecond();
  this->_grid->GlobalSumVector(matHost, nGamma * resultStride);
  _t_gsum += usecond();
}

NAMESPACE_END(Grid);
