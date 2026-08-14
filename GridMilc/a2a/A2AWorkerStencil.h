/******************************************************************************/
/* A2AWorkerStencil.h -- split stencil worker for the a2a meson-field path.    */
/*                                                                            */
/* A2AWorkerSpinTasteStencil is the full-grid, single-task production worker,  */
/* detached from the legacy worker base class hierarchy: it owns its grid, an  */
/* output-mat device cache (via the A2ACache DOD facility), an L/R address     */
/* cache, and the stencil task by concrete pointer. The public entry is        */
/* StagMesonField(mat, lhs, rhs, sizeL, sizeR, ct): full-grid LHS/RHS arrays   */
/* plus an explicit ContractType flag (Full only this cycle; half/mixed is     */
/* pending the checkerboard split).                                           */
/*                                                                            */
/* Part of GridMilc (https://github.com/paboyle/Grid).                       */
/******************************************************************************/
#pragma once

#include <Grid/GridQCDcore.h>
#include <Grid/Grid_Eigen_Tensor.h>
#include <GridMilc/a2a/A2ACache.h>
#include <GridMilc/a2a/A2AContractType.h>
#include <GridMilc/a2a/A2ATaskStencil.h>
#include <GridMilc/spin/StagGamma.h>

NAMESPACE_BEGIN(Grid);

///////////////////////////////////////////////////////////////////////////////
// A2AWorkerSpinTasteStencil: full-grid stencil worker (full-grid only in v3.3).
// The caller passes full-grid LHS/RHS arrays (no E/O splitting); the task
// promotes them directly to the 5D grid. Mirrors the A2AWorkerLocal/Onelink
// lifecycle: setLeft/setRight re-run only when the input address changes
// (address-cache via _l_addr/_r_addr), so a worker kept alive across many calls
// skips redundant 5D re-promotion.
///////////////////////////////////////////////////////////////////////////////
template <class FImpl>
class A2AWorkerSpinTasteStencil {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;

  GridCartesian *_grid;

  // Timing/flops surface (matches the legacy worker base): _t_kernel brackets
  // the task execute, _t_gsum the final multi-rank reduction. getFlops()
  // reports the task's per-(site,l,r,t) flop count.
  double _flops = 0.0, _t_kernel = 0.0, _t_gsum = 0.0;

  // Output-mat device cache (A2ACache facility): grow-only, reused across the
  // many StagMesonField calls a long-lived worker serves, freed at dtor.
  A2AMatCache<scalar_type> _matCache;

  // L/R address cache.
  const FermionField *_l_addr = nullptr;
  const FermionField *_r_addr = nullptr;

  // The stencil task (RAII-owned). Public: benchmarks read task->getFlops()
  // directly (Benchmark_a2a_spin_taste.cc reaches in via
  // worker._stencil_task->getFlops(), which works via unique_ptr::operator->).
  std::unique_ptr<A2ATaskSpinTasteStencil<FImpl>> _stencil_task;

  A2AWorkerSpinTasteStencil() = delete;
  A2AWorkerSpinTasteStencil(GridCartesian *grid,
                            const std::vector<ComplexField> &mom,
                            const std::vector<StagGamma::SpinTastePair> &gammas,
                            LatticeGaugeField *U, int orthogDir)
      : _grid(grid) {
    // Momentum projection is not implemented for the stencil path (same as the
    // sibling A2AWorkerSpinTaste). MesonField passes `ph` unconditionally, so
    // this guards useStencil + non-zero momentum from silently producing
    // unprojected (wrong) results.
    if (mom.size()) {
      assert(0 && "A2AWorkerSpinTasteStencil: momentum projection not implemented");
    }
    _stencil_task = std::make_unique<A2ATaskSpinTasteStencil<FImpl>>(
        grid, orthogDir, gammas, U);
  }

  // Owns both the task and the cache (freeMatCache is a safe no-op on an
  // unused cache). The task itself is auto-freed (unique_ptr); no
  // teardown-order hazard: the task's device state is independent of the
  // output-mat cache.
  ~A2AWorkerSpinTasteStencil() {
    freeMatCache(_matCache);
  }

  void resetCache() { _l_addr = nullptr; _r_addr = nullptr; }

  void setFlops(double flops) { _flops = flops; }
  double getFlops() const { return _flops; }

  // Canonical full-array meson-field entry. lhs/rhs are full-grid vector
  // arrays; ct states the checkerboard/contraction mode explicitly (never
  // probed from the lattice). Only ContractType::Full is implemented;
  // half/mixed lands with the checkerboard split.
  template <typename TensorType>
  void StagMesonField(TensorType &mat, const FermionField *lhs,
                      const FermionField *rhs, int sizeL, int sizeR,
                      ContractType ct = ContractType::Full) {
    // Output-mat device cache + zero (A2ACache facility).
    scalar_type *matDevice =
        ensureMatCache(_matCache, mat.size() * sizeof(scalar_type));
    {
      GRID_TRACE("A2AStencil/ZeroInit");
      zeroMatCache(matDevice, mat.size());
    }

    // Address cache: re-run the 5D promotion only when the input pointer
    // changes; a worker kept alive across many calls skips redundant packing.
    if (_l_addr != lhs) {
      _l_addr = lhs;
      GRID_TRACE("A2AStencil/SetLeft");
      _stencil_task->setLeft(lhs, sizeL);
    }
    if (_r_addr != rhs) {
      _r_addr = rhs;
      GRID_TRACE("A2AStencil/SetRight");
      _stencil_task->setRight(rhs, sizeR);
    }
    _flops = _stencil_task->getFlops();
    _t_kernel = -usecond();
    {
      GRID_TRACE("A2AStencil/Execute");
      _stencil_task->execute(matDevice, ct);
    }
    _t_kernel += usecond();
    {
      GRID_TRACE("A2AStencil/CopyFromDevice");
      copyFromDeviceMat(matDevice, mat.data(), mat.size());
    }
    // Each rank summed only its local sites; the mat values must be
    // GlobalSum'd or multi-rank output is wrong.
    _t_gsum = -usecond();
    {
      GRID_TRACE("A2AStencil/GlobalSum");
      globalSumMat(_grid, mat.data(), mat.size());
    }
    _t_gsum += usecond();
  }
};

NAMESPACE_END(Grid);
