/******************************************************************************/
/* A2AWorkerStencil.h -- split stencil worker for the a2a meson-field path.    */
/*                                                                            */
/* A2AWorkerSpinTasteStencil is the full-grid, single-task production worker,  */
/* detached from the legacy worker base class hierarchy: it owns its grid, an  */
/* output-mat device cache (via the A2ACache DOD facility), an L/R address     */
/* cache, and the stencil task by concrete pointer. The public entry is        */
/* StagMesonField(mat, lhs, rhs, sizeL, sizeR, ct): full-grid LHS/RHS arrays   */
/* plus an explicit ContractType flag. For the half/mixed CB modes             */
/* (LeftHalf/RightHalf/BothHalf) each CB-side array entry packs two copies in  */
/* one full object (E values on even sites, O on odd -- the setCheckerboard   */
/* convention) and the output is one uniform block layout: always (2L, R)      */
/* with rows [0,L) the <e|.> partials and rows [L,2L) the <o|.> partials.      */
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
// A2AWorkerSpinTasteStencil: full-grid stencil worker. The caller passes
// full-grid LHS/RHS arrays (no E/O splitting); the task promotes them directly
// to the 5D grid. For the CB half/mixed contract modes each CB-side array
// entry packs two copies in one full object (E values on even sites, O on odd
// -- the setCheckerboard convention); the task splits the source-site sum by
// parity internally. Any ContractType != Full (including ParityBisect, which
// claims no packed side) requests the parity-split output; the Left/Right
// bits only document which side's arrays are packed. U may be null iff every
// gamma has zero displacement. Mirrors the A2AWorkerLocal/Onelink lifecycle:
// setLeft/setRight re-run only when the input address changes (address-cache
// via _l_addr/_r_addr), so a worker kept alive across many calls skips
// redundant 5D re-promotion.
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

  // 5D-form siblings of the L/R address caches: first element's raw pointer
  // (mirrors the 4D array-base semantics). Each set cross-invalidates its 4D
  // sibling so 4D<->5D alternation on one worker always re-sets the task
  // (a stale sibling would leave the previous form's views active).
  const FermionField *_l5d_addr = nullptr;
  const FermionField *_r5d_addr = nullptr;

  // The stencil task (RAII-owned). Public: benchmarks read task->getFlops()
  // directly (Benchmark_a2a_spin_taste.cc reaches in via
  // worker._stencil_task->getFlops(), which works via unique_ptr::operator->).
  std::unique_ptr<A2ATaskSpinTasteStencil<FImpl>> _stencil_task;

  A2AWorkerSpinTasteStencil() = delete;
  A2AWorkerSpinTasteStencil(GridCartesian *grid,
                            const std::vector<ComplexField> &mom,
                            const std::vector<StagGamma> &gammas,
                            LatticeGaugeField *U, int orthogDir,
                            std::shared_ptr<GridCartesian> grid5d = nullptr)
      : _grid(grid) {
    // Momentum projection is not implemented for the stencil path (same as the
    // sibling A2AWorkerSpinTaste). MesonField passes `ph` unconditionally, so
    // this guards useStencil + non-zero momentum from silently producing
    // unprojected (wrong) results.
    if (mom.size()) {
      assert(0 && "A2AWorkerSpinTasteStencil: momentum projection not implemented");
    }
    _stencil_task = std::make_unique<A2ATaskSpinTasteStencil<FImpl>>(
        grid, orthogDir, gammas, U, grid5d);
  }

  // Owns both the task and the cache (freeMatCache is a safe no-op on an
  // unused cache). The task itself is auto-freed (unique_ptr); no
  // teardown-order hazard: the task's device state is independent of the
  // output-mat cache.
  ~A2AWorkerSpinTasteStencil() {
    freeMatCache(_matCache);
  }

  void resetCache() {
    _l_addr = nullptr;
    _r_addr = nullptr;
    _l5d_addr = nullptr;
    _r5d_addr = nullptr;
  }

  void setFlops(double flops) { _flops = flops; }
  double getFlops() const { return _flops; }

private:
  // Output-prep head shared by all StagMesonField overloads: dims gate,
  // output-mat device cache, zero. Runs BEFORE the task input sets,
  // preserving the original 4D entry's call order exactly.
  template <typename TensorType>
  scalar_type *prepMatOutput(TensorType &mat, int sizeL, int sizeR,
                             ContractType ct) {
    // Contract/dims consistency: sizeL/sizeR are array entry counts; CB
    // modes double dim 3 (the row block split), dim 4 is never doubled.
    int rowMult = (ct == ContractType::Full) ? 1 : 2;
    if (ct == ContractType::undef ||
        (int)mat.dimension(3) != rowMult * sizeL ||
        (int)mat.dimension(4) != sizeR) {
      std::cerr << "A2AWorkerSpinTasteStencil::StagMesonField: mat dims ("
                << mat.dimension(3) << "," << mat.dimension(4)
                << ") inconsistent with sizes (" << rowMult * sizeL << ","
                << sizeR << ") for ContractType " << (int)ct
                << std::endl;
      GridAbort();
    }
    // Output-mat device cache + zero (A2ACache facility).
    scalar_type *matDevice =
        ensureMatCache(_matCache, mat.size() * sizeof(scalar_type));
    {
      GRID_TRACE("A2AStencil/ZeroInit");
      zeroMatCache(matDevice, mat.size());
    }
    return matDevice;
  }

  // Result tail shared by all StagMesonField overloads: flops, execute,
  // copy-back, GlobalSum. Runs AFTER the task input sets (the address-cache
  // gating lives in the public entries).
  template <typename TensorType>
  void finishMatOutput(TensorType &mat, scalar_type *matDevice,
                       ContractType ct) {
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

public:
  // Canonical full-array meson-field entry. lhs/rhs are full-grid vector
  // arrays; ct states the checkerboard/contraction mode explicitly (never
  // probed from the lattice). For the CB modes each CB-side array entry
  // packs two copies in one full object (E values on even sites, O on odd
  // -- the setCheckerboard convention) and the output uses one uniform
  // block layout: always (2L, R), rows [0,L) = <e|.> partials (even source
  // sites), rows [L,2L) = <o|.> partials (odd source sites). The ct
  // Left/Right bits document which side's arrays are packed; they do not
  // change the output shape. ParityBisect requests the same split output
  // when neither side's arrays are packed (the parity source-site filter is
  // well-defined on any full-grid data); Full alone produces the unsplit
  // (L, R) layout.
  template <typename TensorType>
  void StagMesonField(TensorType &mat, const FermionField *lhs,
                      const FermionField *rhs, int sizeL, int sizeR,
                      ContractType ct = ContractType::Full) {
    scalar_type *matDevice = prepMatOutput(mat, sizeL, sizeR, ct);
    // Address cache: re-run the 5D promotion only when the input pointer
    // changes; a worker kept alive across many calls skips redundant packing.
    if (_l_addr != lhs) {
      _l_addr = lhs;
      _l5d_addr = nullptr; // cross-invalidate: force re-set on 4D<->5D switch
      GRID_TRACE("A2AStencil/SetLeft");
      _stencil_task->setLeft(lhs, sizeL);
    }
    if (_r_addr != rhs) {
      _r_addr = rhs;
      _r5d_addr = nullptr; // cross-invalidate
      GRID_TRACE("A2AStencil/SetRight");
      _stencil_task->setRight(rhs, sizeR);
    }
    finishMatOutput(mat, matDevice, ct);
  }

  // 5D zero-copy lhs (borrowed; the task retains shared_ptrs) + 4D rhs.
  // lhs must satisfy the task's 5D contract: every element on the
  // worker-adopted grid5d (construct this worker with the producer's
  // shared_ptr grid5d), .size() == ceil(sizeL/Nsimd), partial-batch lanes
  // zeroed by the caller, contents immutable while the first element's
  // address stays cached (resetCache() to force re-set).
  template <typename TensorType>
  void StagMesonField(TensorType &mat,
                      const std::vector<std::shared_ptr<FermionField>> &lhs,
                      const FermionField *rhs, int sizeL, int sizeR,
                      ContractType ct = ContractType::Full) {
    scalar_type *matDevice = prepMatOutput(mat, sizeL, sizeR, ct);
    if (lhs.empty() && sizeL > 0) {
      std::cerr << "A2AWorkerSpinTasteStencil::StagMesonField: empty 5D lhs "
                   "with sizeL=" << sizeL << std::endl;
      GridAbort();
    }
    const FermionField *l5 = lhs.empty() ? nullptr : lhs[0].get();
    if (_l5d_addr != l5) {
      _l5d_addr = l5;
      _l_addr = nullptr; // cross-invalidate
      GRID_TRACE("A2AStencil/SetLeft5d");
      _stencil_task->setLeft(lhs, sizeL);
    }
    if (_r_addr != rhs) {
      _r_addr = rhs;
      _r5d_addr = nullptr; // cross-invalidate
      GRID_TRACE("A2AStencil/SetRight");
      _stencil_task->setRight(rhs, sizeR);
    }
    finishMatOutput(mat, matDevice, ct);
  }

  // 4D lhs + 5D pack-skip rhs (Exchange copies into task-owned padded
  // buffers; rhs is not borrowed past the setRight call -- the caller may
  // mutate or free it immediately after the call returns).
  template <typename TensorType>
  void StagMesonField(TensorType &mat, const FermionField *lhs,
                      const std::vector<std::shared_ptr<FermionField>> &rhs,
                      int sizeL, int sizeR,
                      ContractType ct = ContractType::Full) {
    scalar_type *matDevice = prepMatOutput(mat, sizeL, sizeR, ct);
    if (rhs.empty() && sizeR > 0) {
      std::cerr << "A2AWorkerSpinTasteStencil::StagMesonField: empty 5D rhs "
                   "with sizeR=" << sizeR << std::endl;
      GridAbort();
    }
    if (_l_addr != lhs) {
      _l_addr = lhs;
      _l5d_addr = nullptr; // cross-invalidate
      GRID_TRACE("A2AStencil/SetLeft");
      _stencil_task->setLeft(lhs, sizeL);
    }
    const FermionField *r5 = rhs.empty() ? nullptr : rhs[0].get();
    if (_r5d_addr != r5) {
      _r5d_addr = r5;
      _r_addr = nullptr; // cross-invalidate
      GRID_TRACE("A2AStencil/SetRight5d");
      _stencil_task->setRight(rhs, sizeR);
    }
    finishMatOutput(mat, matDevice, ct);
  }

  // 5D lhs + 5D rhs (both per-side contracts above apply).
  template <typename TensorType>
  void StagMesonField(TensorType &mat,
                      const std::vector<std::shared_ptr<FermionField>> &lhs,
                      const std::vector<std::shared_ptr<FermionField>> &rhs,
                      int sizeL, int sizeR,
                      ContractType ct = ContractType::Full) {
    scalar_type *matDevice = prepMatOutput(mat, sizeL, sizeR, ct);
    if (lhs.empty() && sizeL > 0) {
      std::cerr << "A2AWorkerSpinTasteStencil::StagMesonField: empty 5D lhs "
                   "with sizeL=" << sizeL << std::endl;
      GridAbort();
    }
    if (rhs.empty() && sizeR > 0) {
      std::cerr << "A2AWorkerSpinTasteStencil::StagMesonField: empty 5D rhs "
                   "with sizeR=" << sizeR << std::endl;
      GridAbort();
    }
    const FermionField *l5 = lhs.empty() ? nullptr : lhs[0].get();
    if (_l5d_addr != l5) {
      _l5d_addr = l5;
      _l_addr = nullptr; // cross-invalidate
      GRID_TRACE("A2AStencil/SetLeft5d");
      _stencil_task->setLeft(lhs, sizeL);
    }
    const FermionField *r5 = rhs.empty() ? nullptr : rhs[0].get();
    if (_r5d_addr != r5) {
      _r5d_addr = r5;
      _r_addr = nullptr; // cross-invalidate
      GRID_TRACE("A2AStencil/SetRight5d");
      _stencil_task->setRight(rhs, sizeR);
    }
    finishMatOutput(mat, matDevice, ct);
  }
};

NAMESPACE_END(Grid);
