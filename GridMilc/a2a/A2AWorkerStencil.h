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
/* convention) and the task writes the RAW (2L, R) parity-split partials and   */
/* the worker expands them into the legacy interleaved (M, M-dagger) slot      */
/* layout on the host (sigma applied in finishMatOutput, after the raw-buffer  */
/* global sum): BothHalf (2L, 2R), LeftHalf (2L, R), RightHalf (L, 2R);        */
/* ParityBisect keeps the raw (2L, R) partials as its final output.            */
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
// parity internally and writes the RAW (2L, R) parity-split partials; the
// worker's finishMatOutput global-sums the raw buffer and expands it into
// the legacy interleaved (M, M-dagger) slot layout on the host (sigma = -1
// for odd popcount(spin^taste)); ParityBisect keeps the raw (2L, R)
// partials as its final output. Any ContractType != Full (including ParityBisect, which
// claims no packed side) requests the parity-split output; the Left/Right
// bits only document which side's arrays are packed. U may be null iff every
// gamma has zero displacement. Mirrors the A2AWorkerLocal/Onelink lifecycle:
// setLeft/setRight re-run only when the input address changes (address-cache
// via _l_addr/_r_addr), so a worker kept alive across many calls skips
// redundant 5D re-promotion; resetLeft/resetRight invalidate a single side
// (resetCache() clears both) so callers can re-pack just the moved leg.
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

  // Raw (2L, R) parity partials: the D2H copy + multi-rank global sum target
  // for the three CB modes (BothHalf/LeftHalf/RightHalf) -- half the BothHalf
  // interleaved volume, and the gsum partition of the pre-fusion build
  // (bitwise at any rank count). Grow-only host scratch; the expansion into
  // the caller's interleaved mat runs in finishMatOutput.
  std::vector<scalar_type> _rawScratch;

  // Per-gamma sigma sign (+1/-1) for the host expansion: sigma = -1 for odd
  // popcount(spin^taste) -- the frozen legacy simdSumHalf/simdSumMixed
  // convention (the pre-77f540a MesonField.hpp setWorkerStencil table).
  std::vector<RealD> _sigmaHost;

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
    // Host sigma table for the CB expansion (finishMatOutput).
    _sigmaHost.reserve(gammas.size());
    for (const auto &g : gammas) {
      int pc = StagGamma::popcountShift(g._spin, g._taste);
      _sigmaHost.push_back((pc & 1) ? -1.0 : 1.0);
    }
  }

  // Owns both the task and the cache (freeMatCache is a safe no-op on an
  // unused cache). The task itself is auto-freed (unique_ptr); no
  // teardown-order hazard: the task's device state is independent of the
  // output-mat cache.
  ~A2AWorkerSpinTasteStencil() {
    freeMatCache(_matCache);
  }

  // Per-side reset: invalidate only one leg's address cache so a caller
  // that re-packs just one side (e.g. the j-major block loop in
  // MesonField.hpp) skips the other side's redundant re-promotion.
  void resetLeft()  { _l_addr = nullptr;  _l5d_addr = nullptr; }
  void resetRight() { _r_addr = nullptr;  _r5d_addr = nullptr; }
  // Combined form kept for external callers that want a full invalidate
  // (Benchmark_a2a_spin_taste.cc forces re-set on every timed rep).
  void resetCache() { resetLeft(); resetRight(); }

  void setFlops(double flops) { _flops = flops; }
  double getFlops() const { return _flops; }

private:
  // Output-prep head shared by all StagMesonField overloads: dims gate,
  // output-mat device cache, zero. Runs BEFORE the task input sets,
  // preserving the original 4D entry's call order exactly.
  template <typename TensorType>
  scalar_type *prepMatOutput(TensorType &mat, int sizeL, int sizeR,
                             ContractType ct) {
    // Contract/dims consistency: sizeL/sizeR are array entry counts. The
    // three CB contract modes emit the LEGACY INTERLEAVED (M, M-dagger)
    // slot layout directly (fused at the task's assembleMat store sites):
    //   BothHalf (2L, 2R), LeftHalf (2L, R), RightHalf (L, 2R).
    // ParityBisect keeps the raw (2L, R) parity-split output (the
    // documented raw-partials escape hatch; no callers today).
    int rowMult, colMult;
    switch (ct) {
    case ContractType::Full:         rowMult = 1; colMult = 1; break;
    case ContractType::BothHalf:     rowMult = 2; colMult = 2; break;
    case ContractType::LeftHalf:     rowMult = 2; colMult = 1; break;
    case ContractType::RightHalf:    rowMult = 1; colMult = 2; break;
    case ContractType::ParityBisect: rowMult = 2; colMult = 1; break;
    default:                         rowMult = 0; colMult = 0; break;
    }
    if (ct == ContractType::undef ||
        (int)mat.dimension(3) != rowMult * sizeL ||
        (int)mat.dimension(4) != colMult * sizeR) {
      std::cerr << "A2AWorkerSpinTasteStencil::StagMesonField: mat dims ("
                << mat.dimension(3) << "," << mat.dimension(4)
                << ") inconsistent with sizes (" << rowMult * sizeL << ","
                << colMult * sizeR << ") for ContractType " << (int)ct
                << std::endl;
      GridAbort();
    }
    // Output-mat device cache: sized to the RAW (2L, R) partial count for
    // the three CB modes (the device stores raw partials; the interleaved
    // final values only ever exist on the host), to the full mat size
    // otherwise (Full/ParityBisect: raw IS the contract).
    size_t rawCount = mat.size();
    if (ct == ContractType::BothHalf)
      rawCount /= 2; // (2L,2R) interleaved -> (2L,R) raw partials
    scalar_type *matDevice =
        ensureMatCache(_matCache, rawCount * sizeof(scalar_type));
    // Per-call device zero (the legacy worker's A2A/ZeroInit precedent).
    // NOT dead -- Phase 4's overwrite argument does NOT transfer to the
    // device mat: assembleMat writes only this rank's local orthog-dir
    // slab (gt = rt + pc*localOrthogDimSize, pc =
    // _grid5d->_processor_coor[orthogDir], A2ATaskStencil.h:736-737,:821)
    // while the D2H copy + global sum below read the FULL raw buffer.
    // Without the zero, other ranks' slabs hold stale/garbage device
    // memory that the global sum assembles into the output. Invisible at
    // single-rank 4^4 (all slabs local -- the bitwise gate cannot see
    // it); load-bearing at multi-rank production (Testing Strategy
    // item 10 trap, second instance; found 2026-10-12).
    {
      GRID_TRACE("A2AStencil/ZeroInit");
      zeroMatCache(matDevice, rawCount);
    }
    if (ct == ContractType::BothHalf || ct == ContractType::LeftHalf ||
        ct == ContractType::RightHalf)
      _rawScratch.resize(rawCount);
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
    // D2H + multi-rank global sum: over the RAW (2L, R) partials for the
    // three CB modes (half the BothHalf interleaved volume; the pre-fusion
    // gsum partition -- bitwise at any rank count), over the final buffer
    // for Full/ParityBisect (raw IS their contract).
    bool expand = (ct == ContractType::BothHalf ||
                   ct == ContractType::LeftHalf ||
                   ct == ContractType::RightHalf);
    scalar_type *gsumTarget = expand ? _rawScratch.data() : mat.data();
    size_t count = expand ? _rawScratch.size() : mat.size();
    {
      GRID_TRACE("A2AStencil/CopyFromDevice");
      copyFromDeviceMat(matDevice, gsumTarget, count);
    }
    // Each rank summed only its local sites; the mat values must be
    // GlobalSum'd or multi-rank output is wrong.
    _t_gsum = -usecond();
    {
      GRID_TRACE("A2AStencil/GlobalSum");
      globalSumMat(_grid, gsumTarget, count);
    }
    _t_gsum += usecond();
    if (expand) {
      GRID_TRACE("A2AStencil/ExpandLegacy");
      // Legacy slot expansion -- reconstructLegacy's exact algebra
      // (pre-77f540a MesonField.hpp), PARALLEL over (e, s, t, l):
      //   BothHalf:  (2l,2r)=M0+M1  (2l,2r+1)=sig*(M0-M1)
      //              (2l+1,2r)=M0-M1 (2l+1,2r+1)=sig*(M0+M1)
      //   LeftHalf:  (2l,r)=M0+M1    (2l+1,r)=M0-M1      (sig-free)
      //   RightHalf: (l,2r)=M0+M1    (l,2r+1)=sig*(M0-M1)
      // with M0 = raw(e,s,t,l,r), M1 = raw(e,s,t,sizeL+l,r), and
      // sig = _sigmaHost[s] (the mat's nstr axis IS the gamma axis:
      // _nstr = gammas.size(), the pre-77f540a table's indexing).
      // Reads use the raw (2L,R) flat mapping r + R*l (RowMajor, the Phase 3
      // conversion precedent); writes via mat's operator() so the final
      // interleaved layout is automatic. thread_for is load-bearing: the
      // serial variant is the 0.3-0.6 s failure mode Phase 5 deleted.
      const int next = (int)mat.dimension(0);
      const int nstr = (int)mat.dimension(1);
      const int nt = (int)mat.dimension(2);
      // sizeL/sizeR are not passed to finishMatOutput: recover them from
      // the FINAL interleaved dims via the per-ct multipliers (the exact
      // inverse of prepMatOutput's dims gate).
      const int rowMult = (ct == ContractType::BothHalf ||
                           ct == ContractType::LeftHalf)
                              ? 2
                              : 1;
      const int colMult = (ct == ContractType::BothHalf ||
                           ct == ContractType::RightHalf)
                              ? 2
                              : 1;
      const int L = (int)mat.dimension(3) / rowMult;
      const int R = (int)mat.dimension(4) / colMult;
      const scalar_type *raw_p = _rawScratch.data();
      thread_for(idx, (size_t)next * nstr * nt * L, {
        int l = (int)(idx % L);
        size_t rest = idx / L;
        int t = (int)(rest % nt);
        rest /= nt;
        int s = (int)(rest % nstr);
        int e = (int)(rest / nstr);
        RealD sig = _sigmaHost[s];
        const scalar_type *rawBase =
            raw_p + (size_t)(e * nstr + s) * nt * (size_t)(2 * L) * R +
            (size_t)t * (size_t)(2 * L) * R;
        for (int r = 0; r < R; ++r) {
          scalar_type M0 = rawBase[(size_t)r + R * l];
          scalar_type M1 = rawBase[(size_t)r + R * (l + L)];
          scalar_type sum = M0 + M1;
          scalar_type dif = M0 - M1;
          if (ct == ContractType::BothHalf) {
            mat(e, s, t, 2 * l, 2 * r) = sum;
            mat(e, s, t, 2 * l, 2 * r + 1) = sig * dif;
            mat(e, s, t, 2 * l + 1, 2 * r) = dif;
            mat(e, s, t, 2 * l + 1, 2 * r + 1) = sig * sum;
          } else if (ct == ContractType::LeftHalf) {
            mat(e, s, t, 2 * l, r) = sum;
            mat(e, s, t, 2 * l + 1, r) = dif;
          } else { // RightHalf
            mat(e, s, t, l, 2 * r) = sum;
            mat(e, s, t, l, 2 * r + 1) = sig * dif;
          }
        }
      });
    }
  }

public:
  // Canonical full-array meson-field entry. lhs/rhs are full-grid vector
  // arrays; ct states the checkerboard/contraction mode explicitly (never
  // probed from the lattice). For the CB modes each CB-side array entry
  // packs two copies in one full object (E values on even sites, O on odd
  // -- the setCheckerboard convention); the task writes the raw (2L, R)
  // parity partials and the worker expands them into the legacy
  // interleaved (M, M-dagger) slot layout on the host (sigma applied in
  // finishMatOutput, after the raw-buffer global sum): BothHalf (2L, 2R),
  // LeftHalf (2L, R), RightHalf (L, 2R); ParityBisect still emits the raw
  // (2L, R) split. The ct
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
