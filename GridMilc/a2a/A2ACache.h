/******************************************************************************/
/* A2ACache.h -- procedural/DOD device-cache facility for the a2a meson-field */
/* workers. De-duplicates the output-mat device-cache reallocation, device    */
/* zero-init, and copy-from-device + multi-rank reduction tail shared by the  */
/* stencil (StagMesonFieldStencil) and legacy (StagMesonField) paths.         */
/*                                                                             */
/* DOD, not OOP: a POD state struct (A2AMatCache) + free functions. No        */
/* inheritance, no dynamic dispatch in the hot path (methodology principle M1).*/
/* The cache is grow-only and reused across the many StagMesonField calls a   */
/* long-lived worker serves, freed at worker dtor (M4).                       */
/*                                                                             */
/* Part of GridMilc (https://github.com/paboyle/Grid).                       */
/******************************************************************************/
#pragma once
// GridCore (not GridQCDcore): the facility names no QCD types. GridCore.h
// transitively provides accelerator_for, acceleratorAllocDevice/FreeDevice,
// acceleratorCopyFromDevice, and GridBase::GlobalSumVector. Matches the
// A2AView.h / StencilGather5d.h include style.
#include <Grid/GridCore.h>

NAMESPACE_BEGIN(Grid);

// POD state for the output-mat device cache. Default-zeroed; held by a worker
// and reused across StagMesonField calls. Shape mirrors precisionChangeWorkspace
// (Grid/lattice/Lattice_transfer.h): raw device pointer, no base class.
template <typename Scalar> struct A2AMatCache {
  Scalar *device = nullptr;
  size_t  bytes  = 0;
};

// Ensure the cache can hold nBytes; grow-only (never shrinks), freeing the
// previous allocation on growth. Returns the device pointer. Equivalent to the
// inline `_cache_bytes < needed` realloc block in A2AWorker.h
// StagMesonField[Stencil] (A2AWorker.h:252-261 and :305-311).
template <typename Scalar>
inline Scalar *ensureMatCache(A2AMatCache<Scalar> &c, size_t nBytes) {
  if (c.bytes < nBytes) {
    if (c.bytes != 0)
      acceleratorFreeDevice(c.device);
    c.bytes  = nBytes;
    c.device = static_cast<Scalar *>(acceleratorAllocDevice(nBytes));
  }
  return c.device;
}

// Release the cache. Safe on a default-constructed/already-freed cache (no-op
// when bytes == 0). Intended for the worker destructor (mirrors the
// `if (_cache_bytes != 0) acceleratorFreeDevice(_cache_device)` in
// ~A2AWorkerBase, A2AWorker.h:49-52).
template <typename Scalar>
inline void freeMatCache(A2AMatCache<Scalar> &c) {
  if (c.bytes != 0) {
    acceleratorFreeDevice(c.device);
    c.device = nullptr;
    c.bytes  = 0;
  }
}

// Zero n scalars on the device. Equivalent to the
// `accelerator_for(idx, n, 1, { matDevice[idx] = 0.0; })` block in the workers.
template <typename Scalar>
inline void zeroMatCache(Scalar *device, size_t n) {
  accelerator_for(idx, n, 1, { device[idx] = Scalar(0); });
}

// Copy device -> host. Equivalent to the acceleratorCopyFromDevice block in the
// workers (A2AWorker.h:285 stencil, :377 legacy). `count` is the element count
// (bytes = count*sizeof(Scalar)). The facility emits no GRID_TRACE of its own;
// Phase 3 wraps the call in a GRID_TRACE region at the call site, matching the
// inline pattern (separate CopyFromDevice / GlobalSum regions).
template <typename Scalar>
inline void copyFromDeviceMat(Scalar *device, Scalar *host, size_t count) {
  acceleratorCopyFromDevice(device, host, count * sizeof(Scalar));
}

// Reduce across MPI ranks. Equivalent to the grid->GlobalSumVector tail in the
// workers (A2AWorker.h:292 stencil, :383 legacy). Trace at the call site.
template <typename Scalar>
inline void globalSumMat(GridBase *grid, Scalar *host, size_t count) {
  grid->GlobalSumVector(host, (int)count);
}

NAMESPACE_END(Grid);
