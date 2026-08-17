/******************************************************************************/
/* StencilGather5d.h -- 5D padded gather + 4D->5D field promotion.             */
/* Uses simd_layout={1,1,1,1,Nsimd} so spatial dims have simd=1 -- no lane    */
/* scattering. The gather is a simple per-site offset into the padded grid.   */
/* Part of GridMilc (https://github.com/paboyle/Grid).                       */
/******************************************************************************/
#pragma once
// GridCore (not GridQCDcore): this header is generic over vobj -- it names no
// QCD types, so all QCD instantiation (LatticeFermion, vColourMatrix, ...) is
// the caller's job (A2ATask.h, the test). Matches A2AView.h's include style.
// PaddedCell is pulled transitively via GridCore.h -> Lattice.h.
#include <Grid/GridCore.h>
#include <GridMilc/a2a/A2AView.h>

NAMESPACE_BEGIN(Grid);

///////////////////////////////////////////////////////////////////////////////
// createGrid5d: creates a 5D GridCartesian from a 4D one.
// Dim 4 = Nsimd (RHS vector batching), simd_layout = {1,1,1,1,Nsimd}.
// Spatial procs inherited from 4D; procs[4] = 1 (no MPI split in vector dim).
// Returns an OWNING unique_ptr (the caller owns the grid; derive non-owning
// raw views with .get() where by-value GridCartesian* is required).
///////////////////////////////////////////////////////////////////////////////
inline std::unique_ptr<GridCartesian> createGrid5d(GridCartesian *grid4d) {
  Coordinate gdim4d = grid4d->_fdimensions;
  Coordinate procs4d = grid4d->_processors;
  int Nsimd = grid4d->Nsimd();

  Coordinate gdim5d(std::vector<int>(
      {gdim4d[0], gdim4d[1], gdim4d[2], gdim4d[3], Nsimd}));
  Coordinate simd5d(std::vector<int>({1, 1, 1, 1, Nsimd}));
  Coordinate procs5d(std::vector<int>(
      {procs4d[0], procs4d[1], procs4d[2], procs4d[3], 1}));

  return std::make_unique<GridCartesian>(gdim5d, simd5d, procs5d);
}

///////////////////////////////////////////////////////////////////////////////
// buildPaddedOffset5d: per-(endpoint, original oSite) padded-grid oSite.
// Because every spatial dim has simd=1, the offset is a genuine per-site
// scalar (no lane term). For procs[d]>1 dims, the interior sits at
// ocoor+depth; for single-rank dims the periodic wrap is folded in directly.
// Returns host std::vector<int> indexed as offset[ep * osites + ss]; callers
// stage a device copy only if needed (the task constructor uploads once).
///////////////////////////////////////////////////////////////////////////////
inline std::vector<int>
buildPaddedOffset5d(GridCartesian *grid5d, GridCartesian *paddedGrid5d,
                    int depth, const std::vector<Coordinate> &endpoints) {
  static constexpr int Nd = 4;
  int osites = grid5d->oSites();
  int nEp = (int)endpoints.size();

  Coordinate ldim = grid5d->_ldimensions;
  Coordinate procs = grid5d->_processors;

  std::vector<int> hostOffset(nEp * osites);

  for (int ep = 0; ep < nEp; ep++) {
    const Coordinate &shift = endpoints[ep];
    for (int ss = 0; ss < osites; ss++) {
      Coordinate ocoor(grid5d->Nd());
      grid5d->oCoorFromOindex(ocoor, ss);

      Coordinate paddedOcoor(grid5d->Nd());
      for (int d = 0; d < Nd; d++) {
        int shifted = ocoor[d] + shift[d];
        if (procs[d] > 1) {
          paddedOcoor[d] = shifted + depth; // interior offset by depth
        } else {
          int L = ldim[d];
          paddedOcoor[d] = ((shifted % L) + L) % L; // periodic wrap
        }
      }
      paddedOcoor[Nd] = 0; // dim 4 (vector) is never shifted
      hostOffset[ep * osites + ss] = paddedGrid5d->oIndexReduced(paddedOcoor);
    }
  }

  return hostOffset;
}

///////////////////////////////////////////////////////////////////////////////
// src4dIndex: maps a 5D spatial oSite (physical local coord phys[0..3]) to the
// (oSite, lane) of the originating 4D SIMD field, using the 4D grid's
// interleaved decomposition (phys[d] = ocoor[d] + rdim[d]*icoor[d]).
// Shared by pack5d and promoteField5d.
///////////////////////////////////////////////////////////////////////////////
accelerator_inline void src4dIndex(int &srcOSite, int &srcLane,
                       const Coordinate &phys, const Coordinate &rdim,
                       const Coordinate &ostride, const Coordinate &istride) {
  static constexpr int Nd = 4;
  srcOSite = 0;
  srcLane = 0;
  for (int d = 0; d < Nd; d++) {
    srcOSite += ostride[d] * (phys[d] % rdim[d]);
    srcLane += istride[d] * (phys[d] / rdim[d]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// src4dSiteFrom5d: decode a 5D spatial oSite `ss` into the (oSite, lane) of
// its originating 4D SIMD field. Wraps the CoorFromIndex + src4dIndex preamble
// shared by pack5d and promoteField5d. Lane-independent: for a fixed ss
// the (srcOSite, srcLane) result is the same for every dim-5 lane, so callers
// may compute it once and reuse across lanes.
///////////////////////////////////////////////////////////////////////////////
accelerator_inline void src4dSiteFrom5d(int &srcOSite, int &srcLane, int ss,
                                        int nd5d, const Coordinate &rdim5d,
                                        const Coordinate &rdim,
                                        const Coordinate &ostride,
                                        const Coordinate &istride) {
  Coordinate phys(nd5d);
  Lexicographic::CoorFromIndex(phys, ss, rdim5d);
  src4dIndex(srcOSite, srcLane, phys, rdim, ostride, istride);
}

///////////////////////////////////////////////////////////////////////////////
// pack5d: pack nVec distinct 4D Lattice<vobj> vectors into ONE 5D Lattice<vobj>,
// each vector occupying a distinct dim-5 lane (lane == vector index). Generic
// over L or R: used for RHS (packed then padded via PaddedCell::Exchange in
// setRight) and LHS (read at-site, unshifted — no Exchange needed) alike.
// Lanes >= nVec are left untouched (caller Zero-fills the destination first).
// Uses the portable per-lane pattern (#ifdef GRID_SIMT) since each lane reads
// a DIFFERENT source vector (setup-time, not the fused kernel).
///////////////////////////////////////////////////////////////////////////////
template <typename vobj>
inline void pack5d(Lattice<vobj> &rhs5d, const Lattice<vobj> *rhs4d, int nVec,
                      GridBase *grid4d, GridCartesian *grid5d) {
  Coordinate rdim = grid4d->_rdimensions;
  Coordinate ostride = grid4d->_ostride;
  Coordinate istride = grid4d->_istride;
  int Nsimd = grid4d->Nsimd();
  int dstOsites = grid5d->oSites();
  // 5D geometry for in-kernel coord decode: oCoorFromOindex()/Nd() are host-
  // only (not accelerator_inline), so call accelerator_inline
  // Lexicographic::CoorFromIndex directly with the captured 5D _rdimensions
  // (same pattern as PaddedCell::GatherSlice).
  Coordinate rdim5d = grid5d->_rdimensions;
  int nd5d = grid5d->Nd();

  A2AFieldView<vobj> srcView;
  srcView.openViews(rhs4d, nVec);
  auto srcView_p = srcView.getView();

  autoView(dst_v, rhs5d, AcceleratorWrite);

  accelerator_for(ss, dstOsites, Nsimd, {
    int srcOSite, srcLane;
    src4dSiteFrom5d(srcOSite, srcLane, ss, nd5d, rdim5d, rdim, ostride, istride);
#ifdef GRID_SIMT
    {
      int lane = acceleratorSIMTlane(Nsimd);
#else
    for (int lane = 0; lane < Nsimd; lane++) {
#endif
      if (lane < nVec) {
        auto scalar = extractLane(srcLane, srcView_p[lane][srcOSite]);
        insertLane(lane, dst_v[ss], scalar);
      }
#ifdef GRID_SIMT
    }
#else
    }
#endif
  });

  srcView.closeViews();
}

///////////////////////////////////////////////////////////////////////////////
// promoteField5d: promote a 4D Lattice<vobj> (SIMD over spatial) to a 5D
// Lattice<vobj> (simd_layout={1,1,1,1,Nsimd}) with the scalar value REPLICATED
// across all Nsimd dim-5 lanes. Used for the gauge field U, LHS L, and W_s:
// these do not vary by RHS, so every lane holds the same value (broadcast).
///////////////////////////////////////////////////////////////////////////////
template <typename vobj>
inline void promoteField5d(Lattice<vobj> &dst5d, const Lattice<vobj> &src4d,
                           GridBase *grid4d, GridCartesian *grid5d) {
  Coordinate rdim = grid4d->_rdimensions;
  Coordinate ostride = grid4d->_ostride;
  Coordinate istride = grid4d->_istride;
  int Nsimd = grid4d->Nsimd();
  int dstOsites = grid5d->oSites();
  Coordinate rdim5d = grid5d->_rdimensions; // 5D geometry for in-kernel coord decode
  int nd5d = grid5d->Nd();

  autoView(src_v, src4d, AcceleratorRead);
  autoView(dst_v, dst5d, AcceleratorWrite);

  accelerator_for(ss, dstOsites, 1, {
    int srcOSite, srcLane;
    src4dSiteFrom5d(srcOSite, srcLane, ss, nd5d, rdim5d, rdim, ostride, istride);
    // Extract the scalar once, replicate into all Nsimd lanes.
    auto scalar = extractLane(srcLane, src_v[srcOSite]);
    vobj val;
    for (int lane = 0; lane < Nsimd; lane++)
      insertLane(lane, val, scalar);
    dst_v[ss] = val;
  });
}

///////////////////////////////////////////////////////////////////////////////
// createGridW: 5D GridCartesian with simd_layout={1,1,1,1,1} (Nsimd=1) for the
// scalar W Lattice. Dim-4 = nDirections (the forward endpoints); spatial procs
// inherited from grid4d; procs[4]=1. Spatial simd=1 matches the 5D gather grid
// (so paddedSS/interiorOffset indices align). Nsimd=1 is consistent with the
// scalar ColourMatrix element (ColourMatrix::Nsimd()==1) -- no SIMD mismatch.
// Returns an OWNING unique_ptr (same ownership convention as createGrid5d).
///////////////////////////////////////////////////////////////////////////////
inline std::unique_ptr<GridCartesian> createGridW(GridCartesian *grid4d, int nDirections) {
  Coordinate gdim4d = grid4d->_fdimensions;
  Coordinate procs4d = grid4d->_processors;
  Coordinate gdimW(std::vector<int>(
      {gdim4d[0], gdim4d[1], gdim4d[2], gdim4d[3], nDirections}));
  Coordinate simdW(std::vector<int>({1, 1, 1, 1, 1}));
  Coordinate procsW(std::vector<int>(
      {procs4d[0], procs4d[1], procs4d[2], procs4d[3], 1}));
  return std::make_unique<GridCartesian>(gdimW, simdW, procsW);
}

///////////////////////////////////////////////////////////////////////////////
// unpackScalarW: unpack a padded 5D SIMD field (replicated dim-5) into a
// SCALAR store (a Lattice<ColourMatrix> view slice), one scalar object per
// padded oSite. Because the field was
// produced by promoteField5d (every dim-5 lane holds the same value),
// extractLane(0) yields the canonical scalar. Used at task construction to
// build the persistent scalar W store from the transient padded grid5d field.
// Generic over vobj/sobj (GridCore-only: no QCD types named here); the caller
// (A2ATask.h) instantiates vobj=vColourMatrix, sobj=ColourMatrix.
///////////////////////////////////////////////////////////////////////////////
template <typename vobj, typename sobj>
inline void unpackScalarW(sobj *dst, const Lattice<vobj> &padded5d,
                          int paddedOsites) {
  autoView(padded_v, padded5d, AcceleratorRead);
  accelerator_for(ss, paddedOsites, 1, {
    dst[ss] = extractLane(0, padded_v[ss]);
  });
}

///////////////////////////////////////////////////////////////////////////////
// buildInteriorOffset: per-source-oSite padded-grid oSite for a SHIFT-0
// (identity) gather = the interior copy of each original site in the padded
// grid. Forward-endpoint W is read at the SOURCE site x (always local); this
// maps ss -> the padded interior oSite. Mirrors buildPaddedOffset5d's
// arithmetic with a zero endpoint. Host-only (oCoorFromOindex/oIndexReduced).
///////////////////////////////////////////////////////////////////////////////
inline std::vector<int>
buildInteriorOffset(GridCartesian *grid5d, GridCartesian *paddedGrid5d,
                    int depth) {
  static constexpr int Nd = 4;
  int osites = grid5d->oSites();
  Coordinate procs = grid5d->_processors;

  std::vector<int> hostOffset(osites);
  for (int ss = 0; ss < osites; ss++) {
    Coordinate ocoor(grid5d->Nd());
    grid5d->oCoorFromOindex(ocoor, ss);
    Coordinate paddedOcoor(grid5d->Nd());
    for (int d = 0; d < Nd; d++) {
      if (procs[d] > 1) {
        paddedOcoor[d] = ocoor[d] + depth; // interior offset by depth
      } else {
        paddedOcoor[d] = ocoor[d]; // shift 0: identity, no periodic wrap needed
      }
    }
    paddedOcoor[Nd] = 0; // dim 4 (vector) never shifted
    hostOffset[ss] = paddedGrid5d->oIndexReduced(paddedOcoor);
  }
  return hostOffset;
}

NAMESPACE_END(Grid);
