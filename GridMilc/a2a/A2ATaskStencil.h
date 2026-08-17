/******************************************************************************/
/* A2ATaskStencil.h -- 5D stencil spin-taste meson-field task + its file-scope */
/* helpers, split off the legacy task machinery.                             */
/* Self-contained (T1): depends on no legacy a2a worker/task header. The task  */
/* owns its state directly (no base-class inheritance).                        */
/*                                                                             */
/* Part of GridMilc (https://github.com/paboyle/Grid).                        */
/******************************************************************************/
#pragma once

#include <Grid/GridQCDcore.h>
#include <GridMilc/a2a/A2AContractType.h>
#include <GridMilc/a2a/A2AView.h>
#include <GridMilc/a2a/StencilGather5d.h>
#include <GridMilc/spin/StagGamma.h>

#ifndef MF_SUM_ARRAY_MAX
#define MF_SUM_ARRAY_MAX 16
#endif

NAMESPACE_BEGIN(Grid);

///////////////////////////////////////////////////////////////////////////////
// spinTasteEndpoints: enumerate the 2^nActive sign-combination endpoints for a
// spin-taste (the lattice displacements s_ep whose covariant transports W_s the
// gauge chain builds). Pure host-side combinatorics; cheap. Each endpoint is
// one of the 2^popcount(spin^taste) +/-1 sign combos over the active directions
// (gmu[] order, X/Y/Z/T).
///////////////////////////////////////////////////////////////////////////////
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
// classifySpinTasteEndpoints: partition the 2^nActive endpoints into
// forward/backward pairs {s, -s} for endpoint pairing. Canonical forward =
// first-active-direction sign +1 (so exactly one of {s,-s} is forward).
// pc=0 (nActive=0, single endpoint) is forward-only with no pair.
//
// Outputs:
//   forwardEndpoints - the nPairs stored forward endpoint coordinates
//   forwardRep[ep]   - compact forward-field index (0..nPairs-1) for endpoint ep
//                      (== its own index if forward, == its pair's if backward)
//   isForward[ep]    - true iff ep is itself a forward endpoint
// The backward read adj(C_s(x-s)) uses forwardRep[ep] to locate the stored
// forward field. Host-only (one-time setup); O(nEp^2) with nEp <= 16.
///////////////////////////////////////////////////////////////////////////////
inline void classifySpinTasteEndpoints(const std::vector<Coordinate> &endpoints,
                                       const std::array<int, 4> &dirs, int nActive,
                                       std::vector<Coordinate> &forwardEndpoints,
                                       std::vector<int> &forwardRep,
                                       std::vector<bool> &isForward) {
  int nEp = (int)endpoints.size();
  forwardEndpoints.clear();
  forwardRep.assign(nEp, -1);
  isForward.assign(nEp, false);

  if (nActive == 0) {
    // pc=0: single endpoint, forward-only, no backward partner.
    forwardEndpoints.push_back(endpoints[0]);
    forwardRep[0] = 0;
    isForward[0] = true;
    return;
  }

  int firstDir = dirs[0];
  // Forward endpoints: first-active-direction sign +1. Assign compact indices.
  for (int ep = 0; ep < nEp; ep++) {
    if (endpoints[ep][firstDir] > 0) {
      isForward[ep] = true;
      forwardRep[ep] = (int)forwardEndpoints.size();
      forwardEndpoints.push_back(endpoints[ep]);
    }
  }
  // Backward endpoints: map to the forward endpoint whose coordinate is the
  // negation of this one (the {s,-s} partner). The full 2^nActive sign cube is
  // closed under negation, so every backward ep finds a forward partner.
  for (int ep = 0; ep < nEp; ep++) {
    if (!isForward[ep]) {
      for (int ep2 = 0; ep2 < nEp; ep2++) {
        if (!isForward[ep2])
          continue;
        bool match = true;
        for (int d = 0; d < 4; d++)
          if (endpoints[ep2][d] != -endpoints[ep][d]) {
            match = false;
            break;
          }
        if (match) {
          forwardRep[ep] = forwardRep[ep2];
          break;
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// spinTasteGaugeChainUnphasedForward: build the UNPHASED symmetric gauge
// transporter C_s for the FORWARD endpoints only, on grid4d (SIMD Cshift).
// Same covariant chain as spinTasteGaugeChain (the symmetric n!-permutation
// sum, CovShiftForward/Backward) but: (1) iterates only forwardEndpoints (the
// nPairs stored reps), and (2) applies NO applyCoeffsAndPhase -- the
// Follana phase + (1/2)^n/n! scaling is applied in the kernel via the per-gamma
// phase field, so the pairing hermiticity C_{-s}(x)=adj(C_s(x-s)) is
// exact (corr always +1). Returns grid4d chains (no promoteField5d here -- the
// task ctor promotes/pads/unpacks to the scalar store). One-time setup cost.
///////////////////////////////////////////////////////////////////////////////
inline std::vector<LatticeColourMatrix>
spinTasteGaugeChainUnphasedForward(const LatticeGaugeField *U,
                                   const StagGamma &spinTaste,
                                   GridCartesian *grid4d,
                                   const std::vector<Coordinate> &forwardEndpoints) {
  static constexpr int Nd = 4;

  int shift = spinTaste._spin ^ spinTaste._taste;
  std::array<int, 4> dirs;
  int nActive = 0;
  for (int j = 0; j < 4; j++)
    if (static_cast<int>(StagGamma::gmu[j]) & shift)
      dirs[nActive++] = j;

  std::vector<LatticeColourMatrix> Udir4d;
  Udir4d.reserve(Nd);
  for (int mu = 0; mu < Nd; mu++) {
    Udir4d.emplace_back(grid4d);
    Udir4d[mu] = PeekIndex<LorentzIndex>(*U, mu);
  }

  LatticeColourMatrix accumChain(grid4d), chain(grid4d);
  std::array<int, 4> perm;

  std::vector<LatticeColourMatrix> Ws;
  Ws.reserve(forwardEndpoints.size());
  for (const auto &s : forwardEndpoints) {
    accumChain = Zero();
    for (int i = 0; i < nActive; i++)
      perm[i] = i;
    do {
      chain = 1.0;
      for (int i = 0; i < nActive; i++) {
        int dir = dirs[perm[i]];
        int sign = s[dir];
        if (sign > 0) {
          chain = Udir4d[dir] * Cshift(chain, dir, +1);       // CovShiftForward
        } else {
          chain = Cshift(adj(Udir4d[dir]) * chain, dir, -1);  // CovShiftBackward
        }
      }
      accumChain = accumChain + chain;
    } while (std::next_permutation(perm.begin(), perm.begin() + nActive));
    Ws.push_back(accumChain); // grid4d, unphased, forward endpoint
  }
  return Ws;
}

///////////////////////////////////////////////////////////////////////////////
// coalescedReadRotate: like Grid's coalescedReadPermute (Tensor_SIMT.h) but a
// general cyclic rotation instead of the fixed single-bit XOR swap. Keeps the
// ordinary coalesced (nsimd=Nsimd, one-thread-per-lane on GPU) launch model —
// no need to collapse to nsimd=1/full-vobj processing.
//   GRID_SIMT (GPU): cross-lane extractLane read. Each of the Nsimd SIMT
//     threads independently reads a DIFFERENT lane of the same
//     fully-materialized vobj — fully lane-parallel, no serialization
//     (identical mechanism to coalescedReadPermute's plane = lane ^ mask,
//     generalized to plane = (lane + s) % Nsimd).
//   else (CPU): CPU has no per-lane threads (coalescedRead returns the whole
//     vector unchanged there) — the rotation must be a genuine SIMD shuffle
//     instruction, so this branch calls the recursive tensor `rotate`
//     (Tensor_class.h) on the full vobj and returns it whole, matching
//     coalescedRead's CPU convention of returning the (here: rotated) vector.
///////////////////////////////////////////////////////////////////////////////
template <typename vobj>
accelerator_inline auto coalescedReadRotate(const vobj &vec, int s,
    int lane = acceleratorSIMTlane(vobj::Nsimd())) -> decltype(coalescedRead(vec)) {
#ifdef GRID_SIMT
  int plane = (lane + s) % vobj::Nsimd();
  return extractLane(plane, vec);
#else
  vobj tmp;
  rotate(tmp, vec, s);
  return tmp;
#endif
}

///////////////////////////////////////////////////////////////////////////////
// promoteColourMatrix: promote a scalar ColourMatrix (read directly from the
// _wScalarGrid view) to a calcColourMatrix. GPU: the per-lane calcColourMatrix
// IS the scalar ColourMatrix; CPU: replicate the scalar into a vColourMatrix.
// Mirrors coalescedReadRotate's #ifdef GRID_SIMT per-lane vs whole-vobj
// convention. (The caller reads wsScalar_p[idx] DIRECTLY -- no coalescedRead,
// per the GPU scalar-vobj expr-template constraint.)
///////////////////////////////////////////////////////////////////////////////
accelerator_inline auto promoteColourMatrix(const ColourMatrix &s)
    -> decltype(coalescedRead(vColourMatrix())) {
#ifdef GRID_SIMT
  return s;
#else
  vColourMatrix v;
  for (int lane = 0; lane < vColourMatrix::Nsimd(); lane++)
    insertLane(lane, v, s);
  return v;
#endif
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
class A2ATaskSpinTasteStencil {
public:
  // Explicit typedefs: the 8 types this task body uses, modeled on the
  // stencil worker's block (no legacy macro dependency).
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename ComplexField::vector_object cobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef decltype(coalescedRead(iSinglet<typename vobj::vector_type>())) calcScalar;
  typedef LatticeView<vobj> FermView;
  typedef LatticeView<cobj> ComplexView;

protected:
  std::vector<StagGamma::SpinTastePair> _gammas;
  LatticeGaugeField *_U;
  GridCartesian *_fullGrid; // 4D full grid (E/O joined, non-owning: caller owns)
  std::unique_ptr<GridCartesian> _grid5d; // 5D grid (simd={1,1,1,1,Nsimd})
  int _orthog_dir;          // orthogonal direction (only ex-base member)
  ContractType _contract_type = ContractType::undef;

  std::unique_ptr<PaddedCell> _cell5d;
  GridCartesian *_paddedGrid5d;

  // Flattened per-(gamma,ep,osite) padded offset (deviceVector).
  deviceVector<int> _allOffsets;
  deviceVector<int> _nEpDev;    // nEndpoints per gamma
  deviceVector<int> _epStartDev;// cumulative global endpoint start per gamma
  int _nEpTotal;
  int _osites; // 5D spatial oSites (== grid5d oSites)

  // Site parity for the half/mixed CB modes: parity[ss] = (x+y+z+t)&1 per 5D
  // oSite (dim-4 lane coordinate excluded), matching the GridRedBlackCartesian
  // {1,1,1,1} checker-dim mask that setCheckerboard/pickCheckerboard use to
  // place CB data into full-grid objects (Lattice_transfer.h). Host-built once
  // in the ctor and uploaded (the A2ATaskBase::generateCoorMap device-map
  // precedent, A2ATask.h). Read by the wantParity 0/1 kernel passes; unused
  // by Full.
  deviceVector<int> _siteParity;

  // LHS packed into Nsimd-lane batches + view. Each _lhs5d[b] holds
  // Nsimd distinct L vectors in its dim-5 lanes (was: one replicated L per l).
  std::vector<Lattice<vobj>> _lhs5d;
  std::shared_ptr<A2AFieldView<vobj>> _lhsView;
  int _sizeL;
  int _nBatchesL;

  // Padded RHS 5D batches + view. The RHS batch-count member was renamed to
  // _nBatchesR for symmetry with _nBatchesL (both L and R now batch into
  // Nsimd-lane groups).
  std::vector<Lattice<vobj>> _paddedRight5d;
  std::shared_ptr<A2AFieldView<vobj>> _paddedRhsView;
  int _sizeR;
  int _nBatchesR;
  int _Nsimd;

  // --- Scalar W Lattice + pairing + phase. The kernel reads these
  //     scalar fields directly.
  // W = Lattice<ColourMatrix> on a 5D Nsimd=1 grid (dim-5 = forward endpoints).
  // Populated/read ONLY via direct view indexing (no expr templates).
  std::unique_ptr<GridCartesian> _gridW;
  std::unique_ptr<PaddedCell> _cellW;
  GridCartesian *_paddedGridW = nullptr;
  std::unique_ptr<Lattice<ColourMatrix>> _wScalarGrid; // on _paddedGridW
  int _nFwdEpTotal = 0;  // dim-5 size (total forward endpoints)
  int _paddedOsites = 0; // padded spatial oSites (== paddedGrid5d oSites)
  deviceVector<int> _forwardRepDev;     // [epTotal] global forward-field index
  deviceVector<int> _isForwardDev;      // [epTotal] 0/1
  deviceVector<int> _interiorOffsetDev; // [osites] shift-0 padded interior oSite
  std::vector<ComplexField> _phaseFields;         // [nGamma] grid5d phase
  std::shared_ptr<A2AFieldView<cobj>> _phaseView;

public:
  A2ATaskSpinTasteStencil(GridCartesian *fullGrid, int orthogDir,
                          const std::vector<StagGamma::SpinTastePair> &gammas,
                          LatticeGaugeField *U)
      : _gammas(gammas), _U(U), _fullGrid(fullGrid), _orthog_dir(orthogDir) {

    _grid5d = createGrid5d(fullGrid);
    _Nsimd = fullGrid->Nsimd();
    _osites = _grid5d->oSites();

    _cell5d = std::make_unique<PaddedCell>(1, _grid5d.get());
    _paddedGrid5d = _cell5d->grids.back();

    // Build per-gamma endpoint offsets + nEndpoints/epStart bookkeeping. Only
    // the offset table + endpoint counts are assembled here; the W_s gauge chain
    // is built separately below as the unphased forward scalar store.
    // spinTasteEndpoints gives the endpoint list used both for nEndpoints and
    // buildPaddedOffset5d.
    std::vector<int> allOffsetsHost;
    std::vector<int> nEpHost, epStartHost;
    _nEpTotal = 0;
    int epCum = 0;
    StagGamma spinTaste;

    for (int g = 0; g < (int)gammas.size(); g++) {
      spinTaste.setSpinTaste(gammas[g]);

      std::vector<Coordinate> endpoints = spinTasteEndpoints(spinTaste);

      nEpHost.push_back((int)endpoints.size());
      epStartHost.push_back(epCum);

      // Flatten this gamma's offsets into the global host table. buildPaddedOffset5d
      // returns a host std::vector<int> directly (no device round-trip); the
      // single upload into _allOffsets happens once, after the loop.
      std::vector<int> epOffsets =
          buildPaddedOffset5d(_grid5d.get(), _paddedGrid5d, 1, endpoints);
      allOffsetsHost.insert(allOffsetsHost.end(),
                            epOffsets.begin(), epOffsets.end());

      epCum += (int)endpoints.size();
      _nEpTotal += (int)endpoints.size();
    }

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

    // Parity table for the CB half/mixed modes (consumed by the wantParity
    // 0/1 vectorSum5d passes). Convention identical to setCheckerboard
    // placement: 4D-site parity = (x+y+z+t)&1 (all four dims checker-masked).
    {
      std::vector<int> parityHost(_osites);
      for (int ss = 0; ss < _osites; ss++) {
        Coordinate ocoor(_grid5d->Nd());
        _grid5d->oCoorFromOindex(ocoor, ss);
        parityHost[ss] = (ocoor[0] + ocoor[1] + ocoor[2] + ocoor[3]) & 1;
      }
      _siteParity = upload(parityHost);
    }

    // === Scalar W store + phase + pairing maps ============
    _paddedOsites = _paddedGrid5d->oSites();

    // Forward-W read site: shift-0 padded interior oSite per source ss.
    {
      std::vector<int> interiorHost =
          buildInteriorOffset(_grid5d.get(), _paddedGrid5d, 1);
      _interiorOffsetDev = upload(interiorHost);
    }

    // Phase fields: applyCoeffsAndPhase on a grid5d ones field mirrors
    // local task's phase pre-bake (its ctor). grid5d so ss matches the
    // kernel osites; phase is spatial-only so dim-5 lanes are replicated. This
    // is the SAME phase the old spinTasteGaugeChain baked into W -- applying it
    // here (separate from the unphased W) is what makes the pairing hermiticity
    // exact; the kernel must NOT also receive phased W (double-phase trap).
    {
      ComplexField ones5d(_grid5d.get());
      ones5d = 1.0;
      StagGamma spinTaste;
      _phaseFields.reserve(gammas.size());
      for (int g = 0; g < (int)gammas.size(); g++)
        _phaseFields.emplace_back(_grid5d.get());
      for (int g = 0; g < (int)gammas.size(); g++) {
        spinTaste.setSpinTaste(gammas[g]);
        spinTaste.applyCoeffsAndPhase(_phaseFields[g], ones5d);
      }
      _phaseView = std::make_shared<A2AFieldView<cobj>>();
      _phaseView->openViews(_phaseFields.data(), (int)gammas.size());
    }

    // Pairing maps: classify per gamma; global forward-field index =
    // fwdEpCum + local compact index. Also accumulates _nFwdEpTotal.
    std::vector<int> forwardRepHost, isForwardHost;
    _nFwdEpTotal = 0;
    {
      StagGamma spinTaste;
      int epBase = 0;   // running global endpoint index (== epCum above)
      int fwdEpCum = 0; // running global forward-field index
      for (int g = 0; g < (int)gammas.size(); g++) {
        spinTaste.setSpinTaste(gammas[g]);
        std::vector<Coordinate> endpoints = spinTasteEndpoints(spinTaste);
        int shift = spinTaste._spin ^ spinTaste._taste;
        std::array<int, 4> dirs;
        int nActive = 0;
        for (int j = 0; j < 4; j++)
          if (static_cast<int>(StagGamma::gmu[j]) & shift)
            dirs[nActive++] = j;
        std::vector<Coordinate> fwd;
        std::vector<int> fwdRepLocal;
        std::vector<bool> isFwdLocal;
        classifySpinTasteEndpoints(endpoints, dirs, nActive, fwd, fwdRepLocal,
                                   isFwdLocal);
        int nEp = (int)endpoints.size();
        forwardRepHost.resize(epBase + nEp);
        isForwardHost.resize(epBase + nEp);
        for (int ep = 0; ep < nEp; ep++) {
          forwardRepHost[epBase + ep] = fwdEpCum + fwdRepLocal[ep];
          isForwardHost[epBase + ep] = isFwdLocal[ep] ? 1 : 0;
        }
        epBase += nEp;
        fwdEpCum += (int)fwd.size();
        _nFwdEpTotal += (int)fwd.size();
      }
    }
    _forwardRepDev = upload(forwardRepHost);
    _isForwardDev = upload(isForwardHost);

    // Scalar W Lattice: gridW is a 5D Nsimd=1 grid whose dim-5 indexes the
    // forward endpoints ("directions"); _wScalarGrid lives on its padded grid
    // (spatial matches paddedGrid5d, so interiorOffset/paddedSS align). Build
    // per forward direction: grid4d unphased chain -> promote grid5d -> pad
    // (Exchange) -> unpackScalarW (extractLane 0) into the dim-5 slice. DIRECT
    // view writes only (no expr/coalescedWrite -- GPU scalar-vobj constraint).
    _gridW = createGridW(fullGrid, _nFwdEpTotal);
    _cellW = std::make_unique<PaddedCell>(1, _gridW.get());
    _paddedGridW = _cellW->grids.back();
    _wScalarGrid = std::make_unique<Lattice<ColourMatrix>>(_paddedGridW);
    {
      // Bind a local reference: autoView(l_v, l, mode) expands l.View(mode)
      // verbatim, so a unique_ptr must be dereferenced into a Lattice& first
      // (else *_wScalarGrid.View(mode) binds .View to the unique_ptr).
      Lattice<ColourMatrix> &wScalar = *_wScalarGrid;
      autoView(wsGrid_v, wScalar, AcceleratorWrite);
      int fwdEpCum = 0;
      StagGamma spinTaste;
      for (int g = 0; g < (int)gammas.size(); g++) {
        spinTaste.setSpinTaste(gammas[g]);
        std::vector<Coordinate> endpoints = spinTasteEndpoints(spinTaste);
        int shift = spinTaste._spin ^ spinTaste._taste;
        std::array<int, 4> dirs;
        int nActive = 0;
        for (int j = 0; j < 4; j++)
          if (static_cast<int>(StagGamma::gmu[j]) & shift)
            dirs[nActive++] = j;
        std::vector<Coordinate> fwd;
        std::vector<int> fwdRepLocal;
        std::vector<bool> isFwdLocal;
        classifySpinTasteEndpoints(endpoints, dirs, nActive, fwd, fwdRepLocal,
                                   isFwdLocal);

        std::vector<LatticeColourMatrix> fwdChains =
            spinTasteGaugeChainUnphasedForward(_U, spinTaste, fullGrid, fwd);
        for (int fp = 0; fp < (int)fwdChains.size(); fp++) {
          LatticeColourMatrix chain5d(_grid5d.get());
          promoteField5d(chain5d, fwdChains[fp], fullGrid, _grid5d.get());
          LatticeColourMatrix paddedChain5d = _cell5d->Exchange(chain5d);
          // extract lane 0 -> scalar Lattice dim-5 slice (fwdEpCum+fp).
          // &wsGrid_v[offset] is a ColourMatrix* into the view; unpackScalarW
          // writes dst[ss] directly (no coalescedWrite).
          unpackScalarW(&wsGrid_v[(size_t)(fwdEpCum + fp) * _paddedOsites],
                        paddedChain5d, _paddedOsites);
        }
        fwdEpCum += (int)fwdChains.size();
      }
    }
  }

  // Teardown is safe-by-construction: members destroy in REVERSE
  // declaration order, and the declaration layout encodes the PaddedCell
  // teardown invariant -- ~PaddedCell -> DeleteGrids reads
  // unpadded_grid->_processors, so a cell must die before the grid it wraps.
  // Declaration order gives _wScalarGrid (Lattice on _paddedGridW) -> _cellW
  // (destroys _paddedGridW) -> _gridW, and _cell5d -> _grid5d.
  // Do NOT reorder the data members.
  ~A2ATaskSpinTasteStencil() = default;

  int getNgamma() { return (int)_gammas.size(); }
  double getFlops() {
    // Per (site, l, r) pairing: one W_s ColourMatrix×FermionVec (66 FLOP) + one
    // innerProduct (22 FLOP) = 88 FLOP × nEndpoints. W*psi is hoisted out of the
    // per-L loop (computed once per (l_batch,r_batch,site,ep), reused across the
    // rotation sweep), so actual W*psi FLOPs drop ~Nsimd×; the reported metric is
    // already sizeL/sizeR-independent, so this per-endpoint value stays valid.
    return 88.0 * _nEpTotal;
  }
  // The worker reads the 4D full grid (to size full-grid input temporaries);
  // _fullGrid is protected, so expose it.
  GridCartesian *getFullGrid() const { return _fullGrid; }

  // setLeft: pack full-grid LHS into Nsimd-sized 5D batches, each
  // batch holding Nsimd distinct L in its dim-5 lanes, via the same pack5d
  // setRight uses for RHS. LHS is unshifted,
  // so no PaddedCell::Exchange (unlike setRight).
  void setLeft(const FermionField *left, int size) {
    _sizeL = size;
    _nBatchesL = (size + _Nsimd - 1) / _Nsimd;
    GridBase *grid4d = left[0].Grid();

    // Release the previous view first: it holds open AcceleratorRead locks
    // on the _lhs5d lattices the writes below would conflict with.
    _lhsView = nullptr;

    // Reallocate only if the batch count changed; steady-state repeat calls
    // (same sizeL) reuse the existing Lattice slots via in-place Zero()+pack5d
    // below, avoiding a clear()+reconstruct of the 5D device buffers on every
    // call (setLeft/setRight are address-cache-gated in the worker, but the
    // pointer commonly changes call-to-call at fixed size).
    if ((int)_lhs5d.size() != _nBatchesL) {
      _lhs5d.clear();
      _lhs5d.reserve(_nBatchesL);
      for (int b = 0; b < _nBatchesL; b++)
        _lhs5d.emplace_back(_grid5d.get());
    }

    {
      GRID_TRACE("A2AStencil/PackLeft");
      for (int b = 0; b < _nBatchesL; b++) {
        int nVec = std::min(_Nsimd, size - b * _Nsimd);
        _lhs5d[b] = Zero(); // zero unused lanes of the partial last batch
        pack5d(_lhs5d[b], left + b * _Nsimd, nVec, grid4d, _grid5d.get());
      }
    }
    _lhsView = std::make_shared<A2AFieldView<vobj>>();
    _lhsView->openViews(_lhs5d.data(), _nBatchesL);
  }

  // setRight: pack full-grid RHS into Nsimd-sized 5D batches, pad via Exchange.
  void setRight(const FermionField *right, int size) {
    _sizeR = size;
    _nBatchesR = (size + _Nsimd - 1) / _Nsimd;

    // Release the previous view first (same hazard as setLeft).
    _paddedRhsView = nullptr;

    // Reallocate only if the batch count changed; steady-state repeat calls
    // (same sizeR) reuse the existing Lattice slots via assignment below,
    // avoiding a clear()+reconstruct of the padded-grid device buffers on
    // every call.
    if ((int)_paddedRight5d.size() != _nBatchesR) {
      _paddedRight5d.clear();
      _paddedRight5d.reserve(_nBatchesR);
      for (int b = 0; b < _nBatchesR; b++)
        _paddedRight5d.emplace_back(_paddedGrid5d);
    }

    GridBase *grid4d = right[0].Grid();
    Lattice<vobj> rhs5d(_grid5d.get());
    for (int b = 0; b < _nBatchesR; b++) {
      int nVec = std::min(_Nsimd, size - b * _Nsimd);
      rhs5d = Zero();
      {
        GRID_TRACE("A2AStencil/PackRhs5d");
        pack5d(rhs5d, right + b * _Nsimd, nVec, grid4d, _grid5d.get());
      }
      {
        GRID_TRACE("A2AStencil/HaloExchange");
        _paddedRight5d[b] = _cell5d->Exchange(rhs5d); // assign in place, no emplace_back
      }
    }
    _paddedRhsView = std::make_shared<A2AFieldView<vobj>>();
    _paddedRhsView->openViews(_paddedRight5d.data(), _nBatchesR);
  }

  //....................................................................
  // execute (ContractType dispatch). Full: one unfiltered vectorSum5d pass
  // + circulant scatter. Half/mixed (LeftHalf/RightHalf/BothHalf): two
  // parity-filtered passes (even/odd source sites) into a parity-split
  // scratch, then assembleMat applies the placement table.
  //....................................................................
  void execute(scalar_type *result_p,
               ContractType contract_type = ContractType::Full) {
    _contract_type = contract_type;
    if (contract_type == ContractType::undef) {
      std::cerr << "A2ATaskSpinTasteStencil::execute: ContractType undef"
                << std::endl;
      GridAbort();
    }
    int nGamma = getNgamma();
    int orthogDir = this->_orthog_dir;
    int localSpatialVolume = _grid5d->_ostride[orthogDir];
    int localOrthogDimSize = _grid5d->_ldimensions[orthogDir];
    int pc = _grid5d->_processor_coor[orthogDir];
    int Nt = _grid5d->GlobalDimensions()[orthogDir];

    bool cbSplit = (contract_type != ContractType::Full);
    int nParity = cbSplit ? 2 : 1;

    // shm_p holds cobj (SIMD) elements indexed, fastest-to-slowest, as
    // (rt, s, l_batch, r_batch, mu) -- rt stays innermost. s is positioned
    // at stride localOrthogDimSize
    // right where a sub-index of l_batch belongs (for fixed lane j, s and the
    // intra-batch L index i are in 1-1 correspondence). Each cobj still
    // encodes Nsimd lanes (extractLane in assembly), so no extra Nsimd factor
    // beyond s. Half/mixed adds one further parity slice: pass p (even/odd
    // source sites) writes into shm_p + p * nGamma * gammaStride.
    int gammaStride = _nBatchesL * _nBatchesR * localOrthogDimSize * _Nsimd;

    cobj *shm_p = static_cast<cobj *>(acceleratorAllocDevice(
        gammaStride * nGamma * nParity * sizeof(cobj)));
    {
      GRID_TRACE("A2AStencil/ZeroScratch");
      accelerator_for(idx, gammaStride * nGamma * nParity, 1,
                      { shm_p[idx] = Zero(); });
    }

    {
      GRID_TRACE("A2AStencil/VectorSum5d");
      for (int p = 0; p < nParity; p++) {
        for (int mu = 0; mu < nGamma; mu += MF_SUM_ARRAY_MAX) {
          int nGammaBlock = std::min(nGamma - mu, MF_SUM_ARRAY_MAX);
          vectorSum5d(shm_p + (size_t)p * nGamma * gammaStride, mu,
                      nGammaBlock, localOrthogDimSize, localSpatialVolume,
                      gammaStride, _siteParity.data(), cbSplit ? p : -1);
        }
      }
    }

    assembleMat(result_p, shm_p, contract_type, nGamma, gammaStride,
                localOrthogDimSize, pc, Nt);

    acceleratorFreeDevice(shm_p);
  }

  //....................................................................
  // assembleMat: circulant-scatter assembly. lane j == r-local; i == l-local;
  // pairing (l_i, r_j) lives at rotation s=(i-j) mod Nsimd, lane j. Per-lane
  // scalar work -> portable acceleratorSIMTlane pattern (matches pack5d).
  // Launched as (l_batch, r_batch), Nsimd (accelerator_for2d); mu is a plain
  // serial loop inside (no per-mu accumulator state to chunk, unlike
  // vectorSum5d's MF_SUM_ARRAY_MAX blocking).
  //
  // CB output-placement: the even/odd placement
  // truth tables for the doubled output slots, keyed on
  // (cbEven x oddShifts); expressed per parity (M0 = even source-site sum;
  // M1 = odd; sigma = -1 for odd popcount, else +1):
  //
  //   BothHalf  slots (row 2l+lc, col 2r+rc):
  //     (0,0): M0+M1          (0,1): sigma*(M0-M1)
  //     (1,0): M0-M1          (1,1): sigma*(M0+M1)
  //   LeftHalf  slots (row 2l+lc, col r):
  //     (0): M0+M1            (1): M0-M1
  //   RightHalf slots (row l, col 2r+rc):
  //     (0): M0+M1            (1): sigma*(M0-M1)
  //
  // Derivation: every endpoint of a gamma displaces by popcount hops, so
  // parity(x+s) = parity(x) ^ (pc%2); with CB copies packed into full
  // objects, the even-site pass pairs L_E with R_E (even pc) or R_O
  // (odd pc). Reducing the cbEven/oddShifts sign tables modulo that
  // routing yields the slot forms above.
  //....................................................................
  void assembleMat(scalar_type *result_p, cobj *shm_p, ContractType ct,
                   int nGamma, int gammaStride, int localOrthogDimSize,
                   int pc, int Nt) {
    bool cbL = ((int)ct & (int)ContractType::LeftHalf) != 0;
    bool cbR = ((int)ct & (int)ContractType::RightHalf) != 0;
    bool cbSplit = (ct != ContractType::Full);
    int nLC = cbL ? 2 : 1;   // l-slot multiplicity (packed pairs doubled)
    int nRC = cbR ? 2 : 1;   // r-slot multiplicity
    int matL = _sizeL * nLC; // output rows
    int matR = _sizeR * nRC; // output cols
    int parityStride = nGamma * gammaStride; // even slice | odd slice

    // Copy members to locals: the GPU lambda captures a CPU `this`.
    auto shm_p_a = shm_p;
    auto result_p_a = result_p;
    int sizeL_a = _sizeL;
    int sizeR_a = _sizeR;
    int nBatchesL_a = _nBatchesL;
    int nBatchesR_a = _nBatchesR;
    int Nsimd = _Nsimd;
    int matL_a = matL;
    int matR_a = matR;
    int nLC_a = nLC;
    int nRC_a = nRC;
    int parityStride_a = parityStride;
    bool cbSplit_a = cbSplit;
    auto nEp_p = _nEpDev.data();
    {
      GRID_TRACE("A2AStencil/Assemble");
      accelerator_for2d(l_batch, nBatchesL_a, r_batch, nBatchesR_a, Nsimd, {
#ifdef GRID_SIMT
        {
          int j = acceleratorSIMTlane(Nsimd);
#else
        for (int j = 0; j < Nsimd; j++) {
#endif
          int r = r_batch * Nsimd + j;
          if (r < sizeR_a) {
            for (int mu = 0; mu < nGamma; mu++) {
              // sigma = -1 for odd popcount: nEp = 2^popcount, so
              // nEp in {2, 8} <=> popcount in {1, 3}.
              scalar_type sig =
                  (nEp_p[mu] == 2 || nEp_p[mu] == 8) ? scalar_type(-1.0)
                                                     : scalar_type(1.0);
              for (int rt = 0; rt < localOrthogDimSize; rt++) {
                int gt = rt + pc * localOrthogDimSize; // T simd=1
                for (int i = 0; i < Nsimd; i++) {
                  int l = l_batch * Nsimd + i;
                  if (l < sizeL_a) {
                    int s = ((i - j) % Nsimd + Nsimd) % Nsimd;
                    int shm_idx =
                        rt +
                        localOrthogDimSize *
                            (s + Nsimd * (l_batch + nBatchesL_a * r_batch)) +
                        mu * gammaStride;
                    auto M0 = TensorRemove(extractLane(j, shm_p_a[shm_idx]));
                    if (!cbSplit_a) {
                      // Full: one slot per (l, r), single parity slice, each
                      // slot written exactly once (plain assignment keeps
                      // execute() free of a pre-zeroed-result dependency).
                      int64_t mat_idx = (int64_t)mu * matL_a * matR_a * Nt +
                                         r + matR_a * (l + matL_a * gt);
                      result_p_a[mat_idx] = M0;
                    } else {
                      auto M1 = TensorRemove(extractLane(
                          j, shm_p_a[shm_idx + parityStride_a]));
                      for (int lc = 0; lc < nLC_a; lc++) {
                        for (int rc = 0; rc < nRC_a; rc++) {
                          scalar_type sA, sB; // slot = sA*M0 + sB*M1
                          if (cbL && cbR) { // BothHalf
                            sA = (rc == 1) ? sig : scalar_type(1.0);
                            if (lc == 0 && rc == 0)
                              sB = scalar_type(1.0);
                            else if (lc == 0 && rc == 1)
                              sB = -sig;
                            else if (lc == 1 && rc == 0)
                              sB = scalar_type(-1.0);
                            else
                              sB = sig;
                          } else if (cbL) { // LeftHalf
                            sA = scalar_type(1.0);
                            sB = (lc == 0) ? scalar_type(1.0)
                                           : scalar_type(-1.0);
                          } else { // RightHalf
                            sA = (rc == 0) ? scalar_type(1.0) : sig;
                            sB = (rc == 0) ? scalar_type(1.0) : -sig;
                          }
                          // int64_t: BothHalf quadruples the flattened
                          // output; int would overflow the mu stride at
                          // 4x smaller sizes than Full (silent wrong slots).
                          int64_t mat_idx =
                              (int64_t)mu * matL_a * matR_a * Nt +
                              nRC_a * r + rc +
                              matR_a * (nLC_a * l + lc + matL_a * gt);
                          result_p_a[mat_idx] += sA * M0 + sB * M1;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      });
    }
  }

  //....................................................................
  // vectorSum5d: fused kernel, shared by the Full and CB half/mixed modes
  // (the name drops "Full" accordingly). Ordinary coalesced (nsimd=Nsimd)
  // launch. Loop order: (lt_batch=[l_batch,rt], r_batch) parallel; (mu, ep,
  // so) serial.
  //
  // The time-slice index rt is folded into the first parallel dimension so that
  // each (l_batch, rt, r_batch) triple gets its own block. For pc=0 this gives
  // nBatchesL * Nt * nBatchesR = 3 * 48 * 38 = 5472 blocks vs the previous 114,
  // filling the 108 SMs and allowing the GPU scheduler to hide L2 latency via
  // warp switching. Each rt writes to a distinct shm slot so no reduction is
  // needed.
  //
  // ep and mu remain outside the so loop so that for fixed (ep, mu) the psi
  // gather reads traverse the padded RHS buffer sequentially
  // (paddedSS = offsets[ep][ss] is a uniform displacement as ss increments),
  // enabling hardware prefetching.
  //
  // wantParity: -1 (Full) accumulates every source site; 0/1 (half/mixed)
  // keep only even/odd-parity sites, so two passes give the assemble step
  // separate even/odd partial sums from the same kernel.
  //....................................................................
  void vectorSum5d(cobj *shm_p, int mu_offset, int nGamma,
                   int localOrthogDimSize, int localSpatialVolume,
                   int gammaStride, const int *siteParity_p, int wantParity) {
    GRID_TRACE("A2AStencil/vectorSum5d");
    int nBatchesL = _nBatchesL;
    int nBatchesR = _nBatchesR;
    int Nsimd = _Nsimd;
    int osites = _osites;
    int paddedOsites = _paddedOsites;

    // Scalar W store: bind a local Lattice& before autoView -- the
    // macro expands w.View(mode) verbatim, so a unique_ptr must be dereferenced
    // first (else *_wScalarGrid.View(mode) binds .View to the unique_ptr).
    Lattice<ColourMatrix> &wScalar = *_wScalarGrid;
    autoView(wsGrid_v, wScalar, AcceleratorRead);
    ColourMatrix *wsScalar_p = &wsGrid_v[0];         // [globalFwdEp][paddedOsites]
    FermView *lhsView_p = _lhsView->getView();       // [l_batch][ss]
    FermView *rhsView_p = _paddedRhsView->getView(); // [r_batch][paddedSS]
    ComplexView *phaseView_p = _phaseView->getView() + mu_offset; // [mu][ss]

    auto offsets_p = _allOffsets.data();        // [(epStart+ep)*osites + ss]
    auto nEp_p = _nEpDev.data();
    auto epStart_p = _epStartDev.data();
    auto forwardRep_p = _forwardRepDev.data();  // [globalEp] -> global fwd-field idx
    auto isForward_p = _isForwardDev.data();    // [globalEp] 0/1
    auto interiorOffset_p = _interiorOffsetDev.data(); // [ss]

    int nBatchesLT = nBatchesL * localOrthogDimSize;
    accelerator_for2d(lt_batch, nBatchesLT, r_batch, nBatchesR, Nsimd, {
      int l_batch = lt_batch / localOrthogDimSize;
      int rt      = lt_batch % localOrthogDimSize;

      calcScalar sum[vobj::Nsimd()][MF_SUM_ARRAY_MAX];
      for (int s = 0; s < Nsimd; s++)
        for (int mu = 0; mu < nGamma; mu++)
          sum[s][mu] = Zero();

      for (int mu = 0; mu < nGamma; mu++) {
        int g = mu_offset + mu;
        int nEp = nEp_p[g];
        int epStart = epStart_p[g];
        for (int ep = 0; ep < nEp; ep++) {
          int ge = epStart + ep;           // global endpoint index
          int fwdEp = forwardRep_p[ge];    // global forward-field index
          bool isFwd = isForward_p[ge] != 0;
          for (int so = 0; so < localSpatialVolume; so++) {
            int ss = rt * localSpatialVolume + so;
            // CB parity split: the non-matching pass handles this site.
            if (wantParity >= 0 && siteParity_p[ss] != wantParity)
              continue;
            int paddedSS = offsets_p[(size_t)ge * osites + ss];
            auto psi = coalescedRead(rhsView_p[r_batch][paddedSS]);
            // W read site: forward -> source x (interiorOffset, local);
            // backward -> x-s (paddedSS, halo) + adj. W is UNPHASED C_s.
            int wSS = isFwd ? interiorOffset_p[ss] : paddedSS;
            auto W = promoteColourMatrix(wsScalar_p[(size_t)fwdEp * paddedOsites + wSS]);
            if (!isFwd)
              W = adj(W);
            auto Wpsi = W * psi;
            // Follana phase + scaling at the SOURCE site x. W is unphased
            // so the {s,-s} pairing hermiticity is exact (corr == +1). Phase is
            // ep-independent; applied per-ep here (the ep-outside-so
            // loop order keeps the RHS buffer sequentially-accessible;
            // mathematically phase * sum_ep IP == sum_ep phase*IP).
            calcScalar gamma_phase = coalescedRead(phaseView_p[mu][ss]);
            for (int s = 0; s < Nsimd; s++)
              sum[s][mu] = sum[s][mu] + gamma_phase * innerProduct(
                  coalescedReadRotate(lhsView_p[l_batch][ss], s), Wpsi);
          }
        }
      }

      for (int s = 0; s < Nsimd; s++)
        for (int mu = 0; mu < nGamma; mu++) {
          int shm_idx =
              rt +
              localOrthogDimSize * (s + Nsimd * (l_batch + nBatchesL * r_batch)) +
              (mu_offset + mu) * gammaStride;
          coalescedWrite(shm_p[shm_idx], sum[s][mu]);
        }
    });
  }

};

NAMESPACE_END(Grid);
