/******************************************************************************/
/* A2ATaskStencil.h -- 5D stencil spin-taste meson-field task + its file-scope */
/* helpers, split off the legacy task machinery.                             */
/* Self-contained (T1): depends on no legacy a2a worker/task header. The task */
/* owns its state directly (no base-class inheritance). */
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
// U may be null when the spin-taste has zero displacement (nActive == 0):
// no links are read and each chain is the identity.
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

  // Null-U tolerance: a zero-displacement spin-taste (nActive == 0) reads no
  // gauge links -- its covariant chain is identically the identity -- so the
  // Udir4d peek loop below is skipped and U may be null for all-local gamma
  // sets. Any displacing gamma still dereferences U; assert the caller did
  // not pass null there.
  assert((nActive == 0) || (U != nullptr));

  std::vector<LatticeColourMatrix> Udir4d;
  if (nActive > 0) {
    Udir4d.reserve(Nd);
    for (int mu = 0; mu < Nd; mu++) {
      Udir4d.emplace_back(grid4d);
      Udir4d[mu] = PeekIndex<LorentzIndex>(*U, mu);
    }
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
          chain = Udir4d[dir] * Cshift(chain, dir, +1); // CovShiftForward
        } else {
          chain = Cshift(adj(Udir4d[dir]) * chain, dir, -1); // CovShiftBackward
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
// J.A local spec set: n zero-displacement spin-tastes (G1,G1). The specs
// drive the endpoint/pairing/W machinery ONLY (single identity endpoint
// per channel, W == 1, no gauge field); the per-channel site phase comes
// from the caller's Amu fields (extPhases), so these objects are never
// consumed for their phases. Validated here (not in the delegating body)
// so the check fires BEFORE the canonical ctor builds any state.
///////////////////////////////////////////////////////////////////////////////
inline std::vector<StagGamma> jaLocalSpecs(int n) {
  if (n <= 0) {
    std::cerr << "A2ATaskSpinTasteStencil: J.A local requires at least one "
                 "Amu phase field"
              << std::endl;
    GridAbort();
  }
  std::vector<StagGamma> specs(n);
  for (int i = 0; i < n; i++)
    specs[i].setSpinTaste(StagGamma::StagAlgebra::G1,
                          StagGamma::StagAlgebra::G1);
  return specs;
}

///////////////////////////////////////////////////////////////////////////////
// J.A one-link spec set: one pc1 spin-taste (gmu[k], G1) per direction k
// (X,Y,Z,T order -- gmu[k] activates exactly direction k): endpoints ±ê_k,
// the standard η_k(x)·(1/2) coefficient baked via the applyCoeffsAndPhase
// path, and the one-hop chain taken from the A_k-dressed gauge field.
///////////////////////////////////////////////////////////////////////////////
inline std::vector<StagGamma> jaOneLinkSpecs() {
  std::vector<StagGamma> specs;
  specs.reserve(StagGamma::gmu.size());
  for (int k = 0; k < (int)StagGamma::gmu.size(); k++)
    specs.emplace_back(StagGamma::gmu[k], StagGamma::StagAlgebra::G1);
  return specs;
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
  std::vector<StagGamma> _gammas; // was: bare pair vector
  LatticeGaugeField *_U;
  GridCartesian *_fullGrid; // 4D full grid (E/O joined, non-owning: caller owns)
  // 5D grid (simd={1,1,1,1,Nsimd}). shared_ptr: either created from fullGrid
  // (createGrid5d) or the ADOPTED caller grid (co-owned with the producer so
  // it outlives the task's views; validated in the ctor). Declaration
  // position unchanged -- the PaddedCell teardown invariant below depends on
  // member order, and shared_ptr destroys at the same position.
  std::shared_ptr<GridCartesian> _grid5d;
  int _orthog_dir; // orthogonal direction (only ex-base member)
  ContractType _contract_type = ContractType::undef;

  std::unique_ptr<PaddedCell> _cell5d;
  GridCartesian *_paddedGrid5d;

  // Flattened per-(gamma,ep,osite) padded offset (deviceVector).
  deviceVector<int> _allOffsets;
  deviceVector<int> _nEpDev;     // nEndpoints per gamma
  deviceVector<int> _epStartDev; // cumulative global endpoint start per gamma
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

  // Borrowed 5D LHS (zero-copy setLeft): shared_ptr copies pin the caller's
  // lattices for the lifetime of _lhsView's open views. Empty on the 4D pack
  // path. RHS needs no counterpart -- setRight's Exchange copies into
  // task-owned _paddedRight5d, so nothing is borrowed past its return.
  std::vector<std::shared_ptr<Lattice<vobj>>> _lhsBorrowed;

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
  deviceVector<int> _forwardRepDev; // [epTotal] global forward-field index
  deviceVector<int> _isForwardDev;  // [epTotal] 0/1
  deviceVector<int> _interiorOffsetDev; // [osites] shift-0 padded interior oSite
  std::vector<ComplexField> _phaseFields; // [nGamma] grid5d phase
  std::shared_ptr<A2AFieldView<cobj>> _phaseView;
  // Momentum phases [nMom] on grid5d (lane-replicated; promoted from the
  // caller's 4D full-grid fields). Empty == no momentum (_nMom == 1): the
  // kernel skips the momentum multiply entirely (zero-mom path
  // arithmetically identical to the pre-momentum kernel). Declaration
  // order: fields BEFORE view (the PaddedCell/layout teardown invariant
  // above; mirrors _phaseFields/_phaseView).
  std::vector<ComplexField> _momFields;
  std::shared_ptr<A2AFieldView<cobj>> _momView;
  int _nSpec = 0; // spec gamma count (_gammas.size(); endpoint tables per spec)
  int _nMom = 1;  // momentum multiplicity of the flat mu axis (>= 1)

public:
  // Zero-momentum form: unchanged public surface (existing call sites).
  A2ATaskSpinTasteStencil(GridCartesian *fullGrid, int orthogDir,
                          const std::vector<StagGamma> &gammas,
                          LatticeGaugeField *U,
                          std::shared_ptr<GridCartesian> grid5d = nullptr)
      : A2ATaskSpinTasteStencil(fullGrid, orthogDir, gammas, U,
                                std::vector<ComplexField>(), grid5d) {}

  // Canonical ctor (momentum-aware; extPhases added for the J.A local path,
  // Slice 3). mom holds one exp(i p·x) phase field per momentum (4D full
  // grid, pointer-identical to fullGrid). The output mu axis becomes
  // nGamma*nMom, g-major (mu = g*nMom + p): allocate a RowMajor
  // (nGamma, nMom, Nt, [2]L, R) tensor and read mf(g, p, t, l, r). The
  // kernel computes the expensive W*psi*innerProduct once per gamma and
  // multiply-accumulates each momentum into its own accumulator slot.
  // extPhases (J.A local): per-spec external phase fields replacing the
  // applyCoeffsAndPhase bake.
  A2ATaskSpinTasteStencil(GridCartesian *fullGrid, int orthogDir,
                          const std::vector<StagGamma> &gammas,
                          LatticeGaugeField *U,
                          const std::vector<ComplexField> &mom,
                          std::shared_ptr<GridCartesian> grid5d = nullptr,
                          const std::vector<ComplexField> *extPhases = nullptr,
                          const std::vector<ComplexField> *extDress = nullptr)
      : _gammas(gammas), _U(U), _fullGrid(fullGrid), _orthog_dir(orthogDir) {

    // 5D grid: adopt the caller's (validated) grid when supplied -- the
    // zero-copy setLeft/setRight overloads require inputs on exactly this
    // grid object (pointer identity) -- or create the canonical one. The
    // unique_ptr rvalue converts to shared_ptr implicitly.
    if (grid5d) {
      if (!grid5dCompatible(fullGrid, grid5d.get())) {
        std::cerr << "A2ATaskSpinTasteStencil: supplied grid5d is not "
                     "layout-compatible with createGrid5d(fullGrid)"
                  << std::endl;
        GridAbort();
      }
      _grid5d = grid5d;
    } else {
      _grid5d = createGrid5d(fullGrid);
    }
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

    for (int g = 0; g < (int)gammas.size(); g++) {
      std::vector<Coordinate> endpoints = spinTasteEndpoints(gammas[g]);

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
      // Phase source: the applyCoeffsAndPhase bake (StagGamma paths) or
      // the caller's external per-spec fields (J.A local). External fields
      // are validated (4D full grid, pointer-identical) and promoted
      // lane-replicated onto grid5d -- the caller's fields ARE the phase
      // (already carrying any coefficient convention; NOTHING is added).
      if (extPhases) {
        if ((int)extPhases->size() != (int)gammas.size()) {
          std::cerr << "A2ATaskSpinTasteStencil: external phase count "
                    << extPhases->size() << " != spec count "
                    << gammas.size() << std::endl;
          GridAbort();
        }
        for (int g = 0; g < (int)extPhases->size(); g++)
          if ((*extPhases)[g].Grid() != fullGrid) {
            std::cerr << "A2ATaskSpinTasteStencil: external phase [" << g
                      << "] must live on the task's 4D full grid "
                         "(pointer identity)"
                      << std::endl;
            GridAbort();
          }
        _phaseFields.reserve(extPhases->size());
        for (int g = 0; g < (int)extPhases->size(); g++) {
          _phaseFields.emplace_back(_grid5d.get());
          promoteField5d(_phaseFields[g], (*extPhases)[g], fullGrid,
                         _grid5d.get());
        }
      } else {
        ComplexField ones5d(_grid5d.get());
        ones5d = 1.0;
        _phaseFields.reserve(gammas.size());
        for (int g = 0; g < (int)gammas.size(); g++)
          _phaseFields.emplace_back(_grid5d.get());
        for (int g = 0; g < (int)gammas.size(); g++) {
          // SIGN-CRITICAL: the STORED object's _negated (applyG5-folded when
          // constructed that way) is baked into _phaseFields. The double-phase
          // trap comment above (W reaching the kernel must be unphased) still
          // applies: the phase is baked exactly once, here.
          gammas[g].applyCoeffsAndPhase(_phaseFields[g], ones5d);
        }
      }
      _phaseView = std::make_shared<A2AFieldView<cobj>>();
      _phaseView->openViews(_phaseFields.data(), (int)_phaseFields.size());
    }

    // Momentum phases: promoted from the caller's 4D full-grid fields to
    // lane-replicated grid5d (same grid/replication contract as the phase
    // bake above). NOT composed with the per-gamma phases here: the kernel
    // multiplies them per-source on the hoisted inner product (the
    // expensive W*psi*innerProduct runs once per gamma regardless of nMom)
    // -- composing here AND multiplying in the kernel would double-phase.
    _nSpec = (int)gammas.size();
    _nMom = mom.empty() ? 1 : (int)mom.size();
    if (!mom.empty()) {
      for (int p = 0; p < (int)mom.size(); p++)
        if (mom[p].Grid() != fullGrid) {
          std::cerr << "A2ATaskSpinTasteStencil: mom[" << p << "] must live "
                       "on the task's 4D full grid (pointer identity)"
                    << std::endl;
          GridAbort();
        }
      _momFields.reserve(mom.size());
      for (int p = 0; p < (int)mom.size(); p++) {
        _momFields.emplace_back(_grid5d.get());
        promoteField5d(_momFields[p], mom[p], fullGrid, _grid5d.get());
      }
      _momView = std::make_shared<A2AFieldView<cobj>>();
      _momView->openViews(_momFields.data(), (int)_momFields.size());
    }

    // Pairing maps: classify per gamma; global forward-field index =
    // fwdEpCum + local compact index. Also accumulates _nFwdEpTotal.
    std::vector<int> forwardRepHost, isForwardHost;
    _nFwdEpTotal = 0;
    {
      int epBase = 0;   // running global endpoint index (== epCum above)
      int fwdEpCum = 0; // running global forward-field index
      for (int g = 0; g < (int)gammas.size(); g++) {
        std::vector<Coordinate> endpoints = spinTasteEndpoints(gammas[g]);
        int shift = gammas[g]._spin ^ gammas[g]._taste;
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

    if (extDress) {
      if (extPhases) {
        std::cerr << "A2ATaskSpinTasteStencil: extPhases and extDress are "
                     "mutually exclusive phase/W sources"
                  << std::endl;
        GridAbort();
      }
      if ((int)extDress->size() != (int)StagGamma::gmu.size() ||
          (int)extDress->size() != (int)gammas.size()) {
        std::cerr << "A2ATaskSpinTasteStencil: link dressing requires "
                     "exactly Nd="
                  << StagGamma::gmu.size() << " fields (one per direction, "
                     "component-indexed) matching the spec count; got "
                  << extDress->size() << " fields for " << gammas.size()
                  << " specs" << std::endl;
        GridAbort();
      }
      if (_U == nullptr) {
        std::cerr << "A2ATaskSpinTasteStencil: link dressing requires a "
                     "gauge field U (one-link displacement)"
                  << std::endl;
        GridAbort();
      }
      for (int k = 0; k < (int)extDress->size(); k++) {
        if ((*extDress)[k].Grid() != fullGrid) {
          std::cerr << "A2ATaskSpinTasteStencil: link dressing field [" << k
                    << "] must live on the task's 4D full grid "
                       "(pointer identity)"
                    << std::endl;
          GridAbort();
        }
        int pc = StagGamma::popcountShift(gammas[k]._spin, gammas[k]._taste);
        if (pc != 1) {
          std::cerr << "A2ATaskSpinTasteStencil: link dressing requires "
                       "popcount-1 specs; spec "
                    << k << " has popcount " << pc << std::endl;
          GridAbort();
        }
      }
    }
    // J.A one-link (extDress): dress each direction-k gauge link with the
    // caller's photon field A_k(x) BEFORE the chain/padded store is built
    // (A_mu lives on the links alongside U_mu -- developer directive). The
    // pc1 chain for spec k then transports through W̃_k(x) = A_k(x)·U_k(x);
    // the kernel's backward read adj(W̃)(x-ê_k) yields A*_k(x-ê)U†_k(x-ê)
    // -- link-symmetric vertex placement with zero kernel changes.
    LatticeGaugeField UchainDressed(fullGrid);
    LatticeGaugeField *Uchain = _U;
    if (extDress) {
      UchainDressed = *_U;
      for (int k = 0; k < (int)StagGamma::gmu.size(); k++) {
        LatticeColourMatrix link = PeekIndex<LorentzIndex>(*_U, k);
        link = (*extDress)[k] * link; // W̃_k(x) = A_k(x) · U_k(x)
        PokeIndex<LorentzIndex>(UchainDressed, link, k);
      }
      Uchain = &UchainDressed;
    }

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
      for (int g = 0; g < (int)gammas.size(); g++) {
        std::vector<Coordinate> endpoints = spinTasteEndpoints(gammas[g]);
        int shift = gammas[g]._spin ^ gammas[g]._taste;
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
            spinTasteGaugeChainUnphasedForward(Uchain, gammas[g], fullGrid,
                                               fwd);
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

  // J.A local ctor: the caller's Amu fields ARE the per-channel site phase
  // (4D full grid, pointer-identical to fullGrid; any coefficient
  // convention already folded caller-side -- nothing is added here). Each
  // channel is a zero-displacement operator (single identity endpoint, no
  // gauge field: U == nullptr is legal, all specs nActive == 0). Arbitrary
  // channel count (the ASlash stochastic-field pattern). Output mu axis:
  // nAmu*nMom g-major -- RowMajor (nAmu, nMom, Nt, [2]L, R), mf(a, p, t, l, r).
  A2ATaskSpinTasteStencil(GridCartesian *fullGrid, int orthogDir,
                          const std::vector<ComplexField> &Amu,
                          const std::vector<ComplexField> &mom,
                          std::shared_ptr<GridCartesian> grid5d = nullptr)
      : A2ATaskSpinTasteStencil(
            fullGrid, orthogDir, jaLocalSpecs((int)Amu.size()), nullptr,
            mom, grid5d, &Amu) {}

  // J.A one-link ctor: exactly Nd (4) photon fields, Amu[k] dressing the
  // direction-k gauge links (W̃_k(x) = A_k(x)·U_k(x); the kernel's backward
  // read adj(W̃)(x-ê_k) carries A*_k(x-ê)U†_k(x-ê) -- link-symmetric
  // placement, zero kernel changes). The standard pc1 spin-taste
  // coefficient η_k(x)·(1/2) is baked into the phase slot (spec gamma
  // (gmu[k], G1)): the operator is the standard one-link bilinear with
  // photon-dressed transport. One output channel per component (J_kA_k);
  // a 3-vector current passes a zero T-component field.
  A2ATaskSpinTasteStencil(GridCartesian *fullGrid, int orthogDir,
                          const std::vector<ComplexField> &Amu,
                          LatticeGaugeField *U,
                          const std::vector<ComplexField> &mom,
                          std::shared_ptr<GridCartesian> grid5d = nullptr)
      : A2ATaskSpinTasteStencil(fullGrid, orthogDir, jaOneLinkSpecs(), U,
                                mom, grid5d, nullptr, &Amu) {}

  // Teardown is safe-by-construction: members destroy in REVERSE
  // declaration order, and the declaration layout encodes the PaddedCell
  // teardown invariant -- ~PaddedCell -> DeleteGrids reads
  // unpadded_grid->_processors, so a cell must die before the grid it wraps.
  // Declaration order gives _wScalarGrid (Lattice on _paddedGridW) -> _cellW
  // (destroys _paddedGridW) -> _gridW, and _cell5d -> _grid5d.
  // Do NOT reorder the data members.
  ~A2ATaskSpinTasteStencil() = default;

  // Flat mu-axis count. With momentum the axis is nSpec*nMom, g-major
  // (mu = g*nMom + p): execute()/assembleMat stay extent-driven over this
  // count. Zero momentum (_nMom == 1) reduces to the spec count.
  int getNgamma() { return _nSpec * _nMom; }
  double getFlops() {
    // Per (site, l, r) pairing: one W_s ColourMatrix x FermionVec (66 FLOP)
    // + one innerProduct (22 FLOP) = 88 FLOP x nEndpoints, computed ONCE
    // per spec gamma and shared across all momenta (the kernel hoists
    // W*psi*innerProduct out of the momentum loop); each extra momentum
    // adds one complex multiply-accumulate (8 FLOP) per endpoint. W*psi is
    // hoisted out of the per-L loop as before (computed once per
    // (l_batch, r_batch, site, ep), reused across the rotation sweep), so
    // the reported metric stays sizeL/sizeR-independent.
    return (88.0 + 8.0 * (_nMom - 1)) * _nEpTotal;
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
    _lhsBorrowed.clear(); // release stale borrowed pins (5D -> 4D switch)

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

  // setLeft (5D zero-copy form): borrow caller-owned, already 5D-batched L
  // (dim-5 lane = vector index) directly -- no Zero, no pack5d, no _lhs5d
  // storage. CONTRACT:
  //  - every element must live on EXACTLY this task's _grid5d object
  //    (pointer identity -- construct the task with the same shared_ptr
  //    grid5d the producer built its lattices on);
  //  - left5d.size() == ceil(size/Nsimd);
  //  - the partial last batch's unused lanes (>= size - b*Nsimd) must be
  //    ZERO by the caller before handoff (borrowed memory is opened
  //    AcceleratorRead; the kernel's l<sizeL discard guard makes finite
  //    garbage harmless, but uninitialised device memory may be NaN/Inf);
  //  - contents are immutable while borrowed (through execute()); mutate
  //    only after a fresh setLeft/setRight call or task destruction. The
  //    worker's address cache (A2AWorkerStencil) gates on the first
  //    element's pointer: same vector + mutation = stale results unless
  //    resetCache().
  // The task retains the shared_ptr vector, so the caller may drop its own
  // references immediately -- destruction cannot dangle the open views.
  void setLeft(const std::vector<std::shared_ptr<Lattice<vobj>>> &left5d,
               int size) {
    _sizeL = size;
    _nBatchesL = (size + _Nsimd - 1) / _Nsimd;
    if ((int)left5d.size() != _nBatchesL) {
      std::cerr << "A2ATaskSpinTasteStencil::setLeft(5d): left5d.size()="
                << left5d.size() << " != nBatches=" << _nBatchesL
                << " for size=" << size << std::endl;
      GridAbort();
    }
    for (int b = 0; b < _nBatchesL; b++)
      if ((*left5d[b]).Grid() != _grid5d.get()) {
        std::cerr << "A2ATaskSpinTasteStencil::setLeft(5d): left5d[" << b
                  << "] is not on this task's _grid5d (pointer identity "
                     "required -- construct the task with the producer's "
                     "grid5d)" << std::endl;
        GridAbort();
      }
    // Release the previous view FIRST (0069a57d ordering): it may hold open
    // AcceleratorRead locks on _lhs5d or previously borrowed lattices.
    _lhsView = nullptr;
    // Free the owned packed batches -- part of the win (sizeL*V of device
    // memory); a later 4D setLeft reallocates through its existing branch.
    _lhs5d.clear();
    // Pin the caller's lattices for the views' lifetime.
    _lhsBorrowed = left5d;
    _lhsView = std::make_shared<A2AFieldView<vobj>>();
    _lhsView->openViews(_lhsBorrowed);
  }

  // setRight (5D pack-skip form): RHS already 5D-batched -- skip Zero+pack5d,
  // but the halo Exchange into task-owned _paddedRight5d is structurally
  // mandatory (the kernel gathers RHS at displaced offsets). Unlike setLeft
  // (5d), NOTHING is borrowed past the return: Exchange copies into padded
  // storage, so the caller may mutate or free right5d immediately after
  // this call returns. Same grid/batch-count contract as setLeft (5d)
  // (pointer identity to _grid5d; partial-batch lanes zeroed by caller --
  // pack5d's lane contract, StencilGather5d.h:122, applies verbatim).
  void setRight(const std::vector<std::shared_ptr<Lattice<vobj>>> &right5d,
                int size) {
    _sizeR = size;
    _nBatchesR = (size + _Nsimd - 1) / _Nsimd;
    if ((int)right5d.size() != _nBatchesR) {
      std::cerr << "A2ATaskSpinTasteStencil::setRight(5d): right5d.size()="
                << right5d.size() << " != nBatches=" << _nBatchesR
                << " for size=" << size << std::endl;
      GridAbort();
    }
    for (int b = 0; b < _nBatchesR; b++)
      if ((*right5d[b]).Grid() != _grid5d.get()) {
        std::cerr << "A2ATaskSpinTasteStencil::setRight(5d): right5d[" << b
                  << "] is not on this task's _grid5d" << std::endl;
        GridAbort();
      }

    // Release the previous view first (same hazard as the 4D form).
    _paddedRhsView = nullptr;

    // Reallocate only if the batch count changed (mirrors the 4D form).
    if ((int)_paddedRight5d.size() != _nBatchesR) {
      _paddedRight5d.clear();
      _paddedRight5d.reserve(_nBatchesR);
      for (int b = 0; b < _nBatchesR; b++)
        _paddedRight5d.emplace_back(_paddedGrid5d);
    }

    for (int b = 0; b < _nBatchesR; b++) {
      GRID_TRACE("A2AStencil/HaloExchange");
      _paddedRight5d[b] = _cell5d->Exchange(*right5d[b]); // no Zero/pack5d
    }
    _paddedRhsView = std::make_shared<A2AFieldView<vobj>>();
    _paddedRhsView->openViews(_paddedRight5d.data(), _nBatchesR);
  }

  //....................................................................
  // execute (ContractType dispatch). Full: one unfiltered vectorSum5d pass
  // + circulant scatter. Half/mixed (LeftHalf/RightHalf/BothHalf):
  // ParityBisect: two parity-filtered passes (even/odd source sites) into a
  // parity-split scratch, then assembleMat scatters the RAW (2L, R)
  // parity-split partials (rows [0,L)=M0, [L,2L)=M1); the sigma-combination
  // into the legacy interleaved (M, M-dagger) slots runs on the HOST in the
  // worker's finishMatOutput (raw-layout gsum + worker-internal expansion).
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

    cobj *shm_p = static_cast<cobj *>(
        acceleratorAllocDevice(gammaStride * nGamma * nParity * sizeof(cobj)));
    {
      GRID_TRACE("A2AStencil/ZeroScratch");
      accelerator_for(idx, gammaStride * nGamma * nParity, 1,
                      { shm_p[idx] = Zero(); });
    }

    {
      GRID_TRACE("A2AStencil/VectorSum5d");
      for (int p = 0; p < nParity; p++) {
        // g-major flat blocking: each block covers nG spec gammas x nP
        // momenta with nG*nP <= MF_SUM_ARRAY_MAX (the per-thread sum[]
        // bound), taking FULL p-ranges per block so the hoisted inner
        // product is shared across every momentum of its gammas. For
        // _nMom == 1 this degenerates to the previous 16-gamma blocks.
        int nPBlock = std::min(_nMom, MF_SUM_ARRAY_MAX);
        int nGBlockMax = std::max(1, MF_SUM_ARRAY_MAX / nPBlock);
        for (int g0 = 0; g0 < _nSpec; g0 += nGBlockMax) {
          int nG = std::min(_nSpec - g0, nGBlockMax);
          for (int p0 = 0; p0 < _nMom; p0 += nPBlock) {
            int nP = std::min(_nMom - p0, nPBlock);
            if (_momView)
              vectorSum5d<true>(shm_p + (size_t)p * nGamma * gammaStride, g0,
                                nG, p0, nP, localOrthogDimSize,
                                localSpatialVolume, gammaStride,
                                _siteParity.data(), cbSplit ? p : -1);
            else
              vectorSum5d<false>(shm_p + (size_t)p * nGamma * gammaStride, g0,
                                 nG, p0, nP, localOrthogDimSize,
                                 localSpatialVolume, gammaStride,
                                 _siteParity.data(), cbSplit ? p : -1);
          }
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
  // serial loop inside.
  //
  // CB output-placement: ALL CB contract modes write the RAW (2L, R)
  // parity-split partials (rows [0,L)=M0, [L,2L)=M1; the ParityBisect
  // escape hatch IS the CB layout again). The sigma-combination into the
  // legacy interleaved (M, M-dagger) slots is a HOST pass in the worker's
  // finishMatOutput, so the D2H copy and multi-rank global sum run over
  // the raw buffer (half the BothHalf interleaved volume) and the gsum
  // partition matches the pre-fusion build (bitwise at any rank count).
  //....................................................................
  void assembleMat(scalar_type *result_p, cobj *shm_p, ContractType ct,
                   int nGamma, int gammaStride, int localOrthogDimSize,
                   int pc, int Nt) {
    bool cbSplit = (ct != ContractType::Full);
    int matL = cbSplit ? 2 * _sizeL : _sizeL; // raw split is always (2L, R)
    int matR = _sizeR;
    int parityStride = nGamma * gammaStride; // even slice | odd slice

    // Copy members to locals: the GPU lambda cannot capture a CPU `this`.
    int sizeL = _sizeL;
    int sizeR = _sizeR;
    int nBatchesL = _nBatchesL;
    int nBatchesR = _nBatchesR;
    int Nsimd = _Nsimd;
    {
      GRID_TRACE("A2AStencil/Assemble");
      accelerator_for2d(l_batch, nBatchesL, r_batch, nBatchesR, static_cast<size_t>(Nsimd), {
#ifdef GRID_SIMT
        {
          int j = acceleratorSIMTlane(Nsimd);
#else
        for (int j = 0; j < Nsimd; j++) {
#endif
          int r = r_batch * Nsimd + j;
          if (r < sizeR) {
            for (int mu = 0; mu < nGamma; mu++) {
              for (int rt = 0; rt < localOrthogDimSize; rt++) {
                int gt = rt + pc * localOrthogDimSize; // T simd=1
                for (int i = 0; i < Nsimd; i++) {
                  int l = l_batch * Nsimd + i;
                  if (l < sizeL) {
                    int s = ((i - j) % Nsimd + Nsimd) % Nsimd;
                    int shm_idx =
                        rt +
                        localOrthogDimSize *
                            (s + Nsimd * (l_batch + nBatchesL * r_batch)) +
                        mu * gammaStride;
                    int64_t matBase = (int64_t)mu * matL * matR * Nt +
                                      (int64_t)matR * matL * gt;
                    auto M0 = TensorRemove(extractLane(j, shm_p[shm_idx]));
                    // Full (single store) / raw CB split: rows [0,L)=M0,
                    // [L,2L)=M1 (BothHalf/LeftHalf/RightHalf join
                    // ParityBisect; the worker expands into the legacy
                    // interleaved slots on the host).
                    int64_t mat_idx = matBase + (int64_t)r + matR * l;
                    result_p[mat_idx] = M0;
                    if (cbSplit) {
                      auto M1 = TensorRemove(
                          extractLane(j, shm_p[shm_idx + parityStride]));
                      result_p[mat_idx + (int64_t)matR * sizeL] = M1;
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
  // vectorSum5d: fused kernel, shared by the Full and CB half/mixed modes.
  // Ordinary coalesced (nsimd=Nsimd) launch. Loop order: (lt_batch =
  // [l_batch, rt], r_batch) parallel; (g, ep, so) serial, MOMENTUM
  // innermost at the accumulate site. The expensive W*psi*innerProduct is
  // computed once per (gamma, endpoint, site) and each momentum of the
  // block multiply-accumulates the SAME inner product into its own
  // accumulator slot sum[s][gb*nP+pb] (g-major flat mu = g*nMom + p) --
  // momentum projection costs one complex MAC + one phase read per extra
  // momentum, never a redone inner product.
  //
  // HasMom selects the kernel at compile time. HasMom == false (no
  // momentum, nP == 1) is the pre-momentum (143f74b) kernel body: no
  // momentum branch and no runtime-strided sum[] slot in the compiled
  // code. A single kernel holding both branches measured 57 ps per
  // (endpoint, site, pair) at zero momentum against 45 ps for 143f74b.
  //
  // The time-slice index rt is folded into the first parallel dimension so
  // each (l_batch, rt, r_batch) triple gets its own block (SM occupancy via
  // warp switching; each rt writes a distinct shm slot so no reduction is
  // needed). ep and g remain outside the so loop so that for fixed (ep, g)
  // the psi gather reads traverse the padded RHS buffer sequentially,
  // enabling hardware prefetching. wantParity: -1 (Full) accumulates every
  // source site; 0/1 (half/mixed) keep only even/odd-parity sites.
  //....................................................................
  template <bool HasMom>
  void vectorSum5d(cobj *shm_p, int g_offset, int nG, int p_offset, int nP,
                   int localOrthogDimSize, int localSpatialVolume,
                   int gammaStride, const int *siteParity_p, int wantParity) {
    GRID_TRACE("A2AStencil/vectorSum5d");
    int nBatchesL = _nBatchesL;
    int nBatchesR = _nBatchesR;
    int Nsimd = _Nsimd;
    int osites = _osites;
    int paddedOsites = _paddedOsites;
    int nMom = _nMom; // flat-axis momentum stride (g-major: mu = g*nMom+p)
    assert(HasMom == (_momView != nullptr) && (HasMom || nP == 1));

    // Scalar W store: bind a local Lattice& before autoView -- the
    // macro expands w.View(mode) verbatim, so a unique_ptr must be dereferenced
    // first (else *_wScalarGrid.View(mode) binds .View to the unique_ptr).
    Lattice<ColourMatrix> &wScalar = *_wScalarGrid;
    autoView(wsGrid_v, wScalar, AcceleratorRead);
    ColourMatrix *wsScalar_p = &wsGrid_v[0];   // [globalFwdEp][paddedOsites]
    FermView *lhsView_p = _lhsView->getView(); // [l_batch][ss]
    FermView *rhsView_p = _paddedRhsView->getView(); // [r_batch][paddedSS]
    ComplexView *phaseView_p = _phaseView->getView(); // [g][ss] per-spec phase
    ComplexView *momView_p =
        HasMom ? _momView->getView() : nullptr; // [p][ss]

    auto offsets_p = _allOffsets.data(); // [(epStart+ep)*osites + ss]
    auto nEp_p = _nEpDev.data();
    auto epStart_p = _epStartDev.data();
    auto forwardRep_p =
        _forwardRepDev.data(); // [globalEp] -> global fwd-field idx
    auto isForward_p = _isForwardDev.data();           // [globalEp] 0/1
    auto interiorOffset_p = _interiorOffsetDev.data(); // [ss]

    int nBatchesLT = nBatchesL * localOrthogDimSize;
    accelerator_for2d(lt_batch, nBatchesLT, r_batch, nBatchesR, static_cast<size_t>(Nsimd), {
      int l_batch = lt_batch / localOrthogDimSize;
      int rt = lt_batch % localOrthogDimSize;

      calcScalar sum[vobj::Nsimd()][MF_SUM_ARRAY_MAX]; // [s][gb*nP+pb]
      for (int s = 0; s < Nsimd; s++)
        for (int m = 0; m < (HasMom ? nG * nP : nG); m++)
          sum[s][m] = Zero();

      for (int gb = 0; gb < nG; gb++) {
        int g = g_offset + gb; // spec gamma (endpoint tables per spec)
        int nEp = nEp_p[g];
        int epStart = epStart_p[g];
        for (int ep = 0; ep < nEp; ep++) {
          int ge = epStart + ep;        // global endpoint index
          int fwdEp = forwardRep_p[ge]; // global forward-field index
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
            auto W = promoteColourMatrix(
                wsScalar_p[(size_t)fwdEp * paddedOsites + wSS]);
            if (!isFwd)
              W = adj(W);
            auto Wpsi = W * psi;
            // Per-spec phase (Follana / later external Amu) at the SOURCE
            // site x. W is unphased so the {s,-s} pairing hermiticity is
            // exact (corr == +1). Phase is ep-independent; applied per-ep
            // here (the ep-outside-so loop order keeps the RHS buffer
            // sequentially-accessible; mathematically phase * sum_ep IP ==
            // sum_ep phase*IP).
            calcScalar gamma_phase = coalescedRead(phaseView_p[g][ss]);
            // Plain if, not if constexpr: nvcc rejects an extended
            // __host__ __device__ lambda that first captures a variable
            // inside a constexpr-if branch. HasMom is a template constant,
            // so the dead branch folds away either way.
            if (HasMom) {
              // Phased inner product computed once per (gamma, endpoint,
              // site, s) into a per-lane scalar and shared by every
              // momentum of the block.
              for (int s = 0; s < Nsimd; s++) {
                calcScalar ip = gamma_phase * innerProduct(
                    coalescedReadRotate(lhsView_p[l_batch][ss], s), Wpsi);
                for (int pb = 0; pb < nP; pb++)
                  sum[s][gb * nP + pb] =
                      sum[s][gb * nP + pb] +
                      coalescedRead(momView_p[p_offset + pb][ss]) * ip;
              }
            } else {
              for (int s = 0; s < Nsimd; s++)
                sum[s][gb] = sum[s][gb] + gamma_phase * innerProduct(
                    coalescedReadRotate(lhsView_p[l_batch][ss], s), Wpsi);
            }
          }
        }
      }

      for (int s = 0; s < Nsimd; s++)
        for (int gb = 0; gb < nG; gb++)
          for (int pb = 0; pb < (HasMom ? nP : 1); pb++) {
            // g-major mu; size_t discipline (8badfad6): every product
            // feeding shm_idx is computed in size_t -- mu*gammaStride and
            // the batch-index product can exceed INT_MAX on large runs.
            int mu = (g_offset + gb) * nMom + (p_offset + pb); // g-major
            size_t shm_idx =
                (size_t)rt +
                (size_t)localOrthogDimSize *
                    ((size_t)s +
                     (size_t)Nsimd * ((size_t)l_batch +
                                      (size_t)nBatchesL * r_batch)) +
                (size_t)mu * gammaStride;
            coalescedWrite(shm_p[shm_idx], sum[s][HasMom ? gb * nP + pb : gb]);
          }
    });
  }
};

NAMESPACE_END(Grid);
