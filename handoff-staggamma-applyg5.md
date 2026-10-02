# Handoff (Grid / GridMilc): make `applyG5` mean ε∘Γ exactly

Date: 2026-09-23
Status: planned, not started
Branch at time of writing: `feature/opt-a2a-general-spin-taste`
Downstream: HadronsMILC (`HadronsMILC/handoff-staggamma-applyg5.md`) and grid-lma (`grid-lma/handoff-staggamma-applyg5.md`). Both depend on the API added here, so do this repo first.

## Goal

Make `applyG5=true` produce the operator ε∘Γ, where ε(x) = (−1)^{x+y+z+t}. "ε∘Γ" means: apply Γ, then multiply by ε. The sign must hold for every spin-taste pair, and it must survive being passed through the pipeline.

This is the convention MILC uses. `GridMilc/tests/Test_staggamma.cc` validates against MILC `ks_spectrum` with `src = eps * applyGamma(g, e_k)` and `snk = eps * applyGamma(g, phi_0)`. The HadronsMILC GaugeProp → Meson pipeline has the same structure, so if `applyG5` means exactly ε∘Γ it reproduces MILC.

## The bug

`StagGamma::ParseSpinTasteString(str, applyG5=true)` (`spin/StagGamma.h:105-121`) does two things wrong:

1. It forms `st * g5` with `operator*`, then keeps only `_spin` and `_taste` (`:113-116`), so `_negated` is thrown away.
2. `operator*`'s Follana A4 sign (`:537-559`) is not consistent with operator composition in any case. For `(G5Y G5)`, the product with its sign equals −Γ∘ε.

A free-field replica of the sign logic is in `handoff-staggamma-applyg5-check.py`, next to this file. It builds each operator as a matrix on a 4⁴ lattice with U = 1. It finds:

- **Local operators:** all 16 local pairs come out as +ε∘Γ.
- **One-link operators:** 32 of the 64 come out as **−ε∘Γ**. These are exactly the ones whose shift is in Y or T:
  `(G1 GY) (G1 GT) (GZ GYZ) (GZ GZT) (GY G1) (GY GYT) (GYZ GZ) (GYZ G5X) (GX GXY) (GX GXT) (GZX G5T) (GZX G5Y) (GXY GX) (GXY G5Z) (G5T GZX) (G5T G5) (GT G1) (GT GYT) (GZT GZ) (GZT G5X) (GYT GY) (GYT GT) (G5X GYZ) (G5X GZT) (GXT GX) (GXT G5Z) (G5Y GZX) (G5Y G5) (G5Z GXY) (G5Z GXT) (G5 G5T) (G5 G5Y)`
- **Why local operators are fine:** LessThan(G5) = 0101 marks the Z and X bits. That is why the Y and T shifts are the ones that flip.

### The identity the fix relies on (checked for all 80 local and one-link pairs)

Let P = (spin⊕G5, taste⊕G5). Then:

- P has the same shift and `_scaling` as Γ.
- osc(P) = osc(Γ) ⊕ 1111, which is exactly ε's phase. Algebraically: LessThan(G5) ⊕ GreaterThan(G5) = 0101 ⊕ 1010.

Hence:

> ε∘O_Γ = O_P **with `_negated` = neg(Γ)**, where neg(·) is `calculateNegation()` evaluated on the pair.

ε is a pure phase applied after the shift chain, so the same identity holds for multi-link operators (popcount 2–4). The current code uses `_negated` = neg(P), which is wrong whenever neg(Γ) ≠ neg(P).

## Changes

### 1. StagGamma API (`spin/StagGamma.h`)

- **Add `setSpinTaste(SpinTastePair g, bool applyG5)`, or equivalently `applyG5Left()`.**
  1. Run `calculatePhase()` for the raw Γ and save `_negated`.
  2. Set `_spin ^= G5` and `_taste ^= G5`, then run `calculatePhase()` again (this sets osc and `_scaling`).
  3. Restore the saved `_negated`.

  The result is exactly ε∘Γ. It does not use `operator*`.
- **Keep the raw label on the object**, e.g. a `SpinTastePair _label` member plus `getLabel()` or `getLabelName()`. After applyG5, `_spin` and `_taste` hold P, but downstream output names use the raw pair.
- **Document the invariant:** once an object carries applyG5 (or any `_negated` not derived from its own pair), calling `setSpinTaste`, `setSpin` or `setTaste` on it throws that sign away. Shared objects must be used through `const&`. `applyGamma` and `operator()` are already `const`.
- **Change `ParseSpinTasteString(str, applyG5)`.**
  - Stop using `operator*`.
  - If the `applyG5=true` form is kept, have it return the P pairs for naming only, with a comment that the pairs carry no sign.
  - Better: add a helper that returns `std::vector<StagGamma>` built with `setSpinTaste(g, applyG5)`, for example `MakeSpinTasteOps(str, applyG5, U*)`.
- **Fix the dead constructor `StagGamma(SpinTastePair initg)` (`:82`).** It constructs a temporary and leaves `*this` uninitialized. No caller uses it today, so either fix it (`: StagGamma(initg.first, initg.second) {}`) or delete it.
- **`operator*`.** Its only in-tree caller is `ParseSpinTasteString`. Either leave it with a comment saying its sign is not composition-consistent (see `(G5Y G5)` above), or fix it and cover it with the test in change 3. Nothing in the applyG5 path should depend on it.
- **Access to `applyCoeffsAndPhase`.** The A2A stencil already calls `spinTaste.applyCoeffsAndPhase(...)` on its own objects (`a2a/A2ATaskStencil.h:413`). No `_negated` accessor is needed if callers use the object directly.

### 2. A2A kernels: take `std::vector<StagGamma>`, not pairs

Every A2A path currently rebuilds the operator from a bare pair with `spinTaste.setSpinTaste(gammas[g])`, which drops `_negated`. These are the sites:

| File | Lines | Role |
|---|---|---|
| `a2a/A2ATaskStencil.h` | 258 (member), 318 (ctor), 341, 412-413, 428, 475 | endpoint offsets, **phase fields**, pairing maps |
| `a2a/A2AWorkerStencil.h` | 73 | `setWorkerStencil` signature |
| `a2a/A2AWorker.h` | 85, 123, 167, 181, 185 | worker setup / popcount checks |
| `a2a/A2ATask.h` | 497, 503, 510, 527, 715, 725, 735, 764, 773, 790, 1169, 1176, 1194, 1198, 1300 | local, one-link and spin-taste tasks, including the applyGamma pre-transform |

Change these signatures to take `const std::vector<StagGamma> &`, and use the objects directly:

- The phase fields become `gammas[g].applyCoeffsAndPhase(_phaseFields[g], ones5d)`.
- The pre-transform becomes `gammas[mu](out, v_j)`.
- Popcount, shift and classification logic can keep reading `_spin` and `_taste` (the P pair): the shift and scaling are the same for Γ and ε∘Γ.

Also update `benchmarks/Benchmark_a2a_spin_taste.cc` and `tests/Test_a2a_*.cc` to the new signatures.

### 3. Regression test (`tests/Test_staggamma.cc`)

Add a mode, or a separate `Test_staggamma_applyg5`, that fails the run on any mismatch. For every one of the 256 (spin, taste) pairs, on a unit gauge and on the 4⁴ `lat.sample` configuration with APBC:

- `StagGamma g; g.setSpinTaste(pair, true); g(out1, f);`
- `out2 = eps * applyGamma(pair, f)`, using the existing `applyEpsilon` helper (`:173-179`).
- Assert `norm2(out1 - out2) == 0` to rounding.

Also check the A2A path: a meson field built from applyG5 objects must equal the one built by explicit ε composition. For the all-to-all path, use `Test_a2a_stencil` against `A2ATask` with the `applyGamma` pre-transform.

## Effect on downstream results

- **Unchanged:** local operators, and correlators whose one-link vertices come in pairs with the same shift direction (such as the axial one-link correlator with the same μ at source and sink), because the sign appears twice.
- **Sign flips:** correlators with an odd number of Y or T one-link vertices, and A2A meson fields for Y and T one-link gammas. Meson fields already on disk carry the old sign.

## Out of scope here

- Consumer changes: see the HadronsMILC and grid-lma handoffs.
- grid-lma keeps a stale copy of StagGamma at `grid-lma/src/cpp/StagGamma.h` (the old `ParseSpinTaste` name and no multi-link support). It is handled in the grid-lma handoff.
