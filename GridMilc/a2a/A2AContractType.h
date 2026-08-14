/******************************************************************************/
/* A2AContractType.h -- shared checkerboard-state vocabulary for the a2a      */
/* meson-field workers/tasks. ContractType was lifted out of A2ATaskBase so   */
/* both the stencil (production) and legacy paths consume it with no          */
/* inheritance coupling.                                                      */
/* Part of GridMilc (https://github.com/paboyle/Grid).                       */
/******************************************************************************/
#pragma once
// GridCore (not GridQCDcore): ContractType names no QCD types. GridCore.h
// transitively provides the serialisation layer (Serializable + the
// GRID_SERIALIZABLE_ENUM macro) the enum is built on. Matches the
// A2AView.h / StencilGather5d.h include style.
#include <Grid/GridCore.h>

NAMESPACE_BEGIN(Grid);

// ContractType encodes the checkerboard/contraction mode of a meson-field
// contraction as a bit field: bit 0 = RightHalf, bit 1 = LeftHalf. BothHalf ==
// LeftHalf | RightHalf. The values and bit semantics are identical to the
// previous nested A2ATaskBase::ContractType; only the declaration site moved
// (now free at Grid-namespace scope, modeled on Grid/qcd/QCD.h::Current).
GRID_SERIALIZABLE_ENUM(ContractType, undef,
                       Full,      0,
                       RightHalf, 1,
                       LeftHalf,  2,
                       BothHalf,  3);

NAMESPACE_END(Grid);
