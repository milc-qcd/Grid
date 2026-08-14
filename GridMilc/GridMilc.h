/*
 * GridMilc.h — umbrella header for the GridMilc add-on library
 * (https://github.com/paboyle/Grid)
 *
 * Include this for the full staggered spin-taste / all-to-all / MILC IO
 * surface:
 *
 *     #include <Grid/Grid.h>
 *     #include <GridMilc/GridMilc.h>
 *
 * GridMilc is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 2 (or, at your option,
 * any later version). See COPYING/LICENSE in the top-level distribution.
 */
#ifndef GRIDMILC_H
#define GRIDMILC_H

#include <GridMilc/util/Version.h>
#include <GridMilc/spin/StagGamma.h>
#include <GridMilc/a2a/A2AView.h>
#include <GridMilc/a2a/A2ATask.h>
#include <GridMilc/a2a/A2ATaskStencil.h>   // Phase 3 (L1.2-05/L1.2-01): relocated stencil task
#include <GridMilc/a2a/A2AWorker.h>
#include <GridMilc/a2a/A2AWorkerStencil.h>   // Phase 3 (L0-01): split stencil worker
#include <GridMilc/io/MilcIO.h>

#endif // GRIDMILC_H
