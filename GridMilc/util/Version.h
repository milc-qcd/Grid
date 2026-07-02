/*
 * GridMilc/util/Version.h — part of GridMilc (https://github.com/paboyle/Grid)
 *
 * GridMilc is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 2 (or, at your option,
 * any later version). See COPYING/LICENSE in the top-level distribution.
 */
#ifndef GRIDMILC_VERSION_H
#define GRIDMILC_VERSION_H

#include <Grid/Namespace.h>

NAMESPACE_BEGIN(Grid)

// GridMilc library version string. Compiled into libGridMilc.a so that the
// otherwise header-only add-on library ships a real, installable archive
// (mirrors Grid/util/version.cc).
const char * gridMilcVersion(void);

NAMESPACE_END(Grid)

#endif // GRIDMILC_VERSION_H
