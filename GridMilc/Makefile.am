#
# GridMilc: MILC-style staggered / all-to-all / IO add-on library for Grid.
# (https://github.com/paboyle/Grid)
#
# Static archive built as a sibling of Grid/. It depends on libGrid.a for
# its symbols (resolved when a final executable links both archives); only
# Grid's headers and build order are needed here (see HMC/Make.inc for the
# in-tree sibling pattern). Compile/link flags come from the globally-
# substituted AM_CXXFLAGS (-I${abs_srcdir}, configure.ac:852) and AM_LDFLAGS
# (-L${cwd}/Grid, configure.ac:854). These AC_SUBST'd values reach every
# subdir Makefile, letting GridMilc find Grid's headers in-tree (alongside
# Grid's own AC_CONFIG_HEADERS default-include handling for Grid/Config.h).
#
SUBDIRS = . tests benchmarks

lib_LIBRARIES = libGridMilc.a

include Make.inc

libGridMilc_a_SOURCES              = $(CCFILES)
libGridMilc_adir                   = $(includedir)/GridMilc
nobase_dist_libGridMilc_a_HEADERS  = $(HFILES)
