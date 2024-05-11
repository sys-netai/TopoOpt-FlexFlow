/**
 * \file realm_defines.h
 * Public-facing definitions of variables configured at build time
 * Keep the syntax checker happy for editor
 */

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++. Keep any C++-isms in
// legion_types.h, or elsewhere.
//
// ******************** IMPORTANT **************************

#define REALM_VERSION "legion-21.03.0-3000-g1c30cf196"

/* #undef DEBUG_REALM */

#define REALM_LIMIT_SYMBOL_VISIBILITY

#define COMPILE_TIME_MIN_LEVEL LEVEL_DEBUG

#define REALM_MAX_DIM 4

/* #undef REALM_USE_OPENMP */
/* #undef REALM_OPENMP_GOMP_SUPPORT */
/* #undef REALM_OPENMP_KMP_SUPPORT */

#define REALM_USE_PYTHON
#define REALM_PYTHON_VERSION_MAJOR 3
#define REALM_PYTHON_LIB "/home/frankwwy/anaconda3/lib/libpython3.8.so"

#define REALM_USE_CUDA
#define REALM_USE_CUDART_HIJACK

/* #undef REALM_USE_HIP */
/* #undef REALM_USE_HIP_HIJACK */

/* #undef REALM_USE_KOKKOS */

/* #undef REALM_USE_GASNET1 */
#define REALM_USE_GASNETEX

/* technically these are defined by per-conduit GASNet include files,
 * but we do it here as well for the benefit of applications that care
 */
#define GASNET_CONDUIT_MPI 1
/* #undef GASNET_CONDUIT_IBV */
/* #undef GASNET_CONDUIT_UDP */
/* #undef GASNET_CONDUIT_ARIES */
/* #undef GASNET_CONDUIT_GEMINI */
/* #undef GASNET_CONDUIT_PSM */
/* #undef GASNET_CONDUIT_UCX */

/* #undef REALM_USE_MPI */

/* #undef REALM_USE_LLVM */
/* #undef REALM_LLVM_VERSION */
/* #undef REALM_ALLOW_MISSING_LLVM_LIBS */

/* #undef REALM_USE_HDF5 */

#define REALM_USE_LIBDL
/* #undef REALM_USE_DLMOPEN */

/* #undef REALM_USE_HWLOC */

/* #undef REALM_USE_PAPI */

/* #undef REALM_RESPONSIVE_TIMELIMIT */
