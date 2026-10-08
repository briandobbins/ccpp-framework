// Host-side C++ scheme taking every argument type added for C++ schemes:
//   use_extra  logical scalar (bool, converted from Fortran logical by the cap)
//   extra      optional array (NULL when the host's active condition is false)
//   state      derived type gpu_state_type, through its generated descriptor
//   mask       logical array, written here as bool
//   work       an array this scheme allocates (allocation handle)
// ftn_types_check recomputes every result and fails the run on a mismatch.
#include "cxx_types_ccpp.h"

#include <cstring>

static void fail(char* errmsg, int* errflg, const char* msg) {
  std::memset(errmsg, ' ', 512);
  std::memcpy(errmsg, msg, std::strlen(msg));
  *errflg = 1;
}

extern "C" void cxx_types_run(int ncol, int nlev, int col_start, bool use_extra,
                              const double* extra, ccpp_gpu_state_type_t* state,
                              double* temp, bool* mask, ccpp_alloc_double_1_t* work,
                              char* errmsg, int* errflg) {
  std::memset(errmsg, ' ', 512);
  *errflg = 0;
  // The optional argument is present exactly when the host says it is active.
  if ((extra != nullptr) != use_extra) {
    fail(errmsg, errflg, "cxx_types: extra present/absent disagrees with use_extra");
    return;
  }
  // The descriptor carries whole host arrays: shape and Fortran lower bound.
  if (state->ps_shape[0] != state->precl_shape[0] || state->ps_lbound[0] != 1 ||
      state->ps == nullptr || col_start < 1 ||
      col_start - 1 + ncol > state->ps_shape[0]) {
    fail(errmsg, errflg, "cxx_types: unexpected derived-type descriptor");
    return;
  }
  // work is unallocated on entry (intent(out)); allocate it as work(1:ncol).
  if (work->data != nullptr) {
    fail(errmsg, errflg, "cxx_types: work already allocated on entry");
    return;
  }
  const int lower[1] = {1}, upper[1] = {ncol};
  double* w = work->allocate(work->handle, lower, upper);
  if (w == nullptr) {
    fail(errmsg, errflg, "cxx_types: allocation failed");
    return;
  }
  for (int i = 0; i < ncol; ++i) {
    const int g = col_start - 1 + i;          // column in the whole host array
    w[i] = state->ps[g] * 1.0e-5 + (use_extra ? 1.0 : 0.0);
    state->precl[g] = state->ps[g] * 1.0e-12;  // written through the descriptor
    for (int k = 0; k < nlev; ++k) {
      const int n = i + ncol * k;              // column-major chunk index
      temp[n] += use_extra ? 0.01 * extra[n] : 0.002;
      mask[n] = temp[n] < 260.0;
    }
  }
}
