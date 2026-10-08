// Device-side C++ scheme (memory_space = device) with the argument types
// added for GPU-side C++ schemes:
//   use_extra  logical scalar in    (converted on the host)
//   extra      optional array in    (NULL when absent; device data otherwise)
//   warm       logical array out    (a 0/1 device scratch array the cap
//                                    converts back to logical on the device)
//   any_warm   logical scalar out   (written on the host after the kernel)
//   state      derived type in      (descriptor; its array members are present
//                                    on the device, including precl, which
//                                    cxx_types just wrote on the host)
//   gwork      allocatable out      (allocated through the handle: the
//                                    callback also creates the device copy)
// warm = temp + 0.001*extra > 262 (temp + 0.0005 when extra is absent).
// Only + and * and comparisons, no FMA: GPU results match the CPU build.
#include "cxx_types_gpu_ccpp.h"

#include <cstring>

extern "C" void cxx_types_gpu_run(int ncol, int nlev, int col_start, bool use_extra,
                                  const double* extra, const double* temp, bool* warm,
                                  bool* any_warm, const ccpp_gpu_state_type_t* state,
                                  ccpp_alloc_double_1_t* gwork,
                                  char* errmsg, int* errflg) {
  std::memset(errmsg, ' ', 512);
  *errflg = 0;
  if ((extra != nullptr) != use_extra) {
    const char msg[] = "cxx_types_gpu: extra present/absent disagrees with use_extra";
    std::memcpy(errmsg, msg, sizeof(msg) - 1);
    *errflg = 1;
    return;
  }
  const int n = ncol * nlev;
  int nwarm = 0;
  if (extra != nullptr) {
#ifdef CCPP_TEST_OMP
    #pragma omp target teams loop map(tofrom: warm[0:n]) map(to: temp[0:n], extra[0:n]) reduction(+:nwarm)
#else
    #pragma acc parallel loop present(temp[0:n], warm[0:n], extra[0:n]) reduction(+:nwarm)
#endif
    for (int m = 0; m < n; ++m) {
      warm[m] = temp[m] + 0.001 * extra[m] > 262.0;
      nwarm += warm[m] ? 1 : 0;
    }
  } else {
#ifdef CCPP_TEST_OMP
    #pragma omp target teams loop map(tofrom: warm[0:n]) map(to: temp[0:n]) reduction(+:nwarm)
#else
    #pragma acc parallel loop present(temp[0:n], warm[0:n]) reduction(+:nwarm)
#endif
    for (int m = 0; m < n; ++m) {
      warm[m] = temp[m] + 0.0005 > 262.0;
      nwarm += warm[m] ? 1 : 0;
    }
  }
  *any_warm = nwarm > 0;

  // gwork(i) = ps*1e-5 + precl*1e6 on the device, from the descriptor's
  // members (whole host arrays; this chunk starts at column col_start).
  const int lower[1] = {1}, upper[1] = {ncol};
  double* gw = gwork->allocate(gwork->handle, lower, upper);
  const double* ps = state->ps;
  const double* precl = state->precl;
  const int ntot = state->ps_shape[0];
  const int off = col_start - state->ps_lbound[0];
#ifdef CCPP_TEST_OMP
  #pragma omp target teams loop map(to: ps[0:ntot], precl[0:ntot]) map(tofrom: gw[0:ncol])
#else
  #pragma acc parallel loop present(ps[0:ntot], precl[0:ntot], gw[0:ncol])
#endif
  for (int i = 0; i < ncol; ++i) {
    gw[i] = ps[off + i] * 1.0e-5 + precl[off + i] * 1.0e6;
  }
}
