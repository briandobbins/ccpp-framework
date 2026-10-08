// C++ implementation of the CPU scheme cpu_check: reads qv and temp on the
// host and reports an error through errflg/errmsg if either is unphysical.
#include "cxx_check_ccpp.h"

#include <cstring>

extern "C" void cxx_check_run(int ncol, int nlev, const double* qv,
                              const double* temp, char* errmsg, int* errflg) {
  std::memset(errmsg, ' ', 512);
  *errflg = 0;
  for (int n = 0; n < ncol * nlev; ++n) {
    if (qv[n] <= 0.0 || temp[n] < 100.0) {
      const char msg[] = "cxx_check: unphysical qv or temp (stale data?)";
      std::memcpy(errmsg, msg, sizeof(msg) - 1);
      *errflg = 1;
      return;
    }
  }
}
