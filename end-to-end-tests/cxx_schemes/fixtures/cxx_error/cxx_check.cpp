// C++ implementation of the CPU scheme cpu_check: reads qv and temp on the
// host and reports an error through errflg/errmsg if either is unphysical.
#include "cxx_check_ccpp.h"

#include <cstring>

extern "C" void cxx_check_run(int ncol, int nlev, const double* qv,
                              const double* temp, char* errmsg, int* errflg) {
  std::memset(errmsg, ' ', 512);
  *errflg = 0;
  for (int n = 0; n < ncol * nlev; ++n) {
    if (true) {  // FIXTURE: always report an error
      const char msg[] = "cxx_check: error raised in C++ reached the Fortran host";
      std::memcpy(errmsg, msg, sizeof(msg) - 1);
      *errflg = 1;
      return;
    }
  }
}
