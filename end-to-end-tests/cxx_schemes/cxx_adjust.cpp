// C++ implementation of the CPU scheme cpu_adjust (same arithmetic), called
// from the CCPP caps through the BIND(C) interface capgen generates for
// language = c++.  Prototype: cxx_adjust_ccpp.h (generated next to the caps).
#include "cxx_adjust_ccpp.h"

#include <cstring>

extern "C" void cxx_adjust_run(int ncol, int nlev, double* temp, double* work,
                               char* errmsg, int* errflg) {
  // Column-major: element (i, k) is at i + ncol*k.
  for (int k = 0; k < nlev; ++k) {
    for (int i = 0; i < ncol; ++i) {
      const int ik = i + ncol * k;
      work[ik] = 0.01 * temp[ik];
      temp[ik] = temp[ik] - 0.5 * work[ik];
    }
  }
  std::memset(errmsg, ' ', 512);   // blank-padded, not NUL-terminated
  *errflg = 0;
}
