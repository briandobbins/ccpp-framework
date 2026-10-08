// A made-up GPU-ported C++ "microphysics" scheme for testing the C++ path
// with device-side data.  Its arrays arrive already present on the device
// (the CCPP caps create and update the device copies), so the kernels use
// present() (OpenACC) or a tofrom map that finds the data already mapped
// (OpenMP).  Built for the CPU without -acc / -mp, the pragmas are ignored.
//
// Only correctly rounded operations (+ - * / sqrt) and no FMA, so the GPU
// result must match the CPU result bit for bit.
#include "cxx_micro_ccpp.h"

#include <cmath>
#include <cstring>

extern "C" void cxx_micro_run(int ncol, int nlev, double* temp, double* qv,
                              const double* ps, double* precl,
                              char* errmsg, int* errflg) {
  const int n = ncol * nlev;
  // Column-major: element (i, k) is at i + ncol*k.
#ifdef CCPP_TEST_OMP
  #pragma omp target teams loop map(tofrom: temp[0:n], qv[0:n], precl[0:ncol]) map(to: ps[0:ncol])
#else
  #pragma acc parallel loop gang vector present(temp[0:n], qv[0:n], ps[0:ncol], precl[0:ncol])
#endif
  for (int i = 0; i < ncol; ++i) {
    const double p_hpa = 0.01 * ps[i];         // surface pressure in hPa
    double carry = 0.0;                         // condensate from the level above
    double rain = 0.0;
#ifndef CCPP_TEST_OMP
    #pragma acc loop seq
#endif
    for (int k = 0; k < nlev; ++k) {
      const int ik = i + ncol * k;
      double t = temp[ik];
      double q = qv[ik];
      // Saturation vapour pressure: a quadratic fit around 0 C (hPa).
      const double tc = t - 273.15;
      double es = 6.112 + 0.4 * tc + 0.0145 * tc * tc;
      if (es < 0.01) es = 0.01;
      // Level pressure falls off linearly with height.
      const double p = p_hpa * (1.0 - 0.1 * k);
      // Saturation mixing ratio, scaled so that some levels condense.
      const double qs = 0.3 * 0.622 * es / (p - es);
      const double excess = q - qs;
      double cond = (excess > 0.0) ? 0.25 * excess : 0.0;
      cond = cond + 0.1 * carry;                // vertical recurrence
      if (cond > 0.9 * q) cond = 0.9 * q;       // never remove all vapour
      q = q - cond;
      t = t + 2500.0 * cond;                    // latent heating (scaled)
      carry = cond;
      rain = rain + cond * std::sqrt(1.0 + 0.001 * t);
      temp[ik] = t;
      qv[ik] = q;
    }
    precl[i] = precl[i] + 0.001 * rain;
  }
  std::memset(errmsg, ' ', 512);
  *errflg = 0;
}
