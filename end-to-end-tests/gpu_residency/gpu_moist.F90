module gpu_moist
  ! GPU-ported scheme with an optional argument and a column reduction.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: gpu_moist_run
contains
!> \section arg_table_gpu_moist_run Argument Table
!! \htmlinclude gpu_moist_run.html
  subroutine gpu_moist_run(ncol, nlev, qv, work, precl, extra, errmsg, errflg)
    integer,                   intent(in)    :: ncol, nlev
    real(kind_phys),           intent(inout) :: qv(:,:)
    real(kind_phys),           intent(in)    :: work(:,:)
    real(kind_phys),           intent(out)   :: precl(:)
    real(kind_phys), optional, intent(in)    :: extra(:,:)
    character(len=*),          intent(out)   :: errmsg
    integer,                   intent(out)   :: errflg
    integer :: i, k
    real(kind_phys) :: acc
    errmsg = ''
    errflg = 0
#ifdef CCPP_TEST_OMP
    !$omp target teams loop collapse(2)
#else
    !$acc parallel loop collapse(2) default(present)
#endif
    do k = 1, nlev
      do i = 1, ncol
        qv(i,k) = 0.99_kind_phys*qv(i,k) + 1.0e-6_kind_phys*work(i,k)
      end do
    end do
    if (present(extra)) then
#ifdef CCPP_TEST_OMP
      !$omp target teams loop collapse(2)
#else
      !$acc parallel loop collapse(2) default(present)
#endif
      do k = 1, nlev
        do i = 1, ncol
          qv(i,k) = qv(i,k) + 1.0e-7_kind_phys*extra(i,k)
        end do
      end do
    end if
#ifdef CCPP_TEST_OMP
    !$omp target teams loop private(acc)
#else
    !$acc parallel loop gang vector default(present) private(acc)
#endif
    do i = 1, ncol
      acc = 0.0_kind_phys
#ifndef CCPP_TEST_OMP
      !$acc loop seq
#endif
      do k = 1, nlev
        acc = acc + qv(i,k)
      end do
      precl(i) = 1.0e-3_kind_phys*acc
    end do
  end subroutine gpu_moist_run
end module gpu_moist
