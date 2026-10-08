module gpu_finish
  ! GPU-ported scheme in the second group: reads temp after the host
  ! changed it between groups, and precl written by the first group.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: gpu_finish_run
contains
!> \section arg_table_gpu_finish_run Argument Table
!! \htmlinclude gpu_finish_run.html
  subroutine gpu_finish_run(ncol, nlev, temp, precl, qv, errmsg, errflg)
    integer,          intent(in)    :: ncol, nlev
    real(kind_phys),  intent(inout) :: temp(:,:)
    real(kind_phys),  intent(in)    :: precl(:)
    real(kind_phys),  intent(in)    :: qv(:,:)
    character(len=*), intent(out)   :: errmsg
    integer,          intent(out)   :: errflg
    integer :: i, k
    errmsg = ''
    errflg = 0
#ifdef CCPP_TEST_OMP
    !$omp target teams loop collapse(2)
#else
    !$acc parallel loop collapse(2) default(present)
#endif
    do k = 1, nlev
      do i = 1, ncol
        temp(i,k) = temp(i,k) + 100.0_kind_phys*precl(i) - 10.0_kind_phys*qv(i,k)
      end do
    end do
  end subroutine gpu_finish_run
end module gpu_finish
