module gpu_heat
  ! GPU-ported scheme: its kernel assumes every array argument is already
  ! on the device (default(present)).  The generated caps guarantee that.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: gpu_heat_run
contains
!> \section arg_table_gpu_heat_run Argument Table
!! \htmlinclude gpu_heat_run.html
  subroutine gpu_heat_run(ncol, nlev, temp, qv, ps, errmsg, errflg)
    integer,          intent(in)    :: ncol, nlev
    real(kind_phys),  intent(inout) :: temp(:,:)
    real(kind_phys),  intent(in)    :: qv(:,:)
    real(kind_phys),  intent(inout) :: ps(:)   ! BUG: metadata says in
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
        temp(i,k) = temp(i,k) + 500.0_kind_phys*qv(i,k) + 1.0e-5_kind_phys*ps(i)
        if (k == 1) ps(i) = 1.0001_kind_phys*ps(i)   ! BUG: ps is intent(in)
      end do
    end do
  end subroutine gpu_heat_run
end module gpu_heat
