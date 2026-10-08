module gpu_liq
  ! GPU scheme: updates ONE constituent and writes ITS tendency, both slices
  ! of the shared constituent arrays.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: gpu_liq_run
contains
!> \section arg_table_gpu_liq_run Argument Table
!! \htmlinclude gpu_liq_run.html
  subroutine gpu_liq_run(ncol, nlev, temp, cld_liq, liq_tend, errmsg, errcode)
    integer,            intent(in)    :: ncol, nlev
    real(kind_phys),    intent(in)    :: temp(:,:)
    real(kind_phys),    intent(inout) :: cld_liq(:,:)
    real(kind_phys),    intent(out)   :: liq_tend(:,:)
    character(len=512), intent(out)   :: errmsg
    integer,            intent(out)   :: errcode
    integer :: i, k
    errmsg = ''
    errcode = 0
#ifdef CCPP_TEST_OMP
    !$omp target teams loop collapse(2)
#else
    !$acc parallel loop collapse(2) default(present)
#endif
    do k = 1, nlev
      do i = 1, ncol
        cld_liq(i,k)  = cld_liq(i,k) + 1.0e-7_kind_phys*(temp(i,k) - 200.0_kind_phys)
        liq_tend(i,k) = 1.0e-9_kind_phys*real(k, kind_phys)
      end do
    end do
  end subroutine gpu_liq_run
end module gpu_liq
