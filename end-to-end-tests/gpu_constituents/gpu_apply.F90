module gpu_apply
  ! GPU scheme: applies every constituent's tendency at once, through the
  ! whole constituent arrays (overlapping the single-constituent slices).
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: gpu_apply_run
contains
!> \section arg_table_gpu_apply_run Argument Table
!! \htmlinclude gpu_apply_run.html
  subroutine gpu_apply_run(dt, const_tend, const, errmsg, errcode)
    real(kind_phys),    intent(in)    :: dt
    real(kind_phys),    intent(inout) :: const_tend(:,:,:)
    real(kind_phys),    intent(inout) :: const(:,:,:)
    character(len=512), intent(out)   :: errmsg
    integer,            intent(out)   :: errcode
    integer :: i, k, m
    errmsg = ''
    errcode = 0
#ifdef CCPP_TEST_OMP
    !$omp target teams loop collapse(3)
#else
    !$acc parallel loop collapse(3) default(present)
#endif
    do m = 1, size(const, 3)
      do k = 1, size(const, 2)
        do i = 1, size(const, 1)
          const(i,k,m) = const(i,k,m) + dt*const_tend(i,k,m)
          const_tend(i,k,m) = 0.0_kind_phys
        end do
      end do
    end do
  end subroutine gpu_apply_run
end module gpu_apply
