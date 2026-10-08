module cpu_adjust
  ! CPU-only scheme sharing temp with a GPU scheme in the same subcycle.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: cpu_adjust_run
contains
!> \section arg_table_cpu_adjust_run Argument Table
!! \htmlinclude cpu_adjust_run.html
  subroutine cpu_adjust_run(ncol, nlev, temp, work, errmsg, errflg)
    integer,          intent(in)    :: ncol, nlev
    real(kind_phys),  intent(inout) :: temp(:,:)
    real(kind_phys),  intent(out)   :: work(:,:)
    character(len=*), intent(out)   :: errmsg
    integer,          intent(out)   :: errflg
    errmsg = ''
    errflg = 0
    work(1:ncol,1:nlev) = 0.01_kind_phys*temp(1:ncol,1:nlev)
    temp(1:ncol,1:nlev) = temp(1:ncol,1:nlev) - 0.5_kind_phys*work(1:ncol,1:nlev)
  end subroutine cpu_adjust_run
end module cpu_adjust
