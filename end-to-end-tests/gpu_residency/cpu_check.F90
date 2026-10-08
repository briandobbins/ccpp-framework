module cpu_check
  ! CPU-only scheme that reads qv, which the host never touches
  ! (host_access = none), so its valid copy normally lives on the device.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: cpu_check_run
contains
!> \section arg_table_cpu_check_run Argument Table
!! \htmlinclude cpu_check_run.html
  subroutine cpu_check_run(ncol, nlev, qv, temp, errmsg, errflg)
    integer,          intent(in)  :: ncol, nlev
    real(kind_phys),  intent(in)  :: qv(:,:)
    real(kind_phys),  intent(in)  :: temp(:,:)
    character(len=*), intent(out) :: errmsg
    integer,          intent(out) :: errflg
    errmsg = ''
    errflg = 0
    if (any(qv(1:ncol,1:nlev) <= 0.0_kind_phys) .or. &
        any(temp(1:ncol,1:nlev) < 100.0_kind_phys)) then
      errmsg = 'cpu_check: unphysical qv or temp (stale data?)'
      errflg = 1
    end if
  end subroutine cpu_check_run
end module cpu_check
