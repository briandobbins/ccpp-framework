module cpu_check_const
  ! CPU scheme in a second group: reads every constituent on the host.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: cpu_check_const_run
contains
!> \section arg_table_cpu_check_const_run Argument Table
!! \htmlinclude cpu_check_const_run.html
  subroutine cpu_check_const_run(const, errmsg, errcode)
    real(kind_phys),    intent(in)  :: const(:,:,:)
    character(len=512), intent(out) :: errmsg
    integer,            intent(out) :: errcode
    errmsg = ''
    errcode = 0
    if (any(const < 0.0_kind_phys)) then
      errmsg = 'cpu_check_const: negative constituent (stale data?)'
      errcode = 1
    end if
  end subroutine cpu_check_const_run
end module cpu_check_const
