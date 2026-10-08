! The shared driver also runs a group "after"; nothing to do there.
module ftn_types_after

  implicit none
  private
  public :: ftn_types_after_run

contains

  subroutine ftn_types_after_run(ncol, errmsg, errflg)
    integer,            intent(in)  :: ncol
    character(len=512), intent(out) :: errmsg
    integer,            intent(out) :: errflg
    errmsg = ''
    errflg = 0
  end subroutine ftn_types_after_run

end module ftn_types_after
