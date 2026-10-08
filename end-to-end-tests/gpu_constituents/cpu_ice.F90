module cpu_ice
  ! CPU scheme: updates a different constituent and its tendency.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: cpu_ice_run
contains
!> \section arg_table_cpu_ice_run Argument Table
!! \htmlinclude cpu_ice_run.html
  subroutine cpu_ice_run(ncol, nlev, temp, cld_ice, ice_tend, errmsg, errcode)
    integer,            intent(in)    :: ncol, nlev
    real(kind_phys),    intent(in)    :: temp(:,:)
    real(kind_phys),    intent(inout) :: cld_ice(:,:)
    real(kind_phys),    intent(out)   :: ice_tend(:,:)
    character(len=512), intent(out)   :: errmsg
    integer,            intent(out)   :: errcode
    errmsg = ''
    errcode = 0
    cld_ice(1:ncol,1:nlev)  = cld_ice(1:ncol,1:nlev) + 2.0e-7_kind_phys*(temp(1:ncol,1:nlev) - 200.0_kind_phys)
    ice_tend(1:ncol,1:nlev) = 3.0e-9_kind_phys
  end subroutine cpu_ice_run
end module cpu_ice
