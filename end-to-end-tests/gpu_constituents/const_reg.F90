module const_reg
  ! Registers the three test constituents (host code, register phase).
  use ccpp_kinds, only: kind_phys
  use ccpp_constituent_prop_mod, only: ccpp_constituent_properties_t
  implicit none
  private
  public :: const_reg_register
contains
!> \section arg_table_const_reg_register Argument Table
!! \htmlinclude const_reg_register.html
  subroutine const_reg_register(dyn_const, errmsg, errcode)
    type(ccpp_constituent_properties_t), allocatable, intent(out) :: dyn_const(:)
    character(len=512), intent(out) :: errmsg
    integer,            intent(out) :: errcode
    errmsg = ''
    errcode = 0
    allocate(dyn_const(3))
    call dyn_const(1)%instantiate(std_name='water_vapor_mixing_ratio_wrt_moist_air', &
         long_name='water vapour', diag_name='Q', units='kg kg-1', &
         default_value=0._kind_phys, vertical_dim='vertical_layer_dimension', &
         advected=.true., errcode=errcode, errmsg=errmsg)
    if (errcode /= 0) return
    call dyn_const(2)%instantiate(std_name='cloud_liquid_dry_mixing_ratio', &
         long_name='cloud liquid', diag_name='CLDLIQ', units='kg kg-1', &
         default_value=0._kind_phys, vertical_dim='vertical_layer_dimension', &
         advected=.true., errcode=errcode, errmsg=errmsg)
    if (errcode /= 0) return
    call dyn_const(3)%instantiate(std_name='cloud_ice_dry_mixing_ratio', &
         long_name='cloud ice', diag_name='CLDICE', units='kg kg-1', &
         default_value=0._kind_phys, vertical_dim='vertical_layer_dimension', &
         advected=.true., errcode=errcode, errmsg=errmsg)
  end subroutine const_reg_register
end module const_reg
