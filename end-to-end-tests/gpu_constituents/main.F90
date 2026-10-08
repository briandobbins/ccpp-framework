program test_gpu_constituents

  ! End-to-end test for GPU data movement of CCPP constituents
  ! (doc/gpu_data_movement_plan.md).  Three constituents are registered by a
  ! scheme; a GPU scheme and a CPU scheme each update one of them and its
  ! tendency, then a GPU scheme applies all tendencies at once.  A CPU
  ! check runs in a second group, and host code changes every constituent
  ! between time steps (a stand-in for the dycore).  Writes every final
  ! constituent value to gpu_constituents_out.txt; CPU and GPU builds must
  ! agree.  Build with -DNO_DEVICE_SYNC when caps lack --directive.

  use, intrinsic :: iso_fortran_env, only: error_unit
  use ccpp_kinds, only: kind_phys
  use ccpp_constituent_prop_mod, only: ccpp_constituent_properties_t
  use const_data, only: ncols, pver, temp, nchunks, chunk_begin, chunk_end
  use test_host_ccpp_cap, only: ccpp_register, ccpp_init, ccpp_physics_init, &
      ccpp_physics_timestep_init, ccpp_physics_run, ccpp_physics_timestep_final, &
      ccpp_physics_final, ccpp_final, ccpp_register_constituents, &
      ccpp_initialize_constituents, ccpp_constituents_array, &
      ccpp_const_get_index, ccpp_deallocate_dynamic_constituents
#ifndef NO_DEVICE_SYNC
  use test_host_ccpp_cap, only: ccpp_physics_device_update_device, &
      ccpp_physics_device_update_host
#endif

  implicit none

  character(len=*), parameter :: suite = 'gpu_const_suite'
  integer, parameter :: nsteps = 3
  type(ccpp_constituent_properties_t), allocatable :: host_constituents(:)
  real(kind_phys), pointer :: const(:,:,:) => null()
  integer :: i, k, m, step, ic, errflg, unit, iq, il, ii
  character(len=512) :: errmsg

  allocate(temp(ncols, pver))
  do k = 1, pver
    do i = 1, ncols
      temp(i,k) = 250.0_kind_phys + real(i, kind_phys) + 2.0_kind_phys*real(k, kind_phys)
    end do
  end do

  call ccpp_register(suite_name=suite, errmsg=errmsg, errflg=errflg)
  call check('ccpp_register')
  allocate(host_constituents(0))
  call ccpp_register_constituents(host_constituents, errmsg=errmsg, errcode=errflg)
  call check('ccpp_register_constituents')
  call ccpp_initialize_constituents(ncols=ncols, num_layers=pver, errcode=errflg, errmsg=errmsg)
  call check('ccpp_initialize_constituents')

  const => ccpp_constituents_array()
  call ccpp_const_get_index('water_vapor_mixing_ratio_wrt_moist_air', iq, errflg, errmsg)
  call ccpp_const_get_index('cloud_liquid_dry_mixing_ratio', il, errflg, errmsg)
  call ccpp_const_get_index('cloud_ice_dry_mixing_ratio', ii, errflg, errmsg)
  call check('ccpp_const_get_index')
  do k = 1, pver
    do i = 1, ncols
      const(i,k,iq) = 1.0e-3_kind_phys*(1.0_kind_phys + 0.01_kind_phys*real(i, kind_phys))
      const(i,k,il) = 1.0e-5_kind_phys*real(k, kind_phys)
      const(i,k,ii) = 2.0e-5_kind_phys*real(k, kind_phys)
    end do
  end do

  call ccpp_init(suite_name=suite, errmsg=errmsg, errflg=errflg)
  call check('ccpp_init')
#ifndef NO_DEVICE_SYNC
  call ccpp_physics_device_update_device(suite_name=suite, errmsg=errmsg, errflg=errflg)
  call check('ccpp_physics_device_update_device')
#endif
  call ccpp_physics_init(suite_name=suite, group_name='all', lb=1, ub=ncols, &
       thread_num=1, nthreads=1, nphys_threads=1, errmsg=errmsg, errflg=errflg)
  call check('ccpp_physics_init')

  do step = 1, nsteps
    call ccpp_physics_timestep_init(suite_name=suite, group_name='all', lb=1, ub=ncols, &
         thread_num=1, nthreads=1, nphys_threads=1, errmsg=errmsg, errflg=errflg)
    call check('ccpp_physics_timestep_init')
    do ic = 1, nchunks
      call ccpp_physics_run(suite_name=suite, group_name='phys', lb=chunk_begin(ic), &
           ub=chunk_end(ic), thread_num=1, nthreads=1, nphys_threads=1, &
           errmsg=errmsg, errflg=errflg)
      call check('ccpp_physics_run(phys)')
    end do
    do ic = 1, nchunks
      call ccpp_physics_run(suite_name=suite, group_name='diag', lb=chunk_begin(ic), &
           ub=chunk_end(ic), thread_num=1, nthreads=1, nphys_threads=1, &
           errmsg=errmsg, errflg=errflg)
      call check('ccpp_physics_run(diag)')
    end do
    call ccpp_physics_timestep_final(suite_name=suite, group_name='all', lb=1, ub=ncols, &
         thread_num=1, nthreads=1, nphys_threads=1, errmsg=errmsg, errflg=errflg)
    call check('ccpp_physics_timestep_final')
    ! Host code between steps (a stand-in for the dycore): changes every
    ! constituent through the host's own pointer.
    const = 0.999_kind_phys*const
  end do

#ifndef NO_DEVICE_SYNC
  call ccpp_physics_device_update_host(suite_name=suite, errmsg=errmsg, errflg=errflg)
  call check('ccpp_physics_device_update_host')
#endif
  open(newunit=unit, file='gpu_constituents_out.txt', status='replace', action='write')
  do m = 1, size(const, 3)
    do k = 1, pver
      do i = 1, ncols
        write(unit, '(3i4,es26.17)') i, k, m, const(i,k,m)
      end do
    end do
  end do
  close(unit)
  write(*, '(a,3es24.15)') 'checksums q liq ice:', sum(const(:,:,iq)), &
       sum(const(:,:,il)), sum(const(:,:,ii))

  call ccpp_physics_final(suite_name=suite, group_name='all', lb=1, ub=ncols, &
       thread_num=1, nthreads=1, nphys_threads=1, errmsg=errmsg, errflg=errflg)
  call check('ccpp_physics_final')
  call ccpp_final(suite_name=suite, errmsg=errmsg, errflg=errflg)
  call check('ccpp_final')
  call ccpp_deallocate_dynamic_constituents()
  write(*, '(a)') 'test_gpu_constituents: completed'

contains

  subroutine check(where)
    character(len=*), intent(in) :: where
    if (errflg /= 0) then
      write(error_unit, '(a)') 'Error in '//where//': '//trim(errmsg)
      stop 1
    end if
  end subroutine check

end program test_gpu_constituents
