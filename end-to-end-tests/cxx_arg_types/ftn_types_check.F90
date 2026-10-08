! Recomputes what cxx_types produced and reports any mismatch through errflg.
! Afterwards flips use_extra, so the optional argument is alternately present
! and absent in successive calls of cxx_types.
module ftn_types_check

  use ccpp_kinds, only: kind_phys
  use gpu_data,   only: gpu_state_type

  implicit none
  private
  public :: ftn_types_check_run

contains

  subroutine ftn_types_check_run(ncol, nlev, col_start, use_extra, extra, state, temp, &
       mask, work, warm, any_warm, gwork, errmsg, errflg)
    integer,                      intent(in)    :: ncol, nlev, col_start
    logical,                      intent(inout) :: use_extra
    real(kind_phys), optional,    intent(in)    :: extra(:,:)
    type(gpu_state_type),         intent(in)    :: state     ! whole host arrays
    real(kind_phys),              intent(in)    :: temp(:,:)
    logical,                      intent(in)    :: mask(:,:)
    real(kind_phys), allocatable, intent(in)    :: work(:)
    logical,                      intent(in)    :: warm(:,:)   ! from cxx_types_gpu
    logical,                      intent(in)    :: any_warm    ! from cxx_types_gpu
    real(kind_phys), allocatable, intent(in)    :: gwork(:)    ! from cxx_types_gpu
    character(len=512),           intent(out)   :: errmsg
    integer,                      intent(out)   :: errflg
    integer :: i, k, g
    real(kind_phys) :: expect
    ! C++ and Fortran may round the same expression differently in the last bit.
    real(kind_phys), parameter :: tol = 1.0e-14_kind_phys

    errmsg = ''
    errflg = 0
    if (present(extra) .neqv. use_extra) then
      errmsg = 'ftn_types_check: extra present/absent disagrees with use_extra'
      errflg = 1
      return
    end if
    if (.not. allocated(work)) then
      errmsg = 'ftn_types_check: work not allocated by cxx_types'
      errflg = 1
      return
    end if
    if (size(work) /= ncol .or. lbound(work, 1) /= 1) then
      errmsg = 'ftn_types_check: work has the wrong shape'
      errflg = 1
      return
    end if
    do i = 1, ncol
      g = col_start - 1 + i
      expect = state%ps(g) * 1.0e-5_kind_phys + merge(1.0_kind_phys, 0.0_kind_phys, use_extra)
      if (abs(work(i) - expect) > tol * abs(expect)) then
        write(errmsg, '(a,i0,a,2es24.15)') 'ftn_types_check: work wrong at column ', i, &
             ': got, expected ', work(i), expect
        errflg = 1
        return
      end if
      if (abs(state%precl(g) - state%ps(g) * 1.0e-12_kind_phys) > tol * state%ps(g) * 1.0e-12_kind_phys) then
        write(errmsg, '(a,i0)') 'ftn_types_check: precl (via descriptor) wrong at column ', i
        errflg = 1
        return
      end if
      do k = 1, nlev
        if (mask(i,k) .neqv. (temp(i,k) < 260.0_kind_phys)) then
          write(errmsg, '(a,2i4)') 'ftn_types_check: mask wrong at ', i, k
          errflg = 1
          return
        end if
      end do
    end do
    ! cxx_types_gpu ran on the device: warm = temp + 0.001*extra > 262, or
    ! temp + 0.0005 > 262 when extra is absent.
    do k = 1, nlev
      do i = 1, ncol
        if (use_extra) then
          expect = temp(i,k) + 0.001_kind_phys * extra(i,k)
        else
          expect = temp(i,k) + 0.0005_kind_phys
        end if
        if (warm(i,k) .neqv. (expect > 262.0_kind_phys)) then
          write(errmsg, '(a,2i4)') 'ftn_types_check: warm (GPU) wrong at ', i, k
          errflg = 1
          return
        end if
      end do
    end do
    if (any_warm .neqv. any(warm)) then
      errmsg = 'ftn_types_check: any_warm (GPU) wrong'
      errflg = 1
      return
    end if
    ! gwork: allocated by cxx_types_gpu through its handle, filled on the
    ! device from the descriptor (precl was written on the host by cxx_types
    ! just before, so a stale device copy of precl would show up here).
    if (.not. allocated(gwork)) then
      errmsg = 'ftn_types_check: gwork not allocated by cxx_types_gpu'
      errflg = 1
      return
    end if
    if (size(gwork) /= ncol) then
      errmsg = 'ftn_types_check: gwork has the wrong size'
      errflg = 1
      return
    end if
    do i = 1, ncol
      g = col_start - 1 + i
      expect = state%ps(g) * 1.0e-5_kind_phys + state%precl(g) * 1.0e6_kind_phys
      if (abs(gwork(i) - expect) > tol * abs(expect)) then
        write(errmsg, '(a,i0,a,2es24.15)') 'ftn_types_check: gwork (GPU) wrong at column ', i, &
             ': got, expected ', gwork(i), expect
        errflg = 1
        return
      end if
    end do
    use_extra = .not. use_extra
  end subroutine ftn_types_check_run

end module ftn_types_check
