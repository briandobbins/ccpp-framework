! Stand-in for FMS mpp_mod clocks for the --timer-style fms test.  Clock
! names are limited to 32 characters, as in FMS.  Checks proper nesting and
! prints call counts and accumulated wall time.
module mpp_mod
  implicit none
  private
  public :: mpp_clock_id, mpp_clock_begin, mpp_clock_end, CLOCK_ROUTINE, timer_report
  integer, parameter :: CLOCK_ROUTINE = 15
  integer, parameter :: maxt = 200, maxd = 20
  character(len=32) :: names(maxt) = ''
  integer :: ncalls(maxt) = 0, nt = 0, depth = 0, stack(maxd) = 0
  integer(kind=8) :: total(maxt) = 0, started(maxt) = 0
contains
  integer function mpp_clock_id(name, flags, grain)
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: flags, grain
    if (len_trim(name) > 32) error stop 'mpp_mod stub: clock name longer than 32'
    do mpp_clock_id = 1, nt
      if (names(mpp_clock_id) == name) return
    end do
    nt = nt + 1; names(nt) = name; mpp_clock_id = nt
  end function mpp_clock_id
  subroutine mpp_clock_begin(id)
    integer, intent(in) :: id
    if (id < 1 .or. id > nt) error stop 'mpp_mod stub: bad clock id'
    depth = depth + 1; stack(depth) = id
    ncalls(id) = ncalls(id) + 1
    call system_clock(started(id))
  end subroutine mpp_clock_begin
  subroutine mpp_clock_end(id)
    integer, intent(in) :: id
    integer(kind=8) :: now
    call system_clock(now)
    if (depth < 1) error stop 'mpp_mod stub: mpp_clock_end without begin'
    if (stack(depth) /= id) error stop 'mpp_mod stub: clock end does not match innermost begin'
    total(id) = total(id) + now - started(id)
    depth = depth - 1
  end subroutine mpp_clock_end
  subroutine timer_report()
    integer :: i
    integer(kind=8) :: rate
    call system_clock(count_rate=rate)
    if (depth /= 0) error stop 'mpp_mod stub: clocks still open at report'
    write(*, '(a)') 'timer report (name, calls, seconds):'
    do i = 1, nt
      write(*, '(2x,a40,i6,f12.6)') names(i), ncalls(i), real(total(i)) / real(rate)
    end do
  end subroutine timer_report
end module mpp_mod
