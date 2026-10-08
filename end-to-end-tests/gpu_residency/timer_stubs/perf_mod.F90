! Stand-in for CIME's perf_mod (GPTL) for the --timer-style cesm test.
! Checks that timers nest properly (each t_stopf names the innermost open
! timer) and prints call counts and accumulated wall time.
module perf_mod
  implicit none
  private
  public :: t_startf, t_stopf, timer_report
  integer, parameter :: maxt = 200, maxd = 20
  character(len=64) :: names(maxt) = ''
  integer :: ncalls(maxt) = 0, nt = 0, depth = 0, stack(maxd) = 0
  integer(kind=8) :: total(maxt) = 0, started(maxt) = 0
contains
  integer function find(name)
    character(len=*), intent(in) :: name
    do find = 1, nt
      if (names(find) == name) return
    end do
    nt = nt + 1; names(nt) = name; find = nt
  end function find
  subroutine t_startf(name)
    character(len=*), intent(in) :: name
    integer :: i
    i = find(name)
    depth = depth + 1; stack(depth) = i
    ncalls(i) = ncalls(i) + 1
    call system_clock(started(i))
  end subroutine t_startf
  subroutine t_stopf(name)
    character(len=*), intent(in) :: name
    integer(kind=8) :: now
    call system_clock(now)
    if (depth < 1) error stop 'perf_mod stub: t_stopf without t_startf'
    if (names(stack(depth)) /= name) then
      write(*, '(4a)') 'perf_mod stub: t_stopf(', trim(name), ') but innermost open timer is ', &
           trim(names(stack(depth)))
      error stop 1
    end if
    total(stack(depth)) = total(stack(depth)) + now - started(stack(depth))
    depth = depth - 1
  end subroutine t_stopf
  subroutine timer_report()
    integer :: i
    integer(kind=8) :: rate
    call system_clock(count_rate=rate)
    if (depth /= 0) error stop 'perf_mod stub: timers still open at report'
    write(*, '(a)') 'timer report (name, calls, seconds):'
    do i = 1, nt
      write(*, '(2x,a40,i6,f12.6)') names(i), ncalls(i), real(total(i)) / real(rate)
    end do
  end subroutine timer_report
end module perf_mod
