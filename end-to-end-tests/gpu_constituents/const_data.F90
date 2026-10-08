module const_data
  ! Host data for the GPU constituent test.
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: ncols, pver, dt, temp, nchunks, chunk_begin, chunk_end
  integer, parameter :: nchunks = 3
  integer, parameter :: chunk_begin(nchunks) = (/ 1,  9, 17 /)
  integer, parameter :: chunk_end(nchunks)   = (/ 8, 16, 20 /)
  integer, parameter :: ncols = 20
  integer, parameter :: pver  = 6
  real(kind_phys), parameter :: dt = 60.0_kind_phys
  real(kind_phys), allocatable :: temp(:,:)
end module const_data
