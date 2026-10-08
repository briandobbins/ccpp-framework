module gpu_data

  ! Host-model data for the GPU data-movement end-to-end test.
  ! See ../../doc/gpu_data_movement_plan.md.

  use ccpp_kinds, only: kind_phys

  implicit none
  private

  public :: ncols, pver, nchunks, chunk_begin, chunk_end
  public :: temp, qv, extra, use_extra
  public :: gpu_state_type, phys_state
  public :: gpu_data_alloc, gpu_data_dealloc

  integer, parameter :: nchunks = 3
  integer, parameter :: chunk_begin(nchunks) = (/ 1,  9, 17 /)
  integer, parameter :: chunk_end(nchunks)   = (/ 8, 16, 20 /)
  integer, parameter :: ncols = 20
  integer, parameter :: pver  = 6

  !! \section arg_table_gpu_state_type
  !! \htmlinclude gpu_state_type.html
  type gpu_state_type
    real(kind_phys), allocatable :: ps(:)
    real(kind_phys), allocatable :: precl(:)
  end type gpu_state_type

  ! Air temperature: the host changes it between the two groups
  ! (host_access = read_write, the default).
  real(kind_phys), allocatable :: temp(:,:)
  ! Water vapour: host code never touches it between CCPP calls
  ! (host_access = none).
  real(kind_phys), allocatable :: qv(:,:)
  ! Optional field, passed only when use_extra is true.
  real(kind_phys), allocatable, target :: extra(:,:)
  logical :: use_extra = .true.

  type(gpu_state_type) :: phys_state

contains

  subroutine gpu_data_alloc()
    allocate(temp(ncols, pver), qv(ncols, pver), extra(ncols, pver))
    allocate(phys_state%ps(ncols), phys_state%precl(ncols))
  end subroutine gpu_data_alloc

  subroutine gpu_data_dealloc()
    deallocate(temp, qv, extra, phys_state%ps, phys_state%precl)
  end subroutine gpu_data_dealloc

end module gpu_data
