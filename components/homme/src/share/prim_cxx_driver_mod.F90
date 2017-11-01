#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif

#ifdef USE_KOKKOS_KERNELS

module prim_cxx_driver_mod

  implicit none

  public :: init_cxx_mpi_structures
  public :: cleanup_cxx_structures

  private :: generate_global_to_local
  private :: init_c_connectivity

#include <mpif.h>

contains

  subroutine init_cxx_mpi_structures (nelemd, par, GridEdge, MetaVertex)
    use dimensions_mod, only : nelem
    use gridgraph_mod,  only : GridEdge_t
    use metagraph_mod,  only : MetaVertex_t
    use parallel_mod,   only : parallel_t, abortmp, MPI_INTEGER
    !
    ! Inputs
    !
    integer, intent(in) :: nelemd
    type(parallel_t), intent(in) :: par
    type(GridEdge_t), intent(in) :: GridEdge(:)
    type(MetaVertex_t), intent(in) :: MetaVertex
    !
    ! Locals
    !
    integer :: Global2Local(nelem)

    call generate_global_to_local(nelemd,par,MetaVertex,Global2Local)

    call init_c_connectivity (nelemd, par, Global2Local, Gridedge)

  end subroutine init_cxx_mpi_structures

  subroutine cleanup_cxx_structures ()
    !
    ! Interfaces
    !
    interface
      subroutine cleanup_mpi_structures () bind(c)
      end subroutine cleanup_mpi_structures
    end interface

    call cleanup_mpi_structures()
  end subroutine cleanup_cxx_structures

  subroutine generate_global_to_local (nelemd, par, MetaVertex, Global2Local)
    use dimensions_mod, only : nelem
    use metagraph_mod,  only : MetaVertex_t
    use parallel_mod,   only : parallel_t, abortmp, MPI_INTEGER
    !
    ! Inputs
    !
    integer, intent(in) :: nelemd
    type(parallel_t), intent(in) :: par
    type(MetaVertex_t), intent(in) :: MetaVertex
    integer, intent(out) :: Global2Local(nelem)
    !
    ! Locals
    !
    integer, allocatable, target :: Local2Global(:,:)
    integer, allocatable :: num_elems(:)
    integer :: ie, ip, ierr, nelemd_max
    integer, pointer :: map_ptr(:)

    ! Differently from F90, all processors need to know the local id of all elements on all processes.
    ! This is because each process needs to be able to write in the proper position in the RMA of other processes,
    ! thus knowing the local id of all the remote elements it needs to sum into

    allocate(num_elems(par%nprocs))

    ! First, gather the number of elements on each process
    call MPI_ALLGATHER(nelemd,1,MPI_INTEGER,num_elems,1,MPI_INTEGER,par%comm,ierr)

    ! Allocate a 2d table of local->global, using a uniform upper bound for the number of element on each process
    nelemd_max = MAXVAL(num_elems)
    allocate (Local2Global(par%nprocs, nelemd_max))

    ! Create the local->global map, and fill it with -1 (useful for debug)
    Local2Global = -1
    do ie=1,SIZE(MetaVertex%members)
      Local2Global(par%rank+1,ie) = MetaVertex%members(ie)%number
    enddo

    do ip=1,par%nprocs
      ! Scatter the element ids owned by this process
      map_ptr => Local2Global(ip,:)
      call MPI_BCAST(map_ptr,num_elems(ip),MPI_INTEGER,ip-1,par%comm,ierr)
    enddo

    ! Fill Global2Local with -1. All should be overwritten, but this way we can check
    Global2Local = -1
    do ip=1,par%nprocs
      do ie=1,num_elems(ip)
        Global2Local(Local2Global(ip,ie)) = ie
      enddo
    enddo

    ierr = MINVAL(Global2Local)
    if (ierr .eq. -1) then
      call abortmp ("Error! Somehow one or more element global ID were not correcdimy mapped to local id's.")
    endif
  end subroutine generate_global_to_local

  subroutine init_c_connectivity (nelemd,par,Global2Local,GridEdge)
    use gridgraph_mod,  only : GridEdge_t
    use dimensions_mod, only : nelem
    use parallel_mod,   only : parallel_t, abortmp, MPI_INTEGER
    !
    ! Interfaces
    !
    interface
      subroutine init_connectivity (nelemd,num_local,num_shared) bind (c)
        use iso_c_binding, only : c_int
        !
        ! Inputs
        !
        integer (kind=c_int), intent(in) :: nelemd, num_local, num_shared
      end subroutine init_connectivity

      subroutine finalize_connectivity () bind(c)
      end subroutine finalize_connectivity

      subroutine add_connection (first_lid, first_pos, first_pid, second_lid, second_pos, second_pid) bind(c)
        use iso_c_binding, only : c_int
        !
        ! Inputs
        !
        integer (kind=c_int), intent(in) :: first_lid,  first_pos,  first_pid
        integer (kind=c_int), intent(in) :: second_lid, second_pos, second_pid
      end subroutine add_connection
    end interface
    !
    ! Inputs
    !
    integer, dimension(nelem), intent(in) :: Global2Local
    integer, intent(in) :: nelemd
    type(parallel_t), intent(in) :: par
    type(GridEdge_t), intent(in) :: GridEdge(:)
    !
    ! Locals
    !
    integer :: ie, num_edges, num_local, num_shared, num_total_global, num_total_on_proc, ierr
    type(GridEdge_t) :: e

    num_local = 0
    num_shared = 0

    num_edges = SIZE(GridEdge)

    do ie=1,num_edges
      e = GridEdge(ie)
      if (e%head%processor_number-1 .eq. par%rank) then
        if(e%head%processor_number .eq. e%tail%processor_number) then
          num_local = num_local+1
        else
          num_shared = num_shared+1
        endif
      endif
    enddo

    ! The following 5 lines are for debug purposes only
    num_total_on_proc = num_shared+num_local
    call MPI_Allreduce(num_total_on_proc,num_total_global,1,MPI_INTEGER,MPI_SUM,par%comm,ierr)
    if (num_total_global /= num_edges) then
      call abortmp ("Error! Somehow the number of local and shared edges don't add up to the total number of edges")
    endif

    call init_connectivity(nelemd, num_local, num_shared)

    do ie=1,num_edges
      e = GridEdge(ie)
      call add_connection(Global2Local(e%head%number),e%head_dir,e%head%processor_number, &
                          Global2Local(e%tail%number),e%tail_dir,e%tail%processor_number)
    enddo

    call finalize_connectivity()
  end subroutine init_c_connectivity

end module prim_cxx_driver_mod

! USE_KOKKOS_KERNELS
#endif
