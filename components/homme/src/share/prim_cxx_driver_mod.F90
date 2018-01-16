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

  subroutine init_cxx_mpi_structures (nelemd, GridEdge, MetaVertex)
    use dimensions_mod, only : nelem
    use gridgraph_mod,  only : GridEdge_t
    use metagraph_mod,  only : MetaVertex_t
    use parallel_mod,   only : parallel_t, abortmp, MPI_INTEGER
    !
    ! Inputs
    !
    integer, intent(in) :: nelemd
    type(GridEdge_t), intent(in) :: GridEdge(:)
    type(MetaVertex_t), intent(in) :: MetaVertex
    !
    ! Locals
    !
    integer :: Global2Local(nelem)

    call generate_global_to_local(MetaVertex,Global2Local)

    call init_c_connectivity (nelemd, Global2Local, Gridedge)

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

  subroutine generate_global_to_local (MetaVertex, Global2Local)
    use dimensions_mod, only : nelem
    use metagraph_mod,  only : MetaVertex_t
    !
    ! Inputs
    !
    type(MetaVertex_t), intent(in) :: MetaVertex
    integer, intent(out) :: Global2Local(nelem)
    !
    ! Locals
    !
    integer :: ie

    ! Defaults all local ids to 0 (meaning not on this process)
    Global2Local = 0
    do ie=1,SIZE(MetaVertex%members)
      Global2Local(MetaVertex%members(ie)%number) = ie
    enddo
  end subroutine generate_global_to_local

  subroutine init_c_connectivity (nelemd,Global2Local,GridEdge)
    use gridgraph_mod,  only : GridEdge_t
    use dimensions_mod, only : nelem
    !
    ! Interfaces
    !
    interface
      subroutine init_connectivity (nelemd) bind (c)
        use iso_c_binding, only : c_int
        !
        ! Inputs
        !
        integer (kind=c_int), intent(in) :: nelemd
      end subroutine init_connectivity

      subroutine finalize_connectivity () bind(c)
      end subroutine finalize_connectivity

      subroutine add_connection (first_lid,  first_gid,  first_pos,  first_pid, &
                                 second_lid, second_gid, second_pos, second_pid) bind(c)
        use iso_c_binding, only : c_int
        !
        ! Inputs
        !
        integer (kind=c_int), intent(in) :: first_lid,  first_gid,  first_pos,  first_pid
        integer (kind=c_int), intent(in) :: second_lid, second_gid, second_pos, second_pid
      end subroutine add_connection
    end interface
    !
    ! Inputs
    !
    integer, dimension(nelem), intent(in) :: Global2Local
    integer, intent(in) :: nelemd
    type(GridEdge_t), intent(in) :: GridEdge(:)
    !
    ! Locals
    !
    integer :: ie, num_edges
    type(GridEdge_t) :: e

    call init_connectivity(nelemd)

    num_edges = SIZE(GridEdge)
    do ie=1,num_edges
      e = GridEdge(ie)
      call add_connection(Global2Local(e%head%number),e%head%number,e%head_dir,e%head%processor_number, &
                          Global2Local(e%tail%number),e%tail%number,e%tail_dir,e%tail%processor_number)
    enddo

    call finalize_connectivity()
  end subroutine init_c_connectivity

end module prim_cxx_driver_mod

! USE_KOKKOS_KERNELS
#endif
