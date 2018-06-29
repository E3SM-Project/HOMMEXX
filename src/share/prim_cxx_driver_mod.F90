#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif

module prim_cxx_driver_mod

  use iso_c_binding, only: c_int

  implicit none

  public :: init_cxx_mpi_comm
  public :: init_cxx_connectivity

  private :: generate_global_to_local
  private :: init_cxx_connectivity_internal

#include <mpif.h>

contains

  subroutine init_cxx_mpi_comm (f_comm)
    use iso_c_binding, only: c_int
    !
    ! Interfaces
    !
    interface
      subroutine reset_cxx_comm (f_comm) bind(c)
        use iso_c_binding, only: c_int
        !
        ! Inputs
        !
        integer(kind=c_int), intent(in) :: f_comm
      end subroutine reset_cxx_comm
    end interface
    !
    ! Inputs
    !
    integer, intent(in) :: f_comm

    call reset_cxx_comm (INT(f_comm,c_int))
  end subroutine init_cxx_mpi_comm

  subroutine init_cxx_connectivity (nelemd, GridEdge, MetaVertex, par)
    use dimensions_mod, only : nelem
    use gridgraph_mod,  only : GridEdge_t
    use metagraph_mod,  only : MetaVertex_t
    use parallel_mod,   only : parallel_t
    !
    ! Inputs
    !
    integer, intent(in) :: nelemd
    type(GridEdge_t),   intent(in) :: GridEdge(:)
    type (parallel_t),  intent(in) :: par
    type(MetaVertex_t), intent(in) :: MetaVertex
    !
    ! Locals
    !
    integer :: Global2Local(nelem)

    call generate_global_to_local(MetaVertex,Global2Local,par)

    call init_cxx_connectivity_internal (nelemd, Global2Local, Gridedge)

  end subroutine init_cxx_connectivity

  subroutine generate_global_to_local (MetaVertex, Global2Local, par)
    use dimensions_mod, only : nelem
    use metagraph_mod,  only : MetaVertex_t
    use parallel_mod,   only : parallel_t, MPI_MAX, MPIinteger_t
    !
    ! Inputs
    !
    type (parallel_t),  intent(in) :: par
    type(MetaVertex_t), intent(in) :: MetaVertex
    integer, intent(out) :: Global2Local(nelem)
    !
    ! Locals
    !
    integer :: ie, ierr

    ! Defaults all local ids to 0 (meaning not on this process)
    Global2Local = 0
    do ie=1,SIZE(MetaVertex%members)
      Global2Local(MetaVertex%members(ie)%number) = ie
    enddo

    call MPI_Allreduce(MPI_IN_PLACE,Global2Local,nelem,MPIinteger_t,MPI_MAX,par%comm,ierr)

  end subroutine generate_global_to_local

  subroutine init_cxx_connectivity_internal (nelemd,Global2Local,GridEdge)
    use gridgraph_mod,  only : GridEdge_t
    use dimensions_mod, only : nelem
    !
    ! Interfaces
    !
    interface
      subroutine init_connectivity (num_local_elems) bind (c)
        use iso_c_binding, only : c_int
        !
        ! Inputs
        !
        integer (kind=c_int), intent(in) :: num_local_elems
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
  end subroutine init_cxx_connectivity_internal

end module prim_cxx_driver_mod
