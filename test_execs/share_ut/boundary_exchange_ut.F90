module boundary_exchange_ut

  use edgetype_mod,   only : EdgeBuffer_t
  use element_mod,    only : element_t
  use gridgraph_mod,  only : GridVertex_t, GridEdge_t
  use hybrid_mod,     only : hybrid_t
  use kinds,          only : real_kind
  use metagraph_mod,  only : MetaVertex_t
  use parallel_mod,   only : parallel_t

  implicit none

  type (GridVertex_t), dimension(:), allocatable :: GridVertex
  type (GridEdge_t),   dimension(:), allocatable :: GridEdge

  type (MetaVertex_t) :: MetaVertex
  type (parallel_t)   :: par
  type (hybrid_t)     :: hybrid
  type (EdgeBuffer_t) :: edge

  type (element_t), dimension(:), allocatable :: elem

  integer :: nelemd

contains

  subroutine initmp_f90 () bind(c)
    use parallel_mod, only : initmp
    use hybrid_mod,   only : hybrid_create

    par = initmp()

    ! No horizontal openmp, for now
    hybrid = hybrid_create(par,0,1)
  end subroutine initmp_f90


  subroutine init_cube_geometry_f90 (ne_in) bind(c)
    use iso_c_binding,  only : c_int
    use cube_mod,       only : CubeTopology, CubeElemCount, CubeEdgeCount
    use dimensions_mod, only : nelem, npart
    use dimensions_mod, only : ne
    use gridgraph_mod,  only : allocate_gridvertex_nbrs
    use spacecurve_mod, only : genspacepart
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: ne_in
    !
    ! Locals
    !
    integer :: ie, num_edges

    ne = ne_in
    nelem = CubeElemCount()
    num_edges = CubeEdgeCount()

    ! Allocate
    allocate (GridVertex(nelem))
    allocate (GridEdge(num_edges))

    do ie=1,nelem
      call allocate_gridvertex_nbrs(GridVertex(ie))
    enddo

    ! Generate mesh connectivity
    call CubeTopology(GridEdge, GridVertex)

    ! Set number of partitions before partitioning mesh
    npart = par%nprocs

    ! Partition mesh among processes
    call genspacepart(GridEdge, GridVertex)

  end subroutine init_cube_geometry_f90

  subroutine init_connectivity_f90 (num_scalar_fields_2d,num_scalar_fields_3d,num_vector_fields_3d,vector_dim) bind(c)
    use iso_c_binding,  only : c_int
    use dimensions_mod, only : nelem, nelemd, nlev
    use edge_mod_base,  only : initEdgeBuffer
    use element_mod,    only : allocate_element_desc
    use metagraph_mod,  only : initMetaGraph, LocalElemCount
    use parallel_mod,   only : iam, syncmp, abortmp, MPI_INTEGER
    use schedtype_mod,  only : Schedule
    use schedule_mod,   only : genEdgeSched

    !
    ! Inputs
    !
    integer(kind=c_int), intent(in) :: num_scalar_fields_2d, num_scalar_fields_3d, num_vector_fields_3d, vector_dim
    !
    ! Locals
    !
    integer :: Global2Local(nelem)
    integer, allocatable, target :: Local2Global(:,:)
    integer, pointer :: map_ptr(:)
    integer :: nelemd_max, ie, ip, ierr
    integer, target :: my_num_elems
    integer, target, allocatable :: num_elems(:)
    integer, pointer :: my_num_elems_ptr, num_elems_ptr(:)


    call initMetaGraph(iam,MetaVertex,GridVertex,GridEdge)

    nelemd = LocalElemCount(MetaVertex)

    allocate (num_elems(par%nprocs))
    my_num_elems_ptr => my_num_elems
    num_elems_ptr => num_elems
    my_num_elems = nelemd

    allocate (elem(nelemd))
    call allocate_element_desc(elem)

    allocate (Schedule(1))
    call genEdgeSched(elem,iam,Schedule(1),MetaVertex)

    call initEdgeBuffer(par,edge,elem,num_scalar_fields_2d + num_scalar_fields_3d*nlev + vector_dim*num_vector_fields_3d*nlev)

    call syncmp(par)

    ! Differently from F90, all processors need to know the local id of all elements on all processes.
    ! This is because each process needs to be able to write in the proper position in the RMA of other processes,
    ! thus knowing the local id of all the remote elements it needs to sum into

    ! First, gather the number of elements on each process
    call MPI_ALLGATHER(my_num_elems_ptr,1,MPI_INTEGER,num_elems_ptr,1,MPI_INTEGER,par%comm,ierr)

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

    ! Pass info to C
    call init_c_connectivity_f90 (nelemd,Global2Local)

  end subroutine init_connectivity_f90

  subroutine init_c_connectivity_f90 (nelemd,Global2Local)
    use gridgraph_mod,  only : GridEdge_t
    use dimensions_mod, only : nelem
    use parallel_mod,   only : abortmp
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
#include <mpif.h>

    !
    ! Inputs
    !
    integer, dimension(nelem), intent(in) :: Global2Local
    integer, intent(in) :: nelemd
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
  end subroutine init_c_connectivity_f90

  subroutine boundary_exchange_test_f90 (field_2d_ptr, field_3d_ptr, field_4d_ptr, &
                                         inner_dim_4d, num_time_levels,            &
                                         idim_2d, idim_3d, idim_4d) bind(c)
    use iso_c_binding,  only : c_ptr, c_f_pointer, c_int
    use dimensions_mod, only : np, nlev, nelemd
    use edge_mod_base,  only : edgevpack, edgevunpack
    use bndry_mod,      only : bndry_exchangev
    !
    ! Inputs
    !
    type (c_ptr), intent(in) :: field_2d_ptr, field_3d_ptr, field_4d_ptr
    integer (kind=c_int), intent(in) :: inner_dim_4d, num_time_levels
    integer (kind=c_int), intent(in) :: idim_2d, idim_3d, idim_4d
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:,:),     pointer :: field_2d
    real (kind=real_kind), dimension(:,:,:,:,:),   pointer :: field_3d
    real (kind=real_kind), dimension(:,:,:,:,:,:), pointer :: field_4d
    integer :: ie, kptr

    call c_f_pointer(field_2d_ptr, field_2d, [np,np,num_time_levels,nelemd])
    call c_f_pointer(field_3d_ptr, field_3d, [np,np,nlev,num_time_levels,nelemd])
    call c_f_pointer(field_4d_ptr, field_4d, [np,np,nlev,inner_dim_4d,num_time_levels,nelemd])

    do ie=1,nelemd
      kptr = 0
      call edgeVpack(edge,field_2d(:,:,idim_2d,ie),1,kptr,ie)

      kptr = 1
      call edgeVpack(edge,field_3d(:,:,:,idim_3d,ie),nlev,kptr,ie)

      kptr = 1 + nlev
      call edgeVpack(edge,field_4d(:,:,:,:,idim_4d,ie),inner_dim_4d*nlev,kptr,ie)
    enddo

    call bndry_exchangev(hybrid,edge)

    do ie=1,nelemd
      kptr = 0
      call edgeVunpack(edge,field_2d(:,:,idim_2d,ie),1,kptr,ie)

      kptr = 1
      call edgeVunpack(edge,field_3d(:,:,:,idim_3d,ie),nlev,kptr,ie)

      kptr = 1 + nlev
      call edgeVunpack(edge,field_4d(:,:,:,:,idim_4d,ie),inner_dim_4d*nlev,kptr,ie)
    enddo
  end subroutine boundary_exchange_test_f90

  subroutine cleanup_f90 () bind(c)
    use schedtype_mod, only : Schedule
    use edge_mod_base, only : FreeEdgeBuffer

    call FreeEdgeBuffer(edge)

    deallocate(elem)
    deallocate(GridVertex)
    deallocate(GridEdge)
    deallocate(Schedule)
  end subroutine cleanup_f90

end module boundary_exchange_ut
