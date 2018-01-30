module hyperviscosity_functor_ut

  use derivative_mod_base, only : derivative_t
  use edgetype_mod,        only : EdgeBuffer_t
  use element_mod,         only : element_t
  use gridgraph_mod,       only : GridVertex_t, GridEdge_t
  use hybrid_mod,          only : hybrid_t
  use kinds,               only : real_kind
  use metagraph_mod,       only : MetaVertex_t
  use parallel_mod,        only : parallel_t
  use quadrature_mod,      only : quadrature_t

  implicit none

  type (GridVertex_t), dimension(:), allocatable :: GridVertex
  type (GridEdge_t),   dimension(:), allocatable :: GridEdge

  type (MetaVertex_t) :: MetaVertex
  type (parallel_t)   :: par
  type (hybrid_t)     :: hybrid
  type (EdgeBuffer_t) :: edge
  type (derivative_t) :: deriv
  type (quadrature_t) :: gp

  type (element_t), dimension(:), allocatable :: elem

  integer :: nelemd

#include <mpif.h>

contains

  subroutine setup_test_f90 (ne_in, h_d_ptr, h_dinv_ptr, h_mp_ptr, h_spheremp_ptr, h_rspheremp_ptr, &
                             h_metdet_ptr, h_metinv_ptr, h_vec_sph2cart_ptr, h_tensorVisc_ptr) bind(c)
    use iso_c_binding,  only : c_ptr, c_f_pointer, c_int, c_double
    use dimensions_mod, only : np
    use quadrature_mod, only : gausslobatto
    !
    ! Inputs
    !
    type(c_ptr), intent(in) :: h_d_ptr, h_dinv_ptr, h_mp_ptr, h_spheremp_ptr, h_rspheremp_ptr, &
                               h_metdet_ptr, h_metinv_ptr, h_vec_sph2cart_ptr, h_tensorVisc_ptr
    integer(kind=c_int), intent(in) :: ne_in
    !
    ! Locals
    !
    real(kind=c_double), pointer, dimension(:,:,:)     :: mp, spheremp, rspheremp, metdet
    real(kind=c_double), pointer, dimension(:,:,:,:,:) :: d, dinv, metinv, tensorVisc
    real(kind=c_double), pointer, dimension(:,:,:,:,:) :: vec_sph2cart
    integer :: ie

    call initmp_f90()
    gp = gausslobatto(np)
    call init_cube_geometry_f90(ne_in)
    call init_derivative_f90()
    call init_connectivity_f90()

    call c_f_pointer(h_d_ptr,            d,              [np,np,2,2,nelemd]);
    call c_f_pointer(h_dinv_ptr,         dinv,           [np,np,2,2,nelemd]);
    call c_f_pointer(h_mp_ptr,           mp,             [np,np,    nelemd]);
    call c_f_pointer(h_spheremp_ptr,     spheremp,       [np,np,    nelemd]);
    call c_f_pointer(h_rspheremp_ptr,    rspheremp,      [np,np,    nelemd]);
    call c_f_pointer(h_metdet_ptr,       metdet,         [np,np,    nelemd]);
    call c_f_pointer(h_metinv_ptr,       metinv,         [np,np,2,2,nelemd]);
    call c_f_pointer(h_vec_sph2cart_ptr, vec_sph2cart,   [np,np,3,2,nelemd]);
    call c_f_pointer(h_tensorVisc_ptr,   tensorVisc,     [np,np,2,2,nelemd]);

    do ie=1,nelemd
      elem(ie)%d(:,:,:,:)               = d(:,:,:,:,ie)
      elem(ie)%dinv(:,:,:,:)            = dinv(:,:,:,:,ie)
      elem(ie)%mp(:,:)                  = mp(:,:,ie)
      elem(ie)%spheremp(:,:)            = spheremp(:,:,ie)
      elem(ie)%rspheremp(:,:)           = rspheremp(:,:,ie)
      elem(ie)%metdet(:,:)              = metdet(:,:,ie)
      elem(ie)%metinv(:,:,:,:)          = metinv(:,:,:,:,ie)
      elem(ie)%vec_sphere2cart(:,:,:,:) = vec_sph2cart(:,:,:,:,ie)
      elem(ie)%tensorVisc(:,:,:,:)      = tensorVisc(:,:,:,:,ie)
    enddo
  end subroutine setup_test_f90

  subroutine initmp_f90 () bind(c)
    use parallel_mod, only : initmp
    use hybrid_mod,   only : hybrid_create

    par = initmp()

    ! No horizontal openmp, for now
    hybrid = hybrid_create(par,0,1)
  end subroutine initmp_f90

  subroutine init_cube_geometry_f90 (ne_in) bind(c)
    use iso_c_binding,   only : c_int
    use cube_mod,        only : CubeTopology, CubeElemCount, CubeEdgeCount
    use dimensions_mod,  only : nelem, npart
    use dimensions_mod,  only : ne
    use gridgraph_mod,   only : allocate_gridvertex_nbrs
    use mass_matrix_mod, only : mass_matrix
    use spacecurve_mod,  only : genspacepart
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

    do ie=1,nelemd
      call set_corner_coordinates(elem(ie))
    enddo
    call assign_node_numbers_to_elem(elem,GridVertex)

    do ie=1,nelemd
      call cube_init_atomic(elem(ie),gp%points)
    enddo

    call mass_matrix(par,elem)

  end subroutine init_cube_geometry_f90

  subroutine init_derivative_f90()
    use derivative_mod_base, only: derivinit
    use dimensions_mod,      only: np
    !
    ! Locals
    !


  end subroutine init_derivative_f90

  subroutine init_connectivity_f90 ()
    use dimensions_mod, only : nelem, nelemd, nlev
    use edge_mod_base,  only : initEdgeBuffer, initEdgeSBuffer
    use element_mod,    only : allocate_element_desc
    use metagraph_mod,  only : initMetaGraph, LocalElemCount
    use parallel_mod,   only : iam, MPI_MAX, MPIinteger_t
    use schedtype_mod,  only : Schedule
    use schedule_mod,   only : genEdgeSched
    !
    ! Locals
    !
    integer :: Global2Local(nelem)
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

    call initEdgeBuffer(par,edge,elem,4*nlev) ! t, dp3d and v

    ! Defaults all local ids to 0 (meaning not on this process)
    Global2Local = 0
    do ie=1,SIZE(MetaVertex%members)
      Global2Local(MetaVertex%members(ie)%number) = ie
    enddo
    call MPI_Allreduce(MPI_IN_PLACE,Global2Local,nelem,MPIinteger_t,MPI_MAX,par%comm,ierr)

    ! Pass info to C
    call init_c_connectivity_f90 (nelemd,Global2Local)

  end subroutine init_connectivity_f90

  subroutine init_c_connectivity_f90 (nelemd,Global2Local)
    use gridgraph_mod,  only : GridEdge_t
    use dimensions_mod, only : nelem
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
    !
    ! Locals
    !
    integer :: ie, num_edges, ierr
    type(GridEdge_t) :: e

    call init_connectivity(nelemd)

    num_edges = SIZE(GridEdge)
    do ie=1,num_edges
      e = GridEdge(ie)
      call add_connection(Global2Local(e%head%number),e%head%number,e%head_dir,e%head%processor_number, &
                          Global2Local(e%tail%number),e%tail%number,e%tail_dir,e%tail%processor_number)
    enddo

    call finalize_connectivity()
  end subroutine init_c_connectivity_f90

  subroutine hyperviscosity_test_f90(temperature_ptr, dp3d_ptr, velocity_ptr, itl) bind(c)
    use iso_c_binding,      only : c_ptr, c_f_pointer, c_int
    use dimensions_mod,     only : nc, np, nlev
    use element_mod,        only : timelevels
    use viscosity_mod_base, only : biharmonic_wk_dp3d
    !
    ! Inputs
    !
    type (c_ptr), intent(in) :: temperature_ptr, dp3d_ptr, velocity_ptr
    integer(kind=c_int), intent(in) :: itl
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:,:,:),    pointer :: temperature
    real (kind=real_kind), dimension(:,:,:,:,:),    pointer :: dp3d
    real (kind=real_kind), dimension(:,:,:,:,:,:),  pointer :: velocity
    integer :: ie,k
    real (kind=real_kind), dimension(np,np,  nlev,nelemd) :: ttens,dptens
    real (kind=real_kind), dimension(np,np,2,nlev,nelemd) :: vtens
    real (kind=real_kind), dimension(nc,nc,2,nlev,nelemd) :: dpflux ! Unused, really, but it is in biharmonic_wk_dp3d signature

    call c_f_pointer(temperature_ptr, temperature, [nlev,np,np,  timelevels,nelemd])
    call c_f_pointer(dp3d_ptr,        dp3d,        [nlev,np,np,  timelevels,nelemd])
    call c_f_pointer(velocity_ptr,    velocity,    [nlev,np,np,2,timelevels,nelemd])

    ! Set states in the elements
    do ie=1,nelemd
      do k=1,nlev
        elem(ie)%state%v(:,:,:,k,itl)  = velocity(k,:,:,:,itl,ie)
        elem(ie)%state%T(:,:,k,itl)    = temperature(k,:,:,itl,ie)
        elem(ie)%state%dp3d(:,:,k,itl) = dp3d(k,:,:,itl,ie)
      enddo
    enddo

    ! Call f90 biharmonic_wk_dp3d subroutine
    call biharmonic_wk_dp3d(elem,dptens,dpflux,ttens,vtens,deriv,edge,hybrid,itl,1,nelemd)

    ! Overwrite input arrays with results
    do ie=1,nelemd
      do k=1,nlev
        velocity(k,:,:,:,itl,ie)  = vtens(:,:,:,k,ie)
        temperature(k,:,:,itl,ie) = ttens(:,:,k,ie)
        dp3d(k,:,:,itl,ie)        = dptens(:,:,k,ie)
      enddo
    enddo

  end subroutine hyperviscosity_test_f90

  subroutine cleanup_f90 () bind(c)
    use schedtype_mod, only : Schedule
    use edge_mod_base, only : FreeEdgeBuffer

    call FreeEdgeBuffer(edge)

    deallocate(elem)
    deallocate(GridVertex)
    deallocate(GridEdge)
    deallocate(Schedule)
  end subroutine cleanup_f90

end module hyperviscosity_functor_ut
