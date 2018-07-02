
module edgetype_mod 

  use kinds, only : int_kind, log_kind, real_kind
  use coordinate_systems_mod, only : cartesian3D_t
  use dimensions_mod, only : max_elements_attached_to_node

  implicit none 
  private 
  save 
   
  integer, public :: initedgebuffer_callid = 0

  type, public :: rotation_t
     integer  :: nbr                                        ! nbr direction: north south east west
     integer  :: reverse                                    ! 0 = do not reverse order
     ! 1 = reverse order
     real (kind=real_kind), dimension(:,:,:), pointer :: R => null()  !  rotation matrix
  end type rotation_t

  type, public :: EdgeDescriptor_t
     integer(kind=int_kind)  :: use_rotation
     integer(kind=int_kind)  :: padding
     integer(kind=int_kind)  :: putmapP(4*max_elements_attached_to_node-8)
     integer(kind=int_kind)  :: getmapP(4*max_elements_attached_to_node-8)
     integer(kind=int_kind)  :: putmapP_ghost(4*max_elements_attached_to_node-8)
     integer(kind=int_kind)  :: getmapP_ghost(4*max_elements_attached_to_node-8)
     integer(kind=int_kind)  :: putmapS(4*max_elements_attached_to_node-8)
     integer(kind=int_kind)  :: getmapS(4*max_elements_attached_to_node-8)
     integer(kind=int_kind)  :: globalID(4*max_elements_attached_to_node-8)
     integer(kind=int_kind)  :: loc2buf(4*max_elements_attached_to_node-8)
     type (cartesian3D_t)  , pointer  :: neigh_corners(:,:) => null()
     integer                          :: actual_neigh_edges
     logical(kind=log_kind)  :: reverse(4*max_elements_attached_to_node-8)
     type (rotation_t), dimension(:), pointer :: rot => null() !  Identifies list of edges
     !  that must be rotated, and how
  end type EdgeDescriptor_t

  type, public :: EdgeBuffer_t
     real (kind=real_kind), dimension(:), allocatable :: buf
     real (kind=real_kind), dimension(:), allocatable :: receive
     integer(kind=int_kind), pointer :: putmap(:,:) => null()
     integer(kind=int_kind), pointer :: getmap(:,:) => null()
     logical(kind=log_kind), pointer :: reverse(:,:) => null()
     integer(kind=int_kind), pointer :: moveLength(:) => null()
     integer(kind=int_kind), pointer :: movePtr(:) => null()
     integer(kind=int_kind), dimension(:), allocatable :: Rrequest,Srequest
     integer(kind=int_kind), dimension(:,:), allocatable :: status
     integer :: nlyr ! Number of layers
     integer :: nbuf ! total size of message passing buffer, includes vertical levels
     integer :: id
     integer :: tag
  end type EdgeBuffer_t

  type, public :: LongEdgeBuffer_t
     integer :: nlyr
     integer :: nbuf
     integer (kind=int_kind), dimension(:,:), pointer :: buf => null()
     integer (kind=int_kind), dimension(:,:), pointer :: receive => null()
  end type LongEdgeBuffer_t


  type, public :: GhostBufferTR_t
     real (kind=real_kind), dimension(:,:,:,:,:), pointer :: buf => null()
     real (kind=real_kind), dimension(:,:,:,:,:), pointer :: receive => null()
     integer :: nlyr ! Number of layers
     integer :: nbuf ! size of the horizontal dimension of the buffers.
  end type GhostBufferTR_t

  type, public :: GhostBuffer3D_t
     real (kind=real_kind), dimension(:,:,:,:), pointer :: buf => null()
     real (kind=real_kind), dimension(:,:,:,:), pointer :: receive => null()
     integer :: nlyr ! Number of layers
     integer :: nhc  ! Number of layers of ghost cells
     integer :: np   ! Number of points in a cell
     integer :: nbuf ! size of the horizontal dimension of the buffers.
     integer :: elem_size ! size of 2D array (first two dimensions of buf())
  end type GhostBuffer3D_t

 
end module edgetype_mod
