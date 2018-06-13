#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module checksum_mod
  use edgetype_mod, only : ghostbuffer3D_t, edgebuffer_t, ghostbuffer3d_t, ghostbufferTR_t
  use kinds, only : real_kind
  use dimensions_mod, only : np, nlev, nelem, nelemd, max_corner_elem, nc


  implicit none

  private
  save

  type (ghostBuffer3D_t)   :: ghostbuf,ghostbuf_cv
  type (ghostBuffer3d_t) :: ghostbuf3d
  type (ghostBufferTR_t) :: ghostbuf_tr
  type (edgeBuffer_t)    :: edge1

  public  :: testchecksum
  private :: genchecksum
  private :: ghostbuf,ghostbuf_cv,ghostbuf3d

contains
  !===============================================================================
  subroutine testchecksum(elem, par,GridEdge)
    use element_mod, only : element_t
    use parallel_mod, only : parallel_t, iam, syncmp
    use gridgraph_mod, only : gridedge_t,printchecksum
    use edge_mod, only : initedgebuffer, edgevpack, edgevunpack, edgedgvpack, &
         edgedgvunpack
    use edgetype_mod, only : edgebuffer_t
    use bndry_mod, only : bndry_exchangev
    use schedtype_mod, only : schedule_t, schedule
    use schedule_mod, only : checkschedule

    implicit none
    type (element_t), intent(in) :: elem(:)
    !================Arguments===============================
    type (parallel_t),   intent(in)           :: par
    type (GridEdge_t),   intent(in),target    :: GridEdge(:)


    !==============Local Allocatables==========================================
    real (kind=real_kind),allocatable      :: TestPattern_g(:,:,:,:), &
         Checksum_g(:,:,:,:),    &
         TestPattern_l(:,:,:,:), &
         Checksum_l(:,:,:,:), &
         Checksum_dg_l(:,:,:,:)
    type (EdgeBuffer_t)                         :: buffer

    !==============Local Temporaries===========================================
    type (Schedule_t),   pointer                :: pSchedule
    logical                                     :: ChecksumError
    integer                                     :: ix,iy
    integer                                     :: numlev
    integer                                     :: il,ig,ie,k
    integer                                     :: i
    integer                                     :: kptr,ielem
    integer                                     :: nSendCycles,nRecvCycles
    !===================Local Parameters=======================================
    logical, parameter                          :: PrChecksum = .FALSE.
    logical, parameter                          :: VerboseTiming=.FALSE.
    !==========================================================================


    allocate(TestPattern_g(np,np,nlev,nelem))
    allocate(Checksum_g(np,np,nlev,nelem))

    call genchecksum(TestPattern_g,Checksum_g,GridEdge)


    ! Setup the pointer to proper Schedule
    pSchedule => Schedule(1)


    allocate(TestPattern_l(np,np,nlev,nelemd))
    allocate(   Checksum_l(np,np,nlev,nelemd))
    allocate(Checksum_dg_l(0:np+1,0:np+1,nlev,nelemd))

    do il=1,nelemd
       Checksum_l(:,:,:,il) = 0.d0
       Checksum_dg_l(:,:,:,il) = 0.d0
       ig = pSchedule%Local2Global(il)
       TestPattern_l(:,:,:,il) = TestPattern_g(:,:,:,ig)
       Checksum_l(:,:,:,il)    = TestPattern_l(:,:,:,il)
    enddo


    if(PrChecksum) then
       if (par%masterproc) print *,'testchecksum:  The GLOBAL pattern is: '
       call PrintCheckSum(TestPattern_g,Checksum_g)
    endif

    if(PrChecksum) then
       if (par%masterproc) print *,'testchecksum:  The LOCAL pattern is: '
       call PrintCheckSum(TestPattern_l,Checksum_l)
    endif

    nSendCycles = pSchedule%nSendCycles
    nRecvCycles = pSchedule%nRecvCycles

    !=======================================
    !  Allocate the communication Buffers
    !=======================================

    call initEdgeBuffer(par,buffer,elem,nlev)


    !=======================================
    !  Synch everybody up
    !=======================================

    call syncmp(par)

    !==================================================
    !   Pack up the communication buffer
    !==================================================

    do ie=1,nelemd
       kptr=0
       numlev=nlev
       call edgeVpack(buffer,TestPattern_l(1,1,1,ie),numlev,kptr,ie)
    enddo
    if (par%masterproc) print *,'testchecksum: after call to edgeVpack'

    !==================================================
    !  Perform the boundary exchange
    !==================================================

    call bndry_exchangeV(par,buffer)

    !==================================================
    !   UnPack and accumulate the communication buffer
    !==================================================

    do ie=1,nelemd
       kptr   = 0
       numlev = nlev
       call edgeVunpack(buffer,Checksum_l(1,1,1,ie),numlev,kptr,ie)
    enddo
    if (par%masterproc) print *,'testchecksum: after call to edgeVunpack'

    !=====================================
    !  Printout the distributed checksum
    !=====================================

    if(PrChecksum) then
       call PrintCheckSum(TestPattern_l,Checksum_l)
    endif

    !============================================================
    !  Verify the distributed checksum against the serial version
    !============================================================
    do i=1,nelemd
       ig       = pSchedule%Local2Global(i)
       ChecksumError=.FALSE.
       do k=1,nlev
          do iy=1,np
             do ix=1,np
                if(Checksum_l(ix,iy,k,i) .ne. Checksum_g(ix,iy,k,ig)) then
                   write(*,100) iam , INT(TestPattern_l(ix,iy,k,i)),   &
                        INT(Checksum_g(ix,iy,k,ig)), INT(Checksum_l(ix,iy,k,i))
                   ChecksumError=.TRUE.
                endif
             enddo
          enddo
       enddo
       if(PrChecksum) then
          if(.NOT. ChecksumError) print *, 'IAM: ',iam,'testchecksum: Element', &
               pSchedule%Local2Global(i),'Verified'
       endif
    enddo
    ! =========================================
    ! Perform checksum for DG boundary exchange
    ! =========================================

    !==================================================
    !   Pack up the communication buffer
    !==================================================

    if (par%masterproc) print *,'testchecksum: before call to edgeVpack'
    do ielem=1,nelemd
       kptr=0
       numlev=nlev
       call edgeDGVpack(buffer,Checksum_l(1,1,1,ielem),numlev,kptr,ielem)
    enddo
    if (par%masterproc) print *,'testchecksum: after call to edgeVpack'

    !==================================================
    !  Perform the boundary exchange
    !==================================================

    call bndry_exchangeV(par,buffer)

    !==================================================
    !   UnPack and accumulate the communication buffer
    !==================================================

    if (par%masterproc) print *,'testchecksum: before call to edgeVunpack'
    do ielem=1,nelemd
       kptr   = 0
       numlev = nlev
       call edgeDGVunpack(buffer,Checksum_dg_l(0,0,1,ielem),numlev,kptr,ielem)
    enddo
    if (par%masterproc) print *,'testchecksum: after call to edgeDGVunpack'

    !==================================================
    !   Check Correctness for DG communication
    !==================================================
    do ie=1,nelemd
       ig       = pSchedule%Local2Global(ie)
       ChecksumError=.FALSE.
       do k=1,nlev
          do i=1,np

             ! =========================
             ! Check South Flux
             ! =========================
             if(Checksum_l(i,1,k,ie) .ne. Checksum_dg_l(i,0,k,ie)) then
                write(*,100) iam , INT(TestPattern_l(i,1,k,ie)), &
                     INT(Checksum_g(i,1,k,ig)), INT(Checksum_l(i,1,k,ie))
                ChecksumError=.TRUE.
             endif

             ! =========================
             ! Check East Flux
             ! =========================
             if(Checksum_l(np,i,k,ie) .ne. Checksum_dg_l(np+1,i,k,ie)) then
                write(*,100) iam , INT(TestPattern_l(np,i,k,ie)), &
                     INT(Checksum_g(np,i,k,ig)), INT(Checksum_l(np,i,k,ie))
                ChecksumError=.TRUE.
             endif

             ! =========================
             ! Check North Flux
             ! =========================
             if(Checksum_l(i,np,k,ie) .ne. Checksum_dg_l(i,np+1,k,ie)) then
                write(*,100) iam , INT(TestPattern_l(i,np,k,ie)), &
                     INT(Checksum_g(i,np,k,ig)), INT(Checksum_l(i,np,k,ie))
                ChecksumError=.TRUE.
             endif

             ! =========================
             ! Check West Flux
             ! =========================
             if(Checksum_l(1,i,k,ie) .ne. Checksum_dg_l(0,i,k,ie)) then
                write(*,100) iam , INT(TestPattern_l(1,i,k,ie)), &
                     INT(Checksum_g(1,i,k,ig)), INT(Checksum_l(1,i,k,ie))
                ChecksumError=.TRUE.
             endif

          enddo
       enddo
       if(PrChecksum) then
          if(.NOT. ChecksumError) print *, 'IAM: ',iam,'testchecksum: Element', &
               pSchedule%Local2Global(ie),'Verified'
       endif
    enddo




    !mem  print *,'testchecksum: before call to deallocate'
    deallocate(buffer%buf)
    deallocate(TestPattern_g)
    deallocate(Checksum_g)
    deallocate(TestPattern_l)
    deallocate(Checksum_l)
    deallocate(Checksum_dg_l)

    call CheckSchedule()


100 format('IAM:',i3,'testchecksum: Error with checksum:',I10,'--->',I10,' !=',I10)

  end subroutine testchecksum

  subroutine genchecksum(TestField,Checksum,GridEdge)
    use gridgraph_mod, only : gridedge_t, edgeindex_t

    implicit none

    type (GridEdge_t), intent(in),target   :: GridEdge(:)
    real(kind=real_kind), intent(inout),target  :: TestField(:,:,:,:)
    real(kind=real_kind), intent(inout),target  :: Checksum(:,:,:,:)

    type (EdgeIndex_t), pointer :: sIndex,dIndex
    type (GridEdge_t), pointer  :: gEdge

    integer                     :: i,ix,iy,k,iedge
    integer                     :: nelem_edge,nwords
    integer                     :: idest,isrc
    logical,parameter           :: PrChecksum =.FALSE.
#ifdef TESTGRID
    nelem_edge = SIZE(GridEdge)

    !  Setup the test pattern
    do i=1,nelem
       do k=1,nlev
          do iy=1,np
             do ix=1,np
                TestField(ix,iy,k,i)= ix + (iy-1)*np + 100*i + 1000000*(k-1)
             enddo
          enddo
       enddo
    enddo

    !  Initalize the checksum array to be the test pattern
    Checksum=TestField

    !  Calculate the checksum
    do iedge = 1,nelem_edge
       gEdge  => GridEdge(iedge)
       isrc   =  gEdge%tail%number
       idest  =  gEdge%head%number
       nwords =  gEdge%wgtV
       sIndex => gEdge%TailIndex
       dIndex => gEdge%HeadIndex


       do k=1,nlev
          do i=1,nwords
             Checksum(GridEdge(iedge)%HeadIndex%ixV(i),GridEdge(iedge)%HeadIndex%iyV(i),k,idest) = &
                  Checksum(GridEdge(iedge)%HeadIndex%ixV(i),GridEdge(iedge)%HeadIndex%iyV(i),k,idest) +  &
                  TestField(GridEdge(iedge)%TailIndex%ixV(i),GridEdge(iedge)%TailIndex%iyV(i),k,isrc)
          enddo
       enddo

    enddo

#if 0
    if(PrChecksum .AND. par%masterproc) then
       call PrintChecksum(TestField,Checksum)
    endif
#endif

100 format(1x,'element=',I2)
110 format(1x,I4,2x,'checksum = ',I5)
#endif
  end subroutine genchecksum

end module checksum_mod
