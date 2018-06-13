#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "omp_config.h"
#define NEWEULER_B4B 1
#define OVERLAP 1

!SUBROUTINES:
!   prim_advec_tracers_remap_rk2()
!      SEM 2D RK2 + monotone remap + hyper viscosity
!      SEM 2D RK2 can use sign-preserving or monotone reconstruction
!
!For RK2 advection of Q:  (example of 2 stage RK for tracers):   dtq = qsplit*dt
!For consistency, if Q=1
!  dp1  = dp(t)- dtq div[ U1 dp(t)]
!  dp2  = dp1  - dtq div[ U2 dp1  ]  + 2*dtq D( dpdiss_ave )
!  dp*  = (dp(t) + dp2 )/2
!       =  dp(t) - dtq  div[ U1 dp(t) + U2 dp1 ]/2   + dtq D( dpdiss_ave )
!
!so we require:
!  U1 = Udp_ave / dp(t)
!  U2 = Udp_ave / dp1
!
!For tracer advection:
!  Qdp1  = Qdp(t)- dtq div[ U1 Qdp(t)]
!  Qdp2  = Qdp1  - dtq div[ U2 Qdp1  ]  + 2*dtq D( Q dpdiss_ave )
!  Qdp*  = (Qdp(t) + Qdp2 )/2
!       =  Qdp(t) - dtq  div[ U1 Qdp(t) + U2 Qdp1 ]   + dtq D( Q dpdiss_ave )
!
!Qdp1:  limit Q, with Q = Qdp1-before-DSS/(dp1-before-DSS)      with dp1 as computed above
!Qdp2:  limit Q, with Q = Qdp2-before-DSS/(dp2-before-DSS)      with dp2 as computed above
!
!For dissipation: Q = Qdp1-after-DSS / dp1-after-DSS
!
!
!last step:
!  remap Qdp* to Qdp(t+1)   [ dp_star(t+1) -> dp(t+1) ]


module prim_advection_mod_base
!
! two formulations.  both are conservative
! u grad Q formulation:
!
!    d/dt[ Q] +  U grad Q  +  eta_dot dp/dn dQ/dp  = 0
!                            ( eta_dot dQ/dn )
!
!    d/dt[ dp/dn ] = div( dp/dn U ) + d/dn ( eta_dot dp/dn )
!
! total divergence formulation:
!    d/dt[dp/dn Q] +  div( U dp/dn Q ) + d/dn ( eta_dot dp/dn Q ) = 0
!
! for convience, rewrite this as dp Q:  (since dn does not depend on time or the horizonal):
! equation is now:
!    d/dt[dp Q] +  div( U dp Q ) + d( eta_dot_dpdn Q ) = 0
!
!
  use kinds, only              : real_kind
  use dimensions_mod, only     : nlev, nlevp, np, qsize, ntrac, nc
  use physical_constants, only : rgas, Rwater_vapor, kappa, g, rearth, rrearth, cp
  use derivative_mod, only     : gradient, vorticity, gradient_wk, derivative_t, divergence, &
                                 gradient_sphere, divergence_sphere
  use element_mod, only        : element_t
  use filter_mod, only         : filter_t, filter_P
  use hybvcoord_mod, only      : hvcoord_t
  use time_mod, only           : TimeLevel_t, smooth, TimeLevel_Qdp
  use prim_si_mod, only        : preq_pressure
  use diffusion_mod, only      : scalar_diffusion, diffusion_init
  use control_mod, only        : integration, test_case, filter_freq_advection,  hypervis_order, &
        statefreq, moisture, TRACERADV_TOTAL_DIVERGENCE, TRACERADV_UGRADQ, &
        nu_p, nu_q, limiter_option, hypervis_subcycle_q, rsplit
  use edge_mod, only           : edgevpack, edgerotate, edgevunpack, initedgebuffer, initedgesbuffer, &
        edgevunpackmin, initghostbuffer3D

  use edgetype_mod, only       : EdgeDescriptor_t, EdgeBuffer_t, ghostbuffer3D_t
  use hybrid_mod, only         : hybrid_t
  use bndry_mod, only          : bndry_exchangev
  use viscosity_mod, only      : biharmonic_wk_scalar, biharmonic_wk_scalar_minmax, neighbor_minmax, &
                                 neighbor_minmax_start, neighbor_minmax_finish
  use perf_mod, only           : t_startf, t_stopf, t_barrierf ! _EXTERNAL
  use parallel_mod, only   : abortmp

  implicit none

  private
  save

  public :: Prim_Advec_Init1, Prim_Advec_Init2, prim_advec_init_deriv
#ifndef USE_KOKKOS_KERNELS
  public :: vertical_remap_interface
  public :: Prim_Advec_Tracers_remap, Prim_Advec_Tracers_remap_rk2
#endif

  type (EdgeBuffer_t)      :: edgeAdv, edgeAdvp1, edgeAdvQminmax, edgeAdv1,  edgeveloc
  type (ghostBuffer3D_t)   :: ghostbuf_tr

  integer,parameter :: DSSeta = 1
  integer,parameter :: DSSomega = 2
  integer,parameter :: DSSdiv_vdp_ave = 3
  integer,parameter :: DSSno_var = -1

  real(kind=real_kind), allocatable, target :: qmin(:,:,:), qmax(:,:,:)

  type (derivative_t), public, allocatable   :: deriv(:) ! derivative struct (nthreads)

!  interface
!    subroutine euler_pull_qmin_qmax_c(qmin_ptr, qmax_ptr) bind(c)
!      use iso_c_binding, only : c_ptr
!      !
!      ! Inputs
!      !
!      type (c_ptr), intent(in) :: qmin_ptr, qmax_ptr
!    end subroutine euler_pull_qmin_qmax_c
!    subroutine euler_pull_data_c(elem_derived_eta_dot_dpdn_ptr, elem_derived_omega_p_ptr, &
!         elem_derived_divdp_proj_ptr, elem_derived_vn0_ptr, elem_derived_dp_ptr, &
!         elem_derived_divdp_ptr, elem_derived_dpdiss_biharmonic_ptr, &
!         elem_state_Qdp_ptr, elem_derived_dpdiss_ave_ptr) bind(c)
!      use iso_c_binding, only : c_ptr
!      !
!      ! Inputs
!      !
!      type (c_ptr), intent(in) :: elem_derived_eta_dot_dpdn_ptr, elem_derived_omega_p_ptr, &
!           elem_derived_divdp_proj_ptr, elem_derived_vn0_ptr, elem_derived_dp_ptr, &
!           elem_derived_divdp_ptr, elem_derived_dpdiss_biharmonic_ptr, &
!           elem_state_Qdp_ptr, elem_derived_dpdiss_ave_ptr
!    end subroutine euler_pull_data_c
!    subroutine euler_push_results_c(elem_derived_eta_dot_dpdn_ptr, elem_derived_omega_p_ptr, &
!         elem_derived_divdp_proj_ptr, elem_state_Qdp_ptr, qmin_ptr, qmax_ptr) bind(c)
!      use iso_c_binding, only : c_ptr
!      !
!      ! Inputs
!      !
!      type (c_ptr), intent(in) :: elem_derived_eta_dot_dpdn_ptr, elem_derived_omega_p_ptr, &
!           elem_derived_divdp_proj_ptr, elem_state_Qdp_ptr, qmin_ptr, qmax_ptr
!    end subroutine euler_push_results_c
!    subroutine euler_exchange_qdp_dss_var_c() bind(c)
!    end subroutine euler_exchange_qdp_dss_var_c
!  end interface

contains

#ifndef USE_KOKKOS_KERNELS
  subroutine advance_qdp_f90 (nets,nete, &
       rhs_multiplier,DSSopt,dp,dpdissk, &
       n0_qdp,dt,Vstar,elem,deriv,Qtens, &
       rhs_viss,Qtens_biharmonic,np1_qdp)
    use kinds,          only : real_kind
    use derivative_mod, only : derivative_t, limiter_optim_iter_full, limiter_clip_and_sum
    use element_mod,    only : element_t
    !
    ! Inputs
    !
    real (kind=real_kind), intent(inout), dimension(np,np,2,nlev,nets:nete) :: Vstar
    real (kind=real_kind), intent(out), dimension(np,np,nlev,qsize,nets:nete) :: Qtens
    real (kind=real_kind), intent(in), dimension(np,np,nlev,qsize,nets:nete) :: Qtens_biharmonic
    real (kind=real_kind), intent(inout), dimension(np,np,nlev,nets:nete) :: dpdissk
    type (element_t),      intent(inout), dimension(:), target :: elem
    type (derivative_t),   intent(in) :: deriv
    real (kind=real_kind), intent(in) :: dt
    integer, intent(in) :: nets, nete, n0_qdp, rhs_viss, np1_qdp, rhs_multiplier, DSSopt
    !
    ! Locals
    !
    real (kind=real_kind), dimension(np,np,2) :: gradQ
    real (kind=real_kind), dimension(np,np)   :: dp_star
    real (kind=real_kind), dimension(np,np,nlev) :: dp ! Could do same as dp_start, but Fortran code doesn't
    real (kind=real_kind), dimension(:,:,:), pointer :: DSSvar
    integer :: ie, k, q
    !
    ! Routine body
    !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !   2D Advection step
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    do ie = nets , nete
       ! note: eta_dot_dpdn is actually dimension nlev+1, but nlev+1 data is
       ! all zero so we only have to DSS 1:nlev
       if ( DSSopt == DSSeta         ) DSSvar => elem(ie)%derived%eta_dot_dpdn(:,:,:)
       if ( DSSopt == DSSomega       ) DSSvar => elem(ie)%derived%omega_p(:,:,:)
       if ( DSSopt == DSSdiv_vdp_ave ) DSSvar => elem(ie)%derived%divdp_proj(:,:,:)

       ! Compute velocity used to advance Qdp
#if (defined COLUMN_OPENMP)
       !$omp parallel do private(k,q)
#endif
       do k = 1 , nlev    !  Loop index added (AAM)
          ! derived variable divdp_proj() (DSS'd version of divdp) will only be correct on 2nd and 3rd stage
          ! but that's ok because rhs_multiplier=0 on the first stage:
          dp(:,:,k) = elem(ie)%derived%dp(:,:,k) - rhs_multiplier * dt * elem(ie)%derived%divdp_proj(:,:,k)
          Vstar(:,:,1,k,ie) = elem(ie)%derived%vn0(:,:,1,k) / dp(:,:,k)
          Vstar(:,:,2,k,ie) = elem(ie)%derived%vn0(:,:,2,k) / dp(:,:,k)

          if (( limiter_option == 8).or.( limiter_option == 9 )) then
             ! Note that the term dpdissk is independent of Q
             ! UN-DSS'ed dp at timelevel n0+1:
             dpdissk(:,:,k,ie) = dp(:,:,k) - dt * elem(ie)%derived%divdp(:,:,k)
             if ( nu_p > 0 .and. rhs_viss /= 0 ) then
                ! add contribution from UN-DSS'ed PS dissipation
                !          dpdiss(:,:) = ( hvcoord%hybi(k+1) - hvcoord%hybi(k) ) *
                !          elem(ie)%derived%psdiss_biharmonic(:,:)
                dpdissk(:,:,k,ie) = dpdissk(:,:,k,ie) - rhs_viss * dt * nu_q &
                     * elem(ie)%derived%dpdiss_biharmonic(:,:,k) / elem(ie)%spheremp(:,:)
             endif
             ! IMPOSE ZERO THRESHOLD.  do this here so it can be turned off for
             ! testing
             do q=1,qsize
                qmin(k,q,ie)=max(qmin(k,q,ie),0d0)
             enddo
          endif  ! limiter == 8

          ! also DSS extra field
          DSSvar(:,:,k) = elem(ie)%spheremp(:,:) * DSSvar(:,:,k)
       enddo
       call edgeVpack( edgeAdvp1 , DSSvar(:,:,1:nlev) , nlev , nlev*qsize , ie)
    enddo

    do ie=nets,nete
#if (defined COLUMN_OPENMP)
 !$omp parallel do private(q,k,gradQ,dp_star)
#endif
      do q = 1 , qsize
        do k = 1 , nlev  !  dp_star used as temporary instead of divdp (AAM)
          ! div( U dp Q),
          gradQ(:,:,1) = Vstar(:,:,1,k,ie) * elem(ie)%state%Qdp(:,:,k,q,n0_qdp)
          gradQ(:,:,2) = Vstar(:,:,2,k,ie) * elem(ie)%state%Qdp(:,:,k,q,n0_qdp)
          dp_star(:,:) = divergence_sphere( gradQ , deriv , elem(ie) )
          Qtens(:,:,k,q,ie) = elem(ie)%state%Qdp(:,:,k,q,n0_qdp) - dt * dp_star(:,:)
        enddo
      enddo
   enddo

   do ie=nets,nete
#if (defined COLUMN_OPENMP)
      !$omp parallel do private(q,k,dp_star)
#endif
      do q=1,qsize
         do k=1,nlev
            ! optionally add in hyperviscosity computed above:
            if ( rhs_viss /= 0 ) Qtens(:,:,k,q,ie) = Qtens(:,:,k,q,ie) + Qtens_biharmonic(:,:,k,q,ie)
         enddo

         if ( limiter_option == 8) then
            ! apply limiter to Q = Qtens / dp_star
            !call t_startf('lim8')
            call limiter_optim_iter_full( Qtens(:,:,:,q,ie), elem(ie)%spheremp(:,:), &
                                          qmin(:,q,ie), qmax(:,q,ie), dpdissk(:,:,:,ie) )
            !call t_stopf('lim8')
         elseif ( limiter_option == 9 ) then
            !call t_startf('lim9')
            call limiter_clip_and_sum(    Qtens(:,:,:,q,ie), elem(ie)%spheremp(:,:), &
                                          qmin(:,q,ie), qmax(:,q,ie), dpdissk(:,:,:,ie) )
            !call t_stopf('lim9')
         endif

         ! apply mass matrix, overwrite np1 with solution:
         ! dont do this earlier, since we allow np1_qdp == n0_qdp
         ! and we dont want to overwrite n0_qdp until we are done using it
         do k = 1 , nlev
            elem(ie)%state%Qdp(:,:,k,q,np1_qdp) = elem(ie)%spheremp(:,:) * Qtens(:,:,k,q,ie)
         enddo

         if ( limiter_option == 4 ) then
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            ! sign-preserving limiter, applied after mass matrix
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            call limiter2d_zero( elem(ie)%state%Qdp(:,:,:,q,np1_qdp))
         endif

         call edgeVpack(edgeAdvp1 , elem(ie)%state%Qdp(:,:,:,q,np1_qdp) , nlev , nlev*(q-1) , ie )
      enddo
   enddo ! ie loop

  end subroutine advance_qdp_f90
#endif

  subroutine Prim_Advec_Init1(par, elem, n_domains)
    use dimensions_mod, only : nlev, qsize, nelemd
    use parallel_mod, only : parallel_t
    use control_mod, only : use_semi_lagrange_transport
    use interpolate_mod,        only : interpolate_tracers_init
    type(parallel_t) :: par
    integer, intent(in) :: n_domains
    type (element_t) :: elem(:)
    type (EdgeDescriptor_t), allocatable :: desc(:)


    integer :: ie
    ! Shared buffer pointers.
    ! Using "=> null()" in a subroutine is usually bad, because it makes
    ! the variable have an implicit "save", and therefore shared between
    ! threads. But in this case we want shared pointers.
    real(kind=real_kind), pointer :: buf_ptr(:) => null()
    real(kind=real_kind), pointer :: receive_ptr(:) => null()


    ! this might be called with qsize=0
    ! allocate largest one first
    ! Currently this is never freed. If it was, only this first one should
    ! be freed, as only it knows the true size of the buffer.
    call initEdgeBuffer(par,edgeAdvp1,elem,qsize*nlev + nlev)
    call initEdgeBuffer(par,edgeAdv,elem,qsize*nlev)
    call initEdgeBuffer(par,edgeAdv1,elem,nlev)
    call initEdgeBuffer(par,edgeveloc,elem,2*nlev)

    ! This is a different type of buffer pointer allocation
    ! used for determine the minimum and maximum value from
    ! neighboring  elements
    call initEdgeSBuffer(par,edgeAdvQminmax,elem,qsize*nlev*2)

    ! Don't actually want these saved, if this is ever called twice.
    nullify(buf_ptr)
    nullify(receive_ptr)

    allocate(deriv(0:n_domains-1))

#ifndef USE_KOKKOS_KERNELS
    ! this static array is shared by all threads, so dimension for all threads (nelemd), not nets:nete:
    allocate (qmin(nlev,qsize,nelemd))
    allocate (qmax(nlev,qsize,nelemd))
#endif

    if  (use_semi_lagrange_transport) then
       call initghostbuffer3D(ghostbuf_tr,nlev*qsize,np)
       call interpolate_tracers_init()
    endif

  end subroutine Prim_Advec_Init1

  subroutine Prim_Advec_Init_deriv(hybrid)

    use kinds,          only : longdouble_kind
    use dimensions_mod, only : nc
    use derivative_mod, only : derivinit
    implicit none
    type (hybrid_t), intent(in) :: hybrid

    ! ==================================
    ! Initialize derivative structure
    ! ==================================
    call derivinit(deriv(hybrid%ithr))
  end subroutine Prim_Advec_Init_deriv

  subroutine Prim_Advec_Init2(elem,hvcoord,hybrid)
    use element_mod   , only : element_t
    use hybvcoord_mod , only : hvcoord_t
    implicit none
    type(element_t)   , intent(in) :: elem(:)
    type(hvcoord_t)   , intent(in) :: hvcoord
    type (hybrid_t)   , intent(in) :: hybrid
    !Nothing to do
  end subroutine Prim_Advec_Init2

!=================================================================================================!
#ifndef USE_KOKKOS_KERNELS
  subroutine Prim_Advec_Tracers_remap( elem , deriv , hvcoord , flt , hybrid , dt , tl , nets , nete )
    use control_mod   , only : use_semi_lagrange_transport
    implicit none
    type (element_t)     , intent(inout) :: elem(:)
    type (derivative_t)  , intent(in   ) :: deriv
    type (hvcoord_t)     , intent(in   ) :: hvcoord
    type (filter_t)      , intent(in   ) :: flt
    type (hybrid_t)      , intent(in   ) :: hybrid
    real(kind=real_kind) , intent(in   ) :: dt
    type (TimeLevel_t)   , intent(inout) :: tl
    integer              , intent(in   ) :: nets
    integer              , intent(in   ) :: nete


    call Prim_Advec_Tracers_remap_rk2( elem , deriv , hvcoord , flt , hybrid , dt , tl , nets , nete )
  end subroutine Prim_Advec_Tracers_remap

subroutine VDOT(rp,Que,rho,mass,hybrid,nets,nete)
  use parallel_mod,        only: global_shared_buf, global_shared_sum
  use global_norms_mod,    only: wrap_repro_sum

  implicit none
  integer             , intent(in)              :: nets
  integer             , intent(in)              :: nete
  real(kind=real_kind), intent(out)             :: rp                (nlev,qsize)
  real(kind=real_kind), intent(in)              :: Que         (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: rho         (np*np,nlev,      nets:nete)
  real(kind=real_kind), intent(in)              :: mass              (nlev,qsize)
  type (hybrid_t)     , intent(in)              :: hybrid

  integer                                       :: k,n,q,ie

  global_shared_buf = 0
  do ie=nets,nete
    n=0
    do q=1,qsize
    do k=1,nlev
      n=n+1
      global_shared_buf(ie,n) = global_shared_buf(ie,n) + DOT_PRODUCT(Que(:,k,q,ie), rho(:,k,ie))
    end do
    end do
  end do

  call wrap_repro_sum(nvars=n, comm=hybrid%par%comm)

  n=0
  do q=1,qsize
  do k=1,nlev
    n=n+1
    rp(k,q) = global_shared_sum(n) - mass(k,q)
  enddo
  enddo

end subroutine VDOT

subroutine Cobra_SLBQP(Que, Que_t, rho, minq, maxq, mass, hybrid, nets, nete)

  use parallel_mod,        only: global_shared_buf, global_shared_sum
  use global_norms_mod,    only: wrap_repro_sum

  implicit none
  integer             , intent(in)              :: nets
  integer             , intent(in)              :: nete
  real(kind=real_kind), intent(out)             :: Que         (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: Que_t       (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: rho         (np*np,nlev,      nets:nete)
  real(kind=real_kind), intent(in)              :: minq        (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: maxq        (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: mass              (nlev,qsize)
  type (hybrid_t)     , intent(in)              :: hybrid

  integer,                            parameter :: max_clip = 50
  real(kind=real_kind),               parameter :: eta = 1D-08
  real(kind=real_kind),               parameter :: hfd = 1D-10
  real(kind=real_kind)                          :: lambda_p          (nlev,qsize)
  real(kind=real_kind)                          :: lambda_c          (nlev,qsize)
  real(kind=real_kind)                          :: rp                (nlev,qsize)
  real(kind=real_kind)                          :: rc                (nlev,qsize)
  real(kind=real_kind)                          :: rd                (nlev,qsize)
  real(kind=real_kind)                          :: alpha             (nlev,qsize)
  integer                                       :: j,k,n,q,ie
  integer                                       :: nclip

  nclip = 0

  Que(:,:,:,:) = Que_t(:,:,:,:)

  Que = MIN(MAX(Que,minq),maxq)

  call VDOT(rp,Que,rho,mass,hybrid,nets,nete)
  nclip = nclip + 1

  if (MAXVAL(ABS(rp)).lt.eta) return

  do ie=nets,nete
  do q=1,qsize
  do k=1,nlev
     Que(:,k,q,ie) = hfd * rho(:,k,ie) + Que_t(:,k,q,ie)
  enddo
  enddo
  enddo

  Que = MIN(MAX(Que,minq),maxq)

  call VDOT(rc,Que,rho,mass,hybrid,nets,nete)

  rd = rc-rp
  if (MAXVAL(ABS(rd)).eq.0) return

  alpha = 0
  WHERE (rd.ne.0) alpha = hfd / rd

  lambda_p = 0
  lambda_c =  -alpha*rp

  do while (MAXVAL(ABS(rc)).gt.eta .and. nclip.lt.max_clip)

    do ie=nets,nete
    do q=1,qsize
    do k=1,nlev
       Que(:,k,q,ie) = (lambda_c(k,q) + hfd) * rho(:,k,ie) + Que_t(:,k,q,ie)
    enddo
    enddo
    enddo
    Que = MIN(MAX(Que,minq),maxq)

    call VDOT(rc,Que,rho,mass,hybrid,nets,nete)
    nclip = nclip + 1

    rd = rp-rc

    if (MAXVAL(ABS(rd)).eq.0) exit

    alpha = 0
    WHERE (rd.ne.0) alpha = (lambda_p - lambda_c) / rd

    rp       = rc
    lambda_p = lambda_c

    lambda_c = lambda_c -  alpha * rc

  enddo
end subroutine Cobra_SLBQP


subroutine Cobra_Elem(Que, Que_t, rho, minq, maxq, mass, hybrid, nets, nete)

  use parallel_mod,        only: global_shared_buf, global_shared_sum
  use global_norms_mod,    only: wrap_repro_sum

  implicit none
  integer             , intent(in)              :: nets
  integer             , intent(in)              :: nete
  real(kind=real_kind), intent(out)             :: Que         (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: Que_t       (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: rho         (np*np,nlev,      nets:nete)
  real(kind=real_kind), intent(in)              :: minq        (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: maxq        (np*np,nlev,qsize,nets:nete)
  real(kind=real_kind), intent(in)              :: mass              (nlev,qsize,nets:nete)
  type (hybrid_t)     , intent(in)              :: hybrid

  integer,                            parameter :: max_clip = 50
  real(kind=real_kind),               parameter :: eta = 1D-10
  real(kind=real_kind),               parameter :: hfd = 1D-08
  real(kind=real_kind)                          :: lambda_p          (nlev,qsize,nets:nete)
  real(kind=real_kind)                          :: lambda_c          (nlev,qsize,nets:nete)
  real(kind=real_kind)                          :: rp                (nlev,qsize,nets:nete)
  real(kind=real_kind)                          :: rc                (nlev,qsize,nets:nete)
  real(kind=real_kind)                          :: rd                (nlev,qsize,nets:nete)
  real(kind=real_kind)                          :: alpha             (nlev,qsize,nets:nete)
  integer                                       :: j,k,n,q,ie
  integer                                       :: nclip
  integer                                       :: mloc(3)

  nclip = 1

  Que(:,:,:,:) = Que_t(:,:,:,:)

  Que = MIN(MAX(Que,minq),maxq)

  do ie=nets,nete
  do q=1,qsize
  do k=1,nlev
    rp(k,q,ie) = DOT_PRODUCT(Que(:,k,q,ie), rho(:,k,ie)) - mass(k,q,ie)
  end do
  end do
  end do

  if (MAXVAL(ABS(rp)).lt.eta) return

  do ie=nets,nete
  do q=1,qsize
  do k=1,nlev
     Que(:,k,q,ie) = hfd * rho(:,k,ie) + Que_t(:,k,q,ie)
  enddo
  enddo
  enddo

  Que = MIN(MAX(Que,minq),maxq)

  do ie=nets,nete
  do q=1,qsize
  do k=1,nlev
    rc(k,q,ie) = DOT_PRODUCT(Que(:,k,q,ie), rho(:,k,ie)) - mass(k,q,ie)
  end do
  end do
  end do

  rd = rc-rp
  if (MAXVAL(ABS(rd)).eq.0) return

  alpha = 0
  WHERE (rd.ne.0) alpha = hfd / rd

  lambda_p = 0
  lambda_c =  -alpha*rp

! if (hybrid%par%masterproc) print *,__FILE__,__LINE__," mass(20,1,4):",mass(20,1,4)
! do k=1,np*np
!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," maxq(k,20,1,4):", &
!     maxq(k,20,1,4) ,minq(k,20,1,4),maxq(k,20,1,4)-minq(k,20,1,4)
! enddo
! do k=1,np*np
!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," Que(k,20,1,4):",Que(k,20,1,4) ,rho(k,20,4)
! enddo
  do while (MAXVAL(ABS(rc)).gt.eta .and. nclip.lt.max_clip)
    nclip = nclip + 1

    do ie=nets,nete
    do q=1,qsize
    do k=1,nlev
!      Que(:,k,q,ie) = (lambda_c(k,q,ie) + hfd) * rho(:,k,ie) + Que_t(:,k,q,ie)
       Que(:,k,q,ie) = lambda_c(k,q,ie) * rho(:,k,ie) + Que_t(:,k,q,ie)
    enddo
    enddo
    enddo

!   do ie=nets,nete
!   do q=1,qsize
!   do k=1,nlev
!     rc(k,q,ie) = DOT_PRODUCT(Que(:,k,q,ie), rho(:,k,ie)) - mass(k,q,ie)
!   end do
!   end do
!   end do
!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," rc(20,1,4):",rc(20,1,4), DOT_PRODUCT(Que(:,20,1,4), rho(:,20,4))

    Que = MIN(MAX(Que,minq),maxq)

    do ie=nets,nete
    do q=1,qsize
    do k=1,nlev
      rc(k,q,ie) = DOT_PRODUCT(Que(:,k,q,ie), rho(:,k,ie)) - mass(k,q,ie)
    end do
    end do
    end do


    mloc = MAXLOC(ABS(rc))
!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," MAXVAL(ABS(rc)):",MAXVAL(ABS(rc)), mloc, nclip

    rd = rp-rc

!   if (MAXVAL(ABS(rd)).eq.0) exit

    alpha = 0
    WHERE (rd.ne.0) alpha = (lambda_p - lambda_c) / rd
!   WHERE (alpha.eq.0.and.MAXVAL(ABS(rc)).gt.eta) alpha=10;

    rp       = rc
    lambda_p = lambda_c

    lambda_c = lambda_c -  alpha * rc

!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," rc(20,1,4):",rc(20,1,4)
!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," rd(20,1,4):",rd(20,1,4)
!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," lambda_p(20,1,4):",lambda_p(20,1,4)
!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," lambda_c(20,1,4):",lambda_c(20,1,4)
!   if (hybrid%par%masterproc) print *,__FILE__,__LINE__," alpha(20,1,4):",alpha(20,1,4)
!   if (hybrid%par%masterproc) print *
  enddo
! if (hybrid%par%masterproc) print *,__FILE__,__LINE__," MAXVAL(ABS(rc)):",MAXVAL(ABS(rc)),eta," nclip:",nclip
end subroutine Cobra_Elem

subroutine ALE_elems_with_dep_points (elem_indexes, dep_points, num_neighbors, ngh_corners)

  use element_mod,            only : element_t
  use dimensions_mod,         only : np
  use coordinate_systems_mod, only : cartesian3D_t, change_coordinates
  use interpolate_mod,        only : point_inside_quad

  implicit none

  ! The ngh_corners array is a list of corners of both elem and all of it's
  ! neighor elements all sorted by global id.
  integer              , intent(in)                :: num_neighbors
  type(cartesian3D_t),intent(in)                   :: ngh_corners(4,num_neighbors)
  integer              , intent(out)               :: elem_indexes(np,np)
  type(cartesian3D_t)  , intent(in)                :: dep_points(np,np)

  integer                                          :: i,j,n
  logical                                          :: inside

  elem_indexes = -1
  do i=1,np
    do j=1,np
! Just itererate the neighbors in global id order to get the same result on every processor.
      do n = 1, num_neighbors
! Mark Taylor's handy dandy point_inside_gc check.
       inside = point_inside_quad (ngh_corners(:,n), dep_points(i,j))
       if (inside) then
         elem_indexes(i,j) = n
         exit
       end if
     end do
    end do
  end do

  if (MINVAL(elem_indexes(:,:))==-1) then
    write (*,*) __FILE__,__LINE__,"Aborting because point not found in neighbor list. Info:"
    do i=1,np
      do j=1,np
        if (elem_indexes(i,j)==-1) then
          write (*,*)   " departure point ",dep_points(i,j)
          do n = 1, num_neighbors
            write (*,*) " quad checked    ",ngh_corners(1,n)
            write (*,*) "                 ",ngh_corners(2,n)
            write (*,*) "                 ",ngh_corners(3,n)
            write (*,*) "                 ",ngh_corners(4,n)
            write (*,*)
          end do
          exit
        end if
      end do
    end do
    call abortmp("ERROR elems_with_dep_points: Can't find departure grid. Time step too long?")
  end if
end subroutine ALE_elems_with_dep_points

function  shape_fcn_deriv(pc) result(dNds)
  real (kind=real_kind), intent(in)  ::  pc(2)
  real (kind=real_kind)              :: dNds(4,2)

  dNds(1, 1) = - 0.25 * (1.0 - pc(2))
  dNds(1, 2) = - 0.25 * (1.0 - pc(1))

  dNds(2, 1) =   0.25 * (1.0 - pc(2))
  dNds(2, 2) = - 0.25 * (1.0 + pc(1))

  dNds(3, 1) =   0.25 * (1.0 + pc(2))
  dNds(3, 2) =   0.25 * (1.0 + pc(1))

  dNds(4, 1) = - 0.25 * (1.0 + pc(2))
  dNds(4, 2) =   0.25 * (1.0 - pc(1))
end function

function inv_2x2(A) result(A_inv)
  real (kind=real_kind), intent(in)  :: A    (2,2)
  real (kind=real_kind)              :: A_inv(2,2)
  real (kind=real_kind) :: det, denom

  det = A(1,1) * A(2,2) - A(2,1) * A(1,2)
  denom = 1/det
  ! inverse:
  A_inv(1,1) =  denom * A(2,2)  !  dxidx
  A_inv(2,1) = -denom * A(2,1)  !  detadx
  A_inv(1,2) = -denom * A(1,2)  !  dxidy
  A_inv(2,2) =  denom * A(1,1)  !  detady
end function

function INV(dxds) result(dsdx)

  real (kind=real_kind), intent(in)  :: dxds(3,2)

  real (kind=real_kind)  ::     dsdx(2,3)
  real (kind=real_kind)  ::      ata(2,2)
  real (kind=real_kind)  ::  ata_inv(2,2)


  !     dxds = | dxdxi   dxdeta |
  !            | dydxi   dydeta |
  !            | dzdxi   dzdeta |
  ata  = MATMUL(TRANSPOSE(dxds), dxds)
  ata_inv = inv_2x2(ata)
  dsdx = MATMUL(ata_inv, TRANSPOSE(dxds))
  !     dsdx = |  dxidx   dxidy   dxidz |
  !            | detadx  detady  detadz |

end function

subroutine shape_fcn(N, pc)
  real (kind=real_kind), intent(out) :: N(4)
  real (kind=real_kind), intent(in)  :: pc(2)

  ! shape function for each node evaluated at param_coords
  N(1) = 0.25 * (1.0 - pc(1)) * (1.0 - pc(2))
  N(2) = 0.25 * (1.0 + pc(1)) * (1.0 - pc(2))
  N(3) = 0.25 * (1.0 + pc(1)) * (1.0 + pc(2))
  N(4) = 0.25 * (1.0 - pc(1)) * (1.0 + pc(2))
end subroutine


function F(coords, pc) result(x)
  real (kind=real_kind), intent(in) :: pc(2), coords(4,3)

  real (kind=real_kind)            :: N(4), x(3)
  call shape_fcn(N,pc)
  x = MATMUL(TRANSPOSE(coords), N)
  x = x/SQRT(DOT_PRODUCT(x,x))
end function

function  DF(coords, pc) result(dxds)
  real (kind=real_kind), intent(in)  :: coords(4,3)
  real (kind=real_kind), intent(in)  :: pc(2)

  real (kind=real_kind)              :: dxds(3,2)
  real (kind=real_kind)              :: dNds(4,2)
  real (kind=real_kind)              ::  dds(3,2)
  real (kind=real_kind)              ::    c(2)
  real (kind=real_kind)              ::    x(3)
  real (kind=real_kind)              ::   xc(3,2)
  real (kind=real_kind)              :: nx, nx2
  integer                            :: i,j

  dNds = shape_fcn_deriv  (pc)
  dds  = MATMUL(TRANSPOSE(coords), dNds)

  x    = F(coords, pc)
  nx2  = DOT_PRODUCT(x,x)
  nx   = SQRT(nx2)
  c    = MATMUL(TRANSPOSE(dds), x)
  do j=1,2
    do i=1,3
      xc(i,j) = x(i)*c(j)
    end do
  end do
  dxds = nx2*dds - xc
  dxds = dxds/(nx*nx2)
end function


function cartesian_parametric_coordinates(sphere, corners3D) result (ref)
  use coordinate_systems_mod, only : cartesian2d_t, cartesian3D_t, spherical_polar_t, spherical_to_cart
  implicit none
  type (spherical_polar_t), intent(in) :: sphere
  type (cartesian3D_t)    , intent(in) :: corners3D(4)  !x,y,z coords of element corners

  type (cartesian2D_t)                 :: ref

  integer,               parameter :: MAXIT = 20
  real (kind=real_kind), parameter :: TOL   = 1.0E-13
  integer,               parameter :: n     = 3

  type (cartesian3D_t)             :: cart
  real (kind=real_kind)            :: coords(4,3), dxds(3,2), dsdx(2,3)
  real (kind=real_kind)            :: p(3), pc(2), dx(3), x(3), ds(2)
  real (kind=real_kind)            :: dist, step

  integer                          :: i,j,k,iter
  do i=1,4
    coords(i,1) = corners3D(i)%x
    coords(i,2) = corners3D(i)%y
    coords(i,3) = corners3D(i)%z
  end do

  pc = 0
  p  = 0
  cart = spherical_to_cart(sphere)

  p(1) = cart%x
  p(2) = cart%y
  p(3) = cart%z

  dx   = 0
  ds   = 0
  dsdx = 0
  dxds = 0

  !*-------------------------------------------------------------------------*!

  ! Initial guess, center of element
  dist = 9999999.
  step = 9999999.
  iter = 0

  do while  (TOL*TOL.lt.dist .and. iter.lt.MAXIT .and. TOL*TOL.lt.step)
    iter = iter + 1

    dxds =  DF (coords, pc)
    x    =   F (coords, pc)
    dsdx = INV (dxds)

    dx   = x - p
    dist = DOT_PRODUCT(dx,dx)
    ds   = MATMUL(dsdx, dx)
    pc   = pc - ds
    step = DOT_PRODUCT(ds,ds)
  enddo

  ref%x = pc(1)
  ref%y = pc(2)
end function


subroutine  ALE_parametric_coords (parametric_coord, elem_indexes, dep_points, num_neighbors, ngh_corners)
  use coordinate_systems_mod, only : cartesian2d_t, cartesian3D_t, spherical_polar_t, change_coordinates, distance
  use interpolate_mod,        only : parametric_coordinates
  use dimensions_mod,         only : np

  implicit none

  type(cartesian2D_t)       , intent(out)       :: parametric_coord(np,np)
  type(cartesian3D_t)       , intent(in)        :: dep_points(np,np)
  integer                   , intent(in)        :: elem_indexes(np,np)
  integer                   , intent(in)        :: num_neighbors
  type(cartesian3D_t)       , intent(in)        :: ngh_corners(4,num_neighbors)

  type (spherical_polar_t)                      :: sphere(np,np)
  integer                                       :: i,j,n
  type(cartesian2D_t)                           :: parametric_test(np,np)
  real(kind=real_kind)                          :: d

  do j=1,np
    sphere(:,j) = change_coordinates(dep_points(:,j))
  end do

  do i=1,np
    do j=1,np
      n = elem_indexes(i,j)
      parametric_coord(i,j)= parametric_coordinates(sphere(i,j),ngh_corners(:,n))
    end do
  end do
end subroutine ALE_parametric_coords


!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
! forward-in-time 2 level vertically lagrangian step
!  this code takes a lagrangian step in the horizontal
! (complete with DSS), and then applies a vertical remap
!
! This routine may use dynamics fields at timelevel np1
! In addition, other fields are required, which have to be
! explicitly saved by the dynamics:  (in elem(ie)%derived struct)
!
! Fields required from dynamics: (in
!    omega_p   it will be DSS'd here, for later use by CAM physics
!              we DSS omega here because it can be done for "free"
!    Consistent mass/tracer-mass advection (used if subcycling turned on)
!       dp()   dp at timelevel n0
!       vn0()  mean flux  < U dp > going from n0 to np1
!
! 3 stage
!    Euler step from t     -> t+.5
!    Euler step from t+.5  -> t+1.0
!    Euler step from t+1.0 -> t+1.5
!    u(t) = u(t)/3 + u(t+2)*2/3
!
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
  subroutine Prim_Advec_Tracers_remap_rk2( elem , deriv , hvcoord , flt , hybrid , dt , tl , nets , nete )
    use iso_c_binding , only : c_ptr, c_loc
    use element_mod   , only : element_t
    use perf_mod      , only : t_startf, t_stopf            ! _EXTERNAL
    use derivative_mod, only : divergence_sphere
    use control_mod   , only : vert_remap_q_alg, qsplit
    implicit none

    type (element_t)     , intent(inout) :: elem(:)
    type (derivative_t)  , intent(in   ) :: deriv
    type (hvcoord_t)     , intent(in   ) :: hvcoord
    type (filter_t)      , intent(in   ) :: flt
    type (hybrid_t)      , intent(in   ) :: hybrid
    real(kind=real_kind) , intent(in   ) :: dt
    type (TimeLevel_t)   , intent(inout) :: tl
    integer              , intent(in   ) :: nets
    integer              , intent(in   ) :: nete

    integer :: i,j,k,l,ie,q,nmin
    integer :: nfilt,rkstage,rhs_multiplier
    integer :: n0_qdp, np1_qdp

    call t_barrierf('sync_prim_advec_tracers_remap_k2', hybrid%par%comm)
    call t_startf('prim_advec_tracers_remap_rk2')
!    call extrae_user_function(1)
    call TimeLevel_Qdp( tl, qsplit, n0_qdp, np1_qdp) !time levels for qdp are not the same
    rkstage = 3 !   3 stage RKSSP scheme, with optimal SSP CFL

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! RK2 2D advection step
    ! note: stage 3 we take the oppertunity to DSS omega
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! use these for consistent advection (preserve Q=1)
    ! derived%vdp_ave        =  mean horiz. flux:   U*dp
    ! derived%eta_dot_dpdn    =  mean vertical velocity (used for remap)
    ! derived%omega_p         =  advection code will DSS this for the physics, but otherwise
    !                            it is not needed
    ! Also: save a copy of div(U dp) in derived%div(:,:,:,1), which will be DSS'd
    !       and a DSS'ed version stored in derived%div(:,:,:,2)

    call t_startf('precomput_divdp')
    call precompute_divdp( elem , hybrid , deriv , dt , nets , nete , n0_qdp )
    call t_stopf('precomput_divdp')

    !rhs_multiplier is for obtaining dp_tracers at each stage:
    !dp_tracers(stage) = dp - rhs_multiplier*dt*divdp_proj

    call t_startf('euler_step_0')
    rhs_multiplier = 0
    call euler_step( np1_qdp , n0_qdp  , dt/2 , elem , hvcoord , hybrid , deriv , nets , nete , DSSdiv_vdp_ave , rhs_multiplier )
    call t_stopf('euler_step_0')

    call t_startf('euler_step_1')
    rhs_multiplier = 1
    call euler_step( np1_qdp , np1_qdp , dt/2 , elem , hvcoord , hybrid , deriv , nets , nete , DSSeta         , rhs_multiplier )
    call t_stopf('euler_step_1')

    call t_startf('euler_step_2')
    rhs_multiplier = 2
    call euler_step( np1_qdp , np1_qdp , dt/2 , elem , hvcoord , hybrid , deriv , nets , nete , DSSomega       , rhs_multiplier )
    call t_stopf('euler_step_2')

    !to finish the 2D advection step, we need to average the t and t+2 results to get a second order estimate for t+1.
    call t_startf('qdp_tavg')
    call qdp_time_avg( elem , rkstage , n0_qdp , np1_qdp , limiter_option , nu_p , nets , nete )
    call t_stopf('qdp_tavg')

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !  Dissipation
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if ( (limiter_option == 8) .or. (limiter_option == 9) ) then
      ! dissipation was applied in RHS.
    else
      call t_startf('ah_scalar')
      call advance_hypervis_scalar(edgeadv,elem,hvcoord,hybrid,deriv,tl%np1,np1_qdp,nets,nete,dt)
      call t_stopf('ah_scalar')
    endif
!    call extrae_user_function(0)

    call t_stopf('prim_advec_tracers_remap_rk2')

  end subroutine prim_advec_tracers_remap_rk2

!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------

  subroutine precompute_divdp( elem , hybrid , deriv , dt , nets , nete , n0_qdp )
    implicit none
    type(element_t)      , intent(inout) :: elem(:)
    type (hybrid_t)      , intent(in   ) :: hybrid
    type (derivative_t)  , intent(in   ) :: deriv
    real(kind=real_kind) , intent(in   ) :: dt
    integer              , intent(in   ) :: nets , nete , n0_qdp
    integer :: ie , k

    do ie = nets , nete
      do k = 1 , nlev   ! div( U dp Q),
        elem(ie)%derived%divdp(:,:,k) = divergence_sphere(elem(ie)%derived%vn0(:,:,:,k),deriv,elem(ie))
        elem(ie)%derived%divdp_proj(:,:,k) = elem(ie)%derived%divdp(:,:,k)
      enddo
    enddo
  end subroutine precompute_divdp
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------

  subroutine qdp_time_avg( elem , rkstage , n0_qdp , np1_qdp , limiter_option , nu_p , nets , nete )
    implicit none
    type(element_t)     , intent(inout) :: elem(:)
    integer             , intent(in   ) :: rkstage , n0_qdp , np1_qdp , nets , nete , limiter_option
    real(kind=real_kind), intent(in   ) :: nu_p
    integer :: ie,q,k
    real(kind=real_kind) :: rrkstage
#ifdef NEWEULER_B4B
    do ie=nets,nete
      elem(ie)%state%Qdp(:,:,:,1:qsize,np1_qdp) =               &
                   ( elem(ie)%state%Qdp(:,:,:,1:qsize,n0_qdp) + &
                     (rkstage-1)*elem(ie)%state%Qdp(:,:,:,1:qsize,np1_qdp) ) / rkstage
    enddo
#else
    rrkstage=1.0d0/real(rkstage,kind=real_kind)
    do ie=nets,nete
      do q=1,qsize
        do k=1,nlev
           elem(ie)%state%Qdp(:,:,k,q,np1_qdp) =               &
               rrkstage *( elem(ie)%state%Qdp(:,:,k,q,n0_qdp) + &
               (rkstage-1)*elem(ie)%state%Qdp(:,:,k,q,np1_qdp) )
        enddo
      enddo
    enddo
#endif
  end subroutine qdp_time_avg

!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------

  subroutine euler_step( np1_qdp , n0_qdp , dt , elem , hvcoord , hybrid , deriv , nets , nete , DSSopt , rhs_multiplier )
    ! ===================================
    ! This routine is the basic foward
    ! euler component used to construct RK SSP methods
    !
    !           u(np1) = u(n0) + dt2*DSS[ RHS(u(n0)) ]
    !
    ! n0 can be the same as np1.
    !
    ! DSSopt = DSSeta or DSSomega:   also DSS eta_dot_dpdn or omega
    !
    ! ===================================
    use kinds          , only : real_kind
    use dimensions_mod , only : np, nlev
    use hybrid_mod     , only : hybrid_t
    use element_mod    , only : element_t, elem_state_Qdp, elem_derived_eta_dot_dpdn, &
         elem_derived_omega_p, elem_derived_divdp_proj, elem_derived_vn0, elem_derived_dp, &
         elem_derived_divdp, elem_derived_dpdiss_biharmonic, elem_derived_dpdiss_ave
    use derivative_mod , only : derivative_t, divergence_sphere, gradient_sphere, vorticity_sphere, &
         divergence_sphere_wk
    use edge_mod       , only : edgevpack, edgevunpack
    use bndry_mod      , only : bndry_exchangev
    use hybvcoord_mod  , only : hvcoord_t
    use parallel_mod, only : abortmp, iam
!!!!!!!!!!!!!!!!!!!!!!
    use utils_mod, only: FrobeniusNorm
!!!!!!!!!!!!!!!!!!!!!!

    implicit none

    integer              , intent(in   )         :: np1_qdp, n0_qdp
    real (kind=real_kind), intent(in   )         :: dt
    type (element_t)     , intent(inout), target :: elem(:)
    type (hvcoord_t)     , intent(in   )         :: hvcoord
    type (hybrid_t)      , intent(in   )         :: hybrid
    type (derivative_t)  , intent(in   )         :: deriv
    integer              , intent(in   )         :: nets
    integer              , intent(in   )         :: nete
    integer              , intent(in   )         :: DSSopt
    integer              , intent(in   )         :: rhs_multiplier

    ! local
    real(kind=real_kind), dimension(np,np,2,nlev,nets:nete),     target :: Vstar
    real(kind=real_kind), dimension(np,np,nlev,qsize,nets:nete), target :: Qtens
    real(kind=real_kind), dimension(np,np,nlev,nets:nete), target       :: dpdissk
    real(kind=real_kind), dimension(np,np  )                            :: divdp, dpdiss
    real(kind=real_kind), dimension(np,np,2)                            :: gradQ
    real(kind=real_kind), dimension(np,np,nlev                )         :: dp,dp_star
    real(kind=real_kind), dimension(np,np,nlev,qsize,nets:nete), target :: Qtens_biharmonic
    real(kind=real_kind), dimension(:,:,:), pointer                     :: DSSvar
    real(kind=real_kind), dimension(nlev)                               :: dp0
    integer :: ie,q,i,j,k, kptr
    integer :: rhs_viss

    !  call t_barrierf('sync_euler_step', hybrid%par%comm)
    OMP_SIMD
    do k = 1 , nlev
       dp0(k) = ( hvcoord%hyai(k+1) - hvcoord%hyai(k) )*hvcoord%ps0 + &
            ( hvcoord%hybi(k+1) - hvcoord%hybi(k) )*hvcoord%ps0
    enddo
    !pw  call t_startf('euler_step')

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !   compute Q min/max values for lim8
    !   compute biharmonic mixing term f
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    rhs_viss = 0
    if ( (limiter_option == 8) .or. (limiter_option == 9) ) then
       call t_startf('bihmix_qminmax')
       ! when running lim8, we also need to limit the biharmonic, so that term needs
       ! to be included in each euler step.  three possible algorithms here:
       ! 1) most expensive:
       !     compute biharmonic (which also computes qmin/qmax) during all 3 stages
       !     be sure to set rhs_viss=1
       !     cost:  3 biharmonic steps with 3 DSS
       !
       ! 2) cheapest:
       !     compute biharmonic (which also computes qmin/qmax) only on first stage
       !     be sure to set rhs_viss=3
       !     reuse qmin/qmax for all following stages (but update based on local qmin/qmax)
       !     cost:  1 biharmonic steps with 1 DSS
       !     main concern:  viscosity
       !
       ! 3)  compromise:
       !     compute biharmonic (which also computes qmin/qmax) only on last stage
       !     be sure to set rhs_viss=3
       !     compute qmin/qmax directly on first stage
       !     reuse qmin/qmax for 2nd stage stage (but update based on local qmin/qmax)
       !     cost:  1 biharmonic steps, 2 DSS
       !
       !  NOTE  when nu_p=0 (no dissipation applied in dynamics to dp equation), we should
       !        apply dissipation to Q (not Qdp) to preserve Q=1
       !        i.e.  laplace(Qdp) ~  dp0 laplace(Q)
       !        for nu_p=nu_q>0, we need to apply dissipation to Q * diffusion_dp
       !
       ! initialize dp, and compute Q from Qdp (and store Q in Qtens_biharmonic)
       do ie = nets , nete
          ! add hyperviscosity to RHS.  apply to Q at timelevel n0, Qdp(n0)/dp
          OMP_SIMD
          do k = 1 , nlev    !  Loop index added with implicit inversion (AAM)
             dp(:,:,k) = elem(ie)%derived%dp(:,:,k) - rhs_multiplier*dt*elem(ie)%derived%divdp_proj(:,:,k)
          enddo
#if (defined COLUMN_OPENMP)
          !$omp parallel do private(q,k) collapse(2)
#endif
          do q = 1 , qsize
             do k=1,nlev
                Qtens_biharmonic(:,:,k,q,ie) = elem(ie)%state%Qdp(:,:,k,q,n0_qdp)/dp(:,:,k)
                if ( rhs_multiplier == 1 ) then
                   ! for this stage, we skip neighbor_minmax() call, but update
                   ! qmin/qmax with any new local extrema:
                   qmin(k,q,ie)=min(qmin(k,q,ie),minval(Qtens_biharmonic(:,:,k,q,ie)))
                   qmax(k,q,ie)=max(qmax(k,q,ie),maxval(Qtens_biharmonic(:,:,k,q,ie)))
                else
                   ! for rhs_multiplier=0,2 we will call neighbor_minmax and compute
                   ! the correct min/max values
                   qmin(k,q,ie)=minval(Qtens_biharmonic(:,:,k,q,ie))
                   qmax(k,q,ie)=maxval(Qtens_biharmonic(:,:,k,q,ie))
                endif
             enddo
          enddo
       enddo
       ! compute element qmin/qmax
       if ( rhs_multiplier == 0 ) then
          ! update qmin/qmax based on neighbor data for lim8
          call t_startf('eus_neighbor_minmax1')
          call neighbor_minmax(hybrid,edgeAdvQminmax,nets,nete,qmin(:,:,nets:nete),qmax(:,:,nets:nete))
          call t_stopf('eus_neighbor_minmax1')
       endif

       ! get new min/max values, and also compute biharmonic mixing term
       if ( rhs_multiplier == 2 ) then
          rhs_viss = 3
          ! two scalings depending on nu_p:
          ! nu_p=0:    qtens_biharmonic *= dp0                   (apply viscosity only to q)
          ! nu_p>0):   qtens_biharmonc *= elem()%psdiss_ave      (for consistency, if nu_p=nu_q)
          if ( nu_p > 0 ) then
             do ie = nets , nete
#ifdef NEWEULER_B4B
#if (defined COLUMN_OPENMP)
                !$omp parallel do private(k, q) collapse(2)
#endif
                do k = 1 , nlev
                   do q = 1 , qsize
                      ! NOTE: divide by dp0 since we multiply by dp0 below
                      Qtens_biharmonic(:,:,k,q,ie)=Qtens_biharmonic(:,:,k,q,ie)&
                           *elem(ie)%derived%dpdiss_ave(:,:,k)/dp0(k)
                   enddo
                enddo
#else
#if (defined COLUMN_OPENMP)
                !$omp parallel do private(k)
#endif
                do k = 1 , nlev
                   ! NOTE: divide by dp0 since we multiply by dp0 below
                   dpdissk(:,:,k,ie) = elem(ie)%derived%dpdiss_ave(:,:,k)/dp0(k)
                enddo
#if (defined COLUMN_OPENMP)
                !$omp parallel do private(q,k) collapse(2)
#endif
                do q = 1 , qsize
                   do k = 1 , nlev
                      Qtens_biharmonic(:,:,k,q,ie)=Qtens_biharmonic(:,:,k,q,ie)*dpdissk(:,:,k,ie)
                   enddo
                enddo
#endif
             enddo ! ie loop
          endif ! nu_p > 0

          !   Previous version of biharmonic_wk_scalar_minmax included a min/max
          !   calculation into the boundary exchange.  This was causing cache issues.
          !   Split the single operation into two separate calls
          !      call neighbor_minmax()
          !      call biharmonic_wk_scalar()
          !
          !      call biharmonic_wk_scalar_minmax( elem , qtens_biharmonic , deriv , edgeAdvQ3 , hybrid , &
          !           nets , nete , qmin(:,:,nets:nete) , qmax(:,:,nets:nete) )
#ifdef OVERLAP
          call neighbor_minmax_start(hybrid,edgeAdvQminmax,nets,nete,qmin(:,:,nets:nete),qmax(:,:,nets:nete))
          call biharmonic_wk_scalar(elem,qtens_biharmonic,deriv,edgeAdv,hybrid,nets,nete)
          do ie = nets , nete
#if (defined COLUMN_OPENMP_notB4B)
             !$omp parallel do private(k, q)
#endif
             do q = 1 , qsize
                do k = 1 , nlev    !  Loop inversion (AAM)
                   ! note: biharmonic_wk() output has mass matrix already applied. Un-apply since we apply again below:
                   qtens_biharmonic(:,:,k,q,ie) = &
                        -rhs_viss*dt*nu_q*dp0(k)*Qtens_biharmonic(:,:,k,q,ie) / elem(ie)%spheremp(:,:)
                enddo
             enddo
          enddo
          call neighbor_minmax_finish(hybrid,edgeAdvQminmax,nets,nete,qmin(:,:,nets:nete),qmax(:,:,nets:nete))
#else

          call t_startf('eus_neighbor_minmax2')
          call neighbor_minmax(hybrid,edgeAdvQminmax,nets,nete,qmin(:,:,nets:nete),qmax(:,:,nets:nete))
          call t_stopf('eus_neighbor_minmax2')
          call biharmonic_wk_scalar(elem,qtens_biharmonic,deriv,edgeAdv,hybrid,nets,nete)

          do ie = nets , nete
#if (defined COLUMN_OPENMP)
             !$omp parallel do private(k, q) collapse(2)
#endif
             do q = 1 , qsize
                do k = 1 , nlev    !  Loop inversion (AAM)
                   ! note: biharmonic_wk() output has mass matrix already applied. Un-apply since we apply again below:
                   qtens_biharmonic(:,:,k,q,ie) = &
                        -rhs_viss*dt*nu_q*dp0(k)*Qtens_biharmonic(:,:,k,q,ie) / elem(ie)%spheremp(:,:)
                enddo
             enddo
          enddo
#endif

       endif
       call t_stopf('bihmix_qminmax')
    endif  ! compute biharmonic mixing term and qmin/qmax
    ! end of limiter_option == 8 or == 9


    call t_startf('eus_2d_advec')
    call t_startf("advance_qdp")
    call advance_qdp_f90(nets,nete, &
         rhs_multiplier,DSSopt,dp,dpdissk, &
         n0_qdp,dt,Vstar,elem,deriv,Qtens, &
         rhs_viss,Qtens_biharmonic,np1_qdp)
    call t_stopf("advance_qdp")
    call t_startf('eus_bexchV')
    call bndry_exchangeV( hybrid , edgeAdvp1 )
    call t_stopf('eus_bexchV')

    do ie = nets , nete
       if ( DSSopt == DSSeta         ) DSSvar => elem(ie)%derived%eta_dot_dpdn(:,:,:)
       if ( DSSopt == DSSomega       ) DSSvar => elem(ie)%derived%omega_p(:,:,:)
       if ( DSSopt == DSSdiv_vdp_ave ) DSSvar => elem(ie)%derived%divdp_proj(:,:,:)

       call edgeVunpack( edgeAdvp1 , DSSvar(:,:,1:nlev) , nlev , qsize*nlev , ie )
       OMP_SIMD
       do k = 1 , nlev
          DSSvar(:,:,k) = DSSvar(:,:,k) * elem(ie)%rspheremp(:,:)
       enddo

#if (defined COLUMN_OPENMP)
       !$omp parallel do private(q,k)
#endif
       do q = 1 , qsize
          call edgeVunpack( edgeAdvp1 , elem(ie)%state%Qdp(:,:,:,q,np1_qdp) , nlev , nlev*(q-1) , ie )
          do k = 1 , nlev    !  Potential loop inversion (AAM)
             elem(ie)%state%Qdp(:,:,k,q,np1_qdp) = elem(ie)%rspheremp(:,:) * elem(ie)%state%Qdp(:,:,k,q,np1_qdp)
          enddo
       enddo
    enddo
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
    !$OMP BARRIER
#endif
#endif

    call t_stopf('eus_2d_advec')

  end subroutine euler_step
!-----------------------------------------------------------------------------



  subroutine limiter2d_zero(Q)
  ! mass conserving zero limiter (2D only).  to be called just before DSS
  !
  ! this routine is called inside a DSS loop, and so Q had already
  ! been multiplied by the mass matrix.  Thus dont include the mass
  ! matrix when computing the mass = integral of Q over the element
  !
  ! ps is only used when advecting Q instead of Qdp
  ! so ps should be at one timelevel behind Q
  implicit none
  real (kind=real_kind), intent(inout) :: Q(np,np,nlev)

  ! local
  real (kind=real_kind) :: dp(np,np)
  real (kind=real_kind) :: mass,mass_new,ml
  integer i,j,k

  do k = nlev , 1 , -1
    mass = 0
    do j = 1 , np
      do i = 1 , np
        !ml = Q(i,j,k)*dp(i,j)*spheremp(i,j)  ! see above
        ml = Q(i,j,k)
        mass = mass + ml
      enddo
    enddo

    ! negative mass.  so reduce all postive values to zero
    ! then increase negative values as much as possible
    if ( mass < 0 ) Q(:,:,k) = -Q(:,:,k)
    mass_new = 0
    do j = 1 , np
      do i = 1 , np
        if ( Q(i,j,k) < 0 ) then
          Q(i,j,k) = 0
        else
          ml = Q(i,j,k)
          mass_new = mass_new + ml
        endif
      enddo
    enddo

    ! now scale the all positive values to restore mass
    if ( mass_new > 0 ) Q(:,:,k) = Q(:,:,k) * abs(mass) / mass_new
    if ( mass     < 0 ) Q(:,:,k) = -Q(:,:,k)
  enddo
  end subroutine limiter2d_zero

!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------

  subroutine advance_hypervis_scalar( edgeAdv , elem , hvcoord , hybrid , deriv , nt , nt_qdp , nets , nete , dt2 )
  !  hyperviscsoity operator for foward-in-time scheme
  !  take one timestep of:
  !          Q(:,:,:,np) = Q(:,:,:,np) +  dt2*nu*laplacian**order ( Q )
  !
  !  For correct scaling, dt2 should be the same 'dt2' used in the leapfrog advace
  use kinds          , only : real_kind
  use dimensions_mod , only : np, nlev
  use hybrid_mod     , only : hybrid_t
  use element_mod    , only : element_t
  use derivative_mod , only : derivative_t
  use edge_mod       , only : edgevpack, edgevunpack
  use edgetype_mod   , only : EdgeBuffer_t
  use bndry_mod      , only : bndry_exchangev
  use perf_mod       , only : t_startf, t_stopf                          ! _EXTERNAL
  implicit none
  type (EdgeBuffer_t)  , intent(inout)         :: edgeAdv
  type (element_t)     , intent(inout), target :: elem(:)
  type (hvcoord_t)     , intent(in   )         :: hvcoord
  type (hybrid_t)      , intent(in   )         :: hybrid
  type (derivative_t)  , intent(in   )         :: deriv
  integer              , intent(in   )         :: nt
  integer              , intent(in   )         :: nt_qdp
  integer              , intent(in   )         :: nets
  integer              , intent(in   )         :: nete
  real (kind=real_kind), intent(in   )         :: dt2

  ! local
  real (kind=real_kind), dimension(np,np,nlev,qsize,nets:nete) :: Qtens
  real (kind=real_kind), dimension(np,np,nlev                ) :: dp
  real (kind=real_kind) :: dt,dp0
  integer :: k , i , j , ie , ic , q

  if ( nu_q           == 0 ) return
  if ( hypervis_order /= 2 ) return
!   call t_barrierf('sync_advance_hypervis_scalar', hybrid%par%comm)
  call t_startf('advance_hypervis_scalar')

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  hyper viscosity
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  dt = dt2 / hypervis_subcycle_q

  do ic = 1 , hypervis_subcycle_q
    do ie = nets , nete
      ! Qtens = Q/dp   (apply hyperviscsoity to dp0 * Q, not Qdp)
      ! various options:
      !   1)  biharmonic( Qdp )
      !   2)  dp0 * biharmonic( Qdp/dp )
      !   3)  dpave * biharmonic(Q/dp)
      ! For trace mass / mass consistenciy, we use #2 when nu_p=0
      ! and #e when nu_p>0, where dpave is the mean mass flux from the nu_p
      ! contribution from dynamics.

      if (nu_p>0) then
#if (defined COLUMN_OPENMP)
!$omp parallel do private(q,k) collapse(2)
#endif
        do q = 1 , qsize
          do k = 1 , nlev
            dp(:,:,k) = elem(ie)%derived%dp(:,:,k) - dt2*elem(ie)%derived%divdp_proj(:,:,k)
            Qtens(:,:,k,q,ie) = elem(ie)%derived%dpdiss_ave(:,:,k)*&
                                elem(ie)%state%Qdp(:,:,k,q,nt_qdp) / dp(:,:,k)
          enddo
        enddo

      else
#if (defined COLUMN_OPENMP)
!$omp parallel do private(q,k,dp0) collapse(2)
#endif
        do q = 1 , qsize
          do k = 1 , nlev
            dp(:,:,k) = elem(ie)%derived%dp(:,:,k) - dt2*elem(ie)%derived%divdp_proj(:,:,k)
            dp0 = ( hvcoord%hyai(k+1) - hvcoord%hyai(k) ) * hvcoord%ps0 + &
                  ( hvcoord%hybi(k+1) - hvcoord%hybi(k) ) * hvcoord%ps0
            Qtens(:,:,k,q,ie) = dp0*elem(ie)%state%Qdp(:,:,k,q,nt_qdp) / dp(:,:,k)
          enddo
        enddo
      endif
    enddo ! ie loop

    ! compute biharmonic operator. Qtens = input and output
    call biharmonic_wk_scalar( elem , Qtens , deriv , edgeAdv , hybrid , nets , nete )

    do ie = nets , nete
#if (defined COLUMN_OPENMP)
!$omp parallel do private(q,k,j,i)
#endif
      do q = 1 , qsize
        do k = 1 , nlev
          do j = 1 , np
            do i = 1 , np
              ! advection Qdp.  For mass advection consistency:
              ! DIFF( Qdp) ~   dp0 DIFF (Q)  =  dp0 DIFF ( Qdp/dp )
              elem(ie)%state%Qdp(i,j,k,q,nt_qdp) = elem(ie)%state%Qdp(i,j,k,q,nt_qdp) * elem(ie)%spheremp(i,j) &
                                                   - dt * nu_q * Qtens(i,j,k,q,ie)
            enddo
          enddo
        enddo

        if (limiter_option .ne. 0 ) then
           ! smooth some of the negativities introduced by diffusion:
           call limiter2d_zero( elem(ie)%state%Qdp(:,:,:,q,nt_qdp) )
        endif

      enddo
      call edgeVpack  ( edgeAdv , elem(ie)%state%Qdp(:,:,:,:,nt_qdp) , qsize*nlev , 0 , ie )
    enddo ! ie loop

    call t_startf('ah_scalar_bexchV')
    call bndry_exchangeV( hybrid , edgeAdv )
    call t_stopf('ah_scalar_bexchV')

    do ie = nets , nete
      call edgeVunpack( edgeAdv , elem(ie)%state%Qdp(:,:,:,:,nt_qdp) , qsize*nlev , 0 , ie )
#if (defined COLUMN_OPENMP)
!$omp parallel do private(q,k) collapse(2)
#endif
      do q = 1 , qsize
        ! apply inverse mass matrix
        do k = 1 , nlev
          elem(ie)%state%Qdp(:,:,k,q,nt_qdp) = elem(ie)%rspheremp(:,:) * elem(ie)%state%Qdp(:,:,k,q,nt_qdp)
        enddo
      enddo
    enddo ! ie loop
#ifdef DEBUGOMP
#if (defined HORIZ_OPENMP)
!$OMP BARRIER
#endif
#endif
  enddo
  call t_stopf('advance_hypervis_scalar')
  end subroutine advance_hypervis_scalar
#endif

#ifndef USE_KOKKOS_KERNELS
  subroutine vertical_remap_interface(hybrid,elem,hvcoord,dt,np1,np1_qdp,nets,nete)
    use kinds,          only: real_kind
    use hybvcoord_mod,  only: hvcoord_t
    use hybrid_mod,     only: hybrid_t

    implicit none

    type (hybrid_t),  intent(in)      :: hybrid  ! distributed parallel structure (shared)
    type (element_t), intent(inout)   :: elem(:)
    type (hvcoord_t), intent(in)      :: hvcoord
    real (kind=real_kind), intent(in) :: dt
    integer, intent(in)               :: np1,np1_qdp,nets,nete

    call t_startf('total vertical remap time')
    call vertical_remap(hybrid,elem,hvcoord,dt,np1,np1_qdp,nets,nete)
    call t_stopf('total vertical remap time')
  end subroutine vertical_remap_interface

  subroutine vertical_remap(hybrid,elem,hvcoord,dt,np1,np1_qdp,nets,nete)

  ! This routine is called at the end of the vertically Lagrangian
  ! dynamics step to compute the vertical flux needed to get back
  ! to reference eta levels
  !
  ! input:
  !     derived%dp()  delta p on levels at beginning of timestep
  !     state%dp3d(np1)  delta p on levels at end of timestep
  ! output:
  !     state%ps_v(np1)          surface pressure at time np1
  !     derived%eta_dot_dpdn()   vertical flux from final Lagrangian
  !                              levels to reference eta levels
  !
  use kinds,          only: real_kind
  use hybvcoord_mod,  only: hvcoord_t
  use vertremap_mod,  only: remap1, remap1_nofilter, remap_q_ppm ! _EXTERNAL (actually INTERNAL)
  use control_mod,    only: rsplit, tracer_transport_type
  use parallel_mod,   only: abortmp
  use hybrid_mod,     only: hybrid_t

  type (hybrid_t),  intent(in)    :: hybrid  ! distributed parallel structure (shared)
  real (kind=real_kind)           :: cdp(1:nc,1:nc,nlev,ntrac)
  real (kind=real_kind)           :: psc(nc,nc), dpc(nc,nc,nlev),dpc_star(nc,nc,nlev)
  type (element_t), intent(inout) :: elem(:)
  type (hvcoord_t)                :: hvcoord
  real (kind=real_kind)           :: dt

  integer :: ie,i,j,k,np1,nets,nete,np1_qdp
  integer :: q

  real (kind=real_kind), dimension(np,np,nlev)  :: dp,dp_star
  real (kind=real_kind), dimension(np,np,nlev,2)  :: ttmp

  call t_startf('vertical_remap')

  ! reference levels:
  !   dp(k) = (hyai(k+1)-hyai(k))*ps0 + (hybi(k+1)-hybi(k))*ps_v(i,j)
  !   hybi(1)=0          pure pressure at top of atmosphere
  !   hyai(1)=ptop
  !   hyai(nlev+1) = 0   pure sigma at bottom
  !   hybi(nlev+1) = 1
  !
  ! sum over k=1,nlev
  !  sum(dp(k)) = (hyai(nlev+1)-hyai(1))*ps0 + (hybi(nlev+1)-hybi(1))*ps_v
  !             = -ps0 + ps_v
  !  ps_v =  ps0+sum(dp(k))
  !
  ! reference levels:
  !    dp(k) = (hyai(k+1)-hyai(k))*ps0 + (hybi(k+1)-hybi(k))*ps_v
  ! floating levels:
  !    dp_star(k) = dp(k) + dt_q*(eta_dot_dpdn(i,j,k+1) - eta_dot_dpdn(i,j,k) )
  ! hence:
  !    (dp_star(k)-dp(k))/dt_q = (eta_dot_dpdn(i,j,k+1) - eta_dot_dpdn(i,j,k) )
  !

  do ie=nets,nete
     ! update final ps_v
     elem(ie)%state%ps_v(:,:,np1) = 0.0
     do k=1,nlev
        elem(ie)%state%ps_v(:,:,np1) = elem(ie)%state%ps_v(:,:,np1) + elem(ie)%state%dp3d(:,:,k,np1)
     end do
     elem(ie)%state%ps_v(:,:,np1) = elem(ie)%state%ps_v(:,:,np1) + hvcoord%hyai(1)*hvcoord%ps0
     do k=1,nlev
        ! target layer thickness
        dp(:,:,k) = ( hvcoord%hyai(k+1) - hvcoord%hyai(k) )*hvcoord%ps0 + &
             ( hvcoord%hybi(k+1) - hvcoord%hybi(k))*elem(ie)%state%ps_v(:,:,np1)
        if (rsplit==0) then
           ! source layer thickness
           dp_star(:,:,k) = dp(:,:,k) + dt*(elem(ie)%derived%eta_dot_dpdn(:,:,k+1) -&
                elem(ie)%derived%eta_dot_dpdn(:,:,k))
        else
           ! source layer thickness
           dp_star(:,:,k) = elem(ie)%state%dp3d(:,:,k,np1)
        endif
     enddo

     if (minval(dp_star)<0) then
        do k=1,nlev
        do i=1,np
        do j=1,np
           if (dp_star(i,j,k ) < 0) then
              print *,'level = ',k
              print *,"column location lat,lon(radians):",elem(ie)%spherep(i,j)%lat,elem(ie)%spherep(i,j)%lon
           endif
        enddo
        enddo
        enddo
        call abortmp('negative layer thickness.  timestep or remap time too large')
     endif

     if (rsplit>0) then
        !  REMAP u,v,T from levels in dp3d() to REF levels
        ttmp(:,:,:,1)=elem(ie)%state%t(:,:,:,np1)
        ttmp(:,:,:,1)=ttmp(:,:,:,1)*dp_star

        ! ttmp    Field to be remapped
        ! np      Number of points ??? Why is this a parameter
        ! 1       Number of fields to remap
        ! dp_star Source layer thickness
        ! dp      Target layer thickness
        call t_startf('vertical_remap1_1')
        call remap1(ttmp,np,1,dp_star,dp)
        call t_stopf('vertical_remap1_1')

        elem(ie)%state%t(:,:,:,np1)=ttmp(:,:,:,1)/dp

        ttmp(:,:,:,1)=elem(ie)%state%v(:,:,1,:,np1)*dp_star
        ttmp(:,:,:,2)=elem(ie)%state%v(:,:,2,:,np1)*dp_star

        call t_startf('vertical_remap1_2')
        call remap1(ttmp,np,2,dp_star,dp)
        call t_stopf('vertical_remap1_2')

        elem(ie)%state%v(:,:,1,:,np1)=ttmp(:,:,:,1)/dp
        elem(ie)%state%v(:,:,2,:,np1)=ttmp(:,:,:,2)/dp
     endif

     ! remap the gll tracers from lagrangian levels (dp_star)  to REF levels dp
     if (qsize>0) then

       call t_startf('vertical_remap1_3')
       call remap1(elem(ie)%state%Qdp(:,:,:,:,np1_qdp),np,qsize,dp_star,dp)
       call t_stopf('vertical_remap1_3')

     endif

  enddo
  call t_stopf('vertical_remap')
  end subroutine vertical_remap
#endif

end module prim_advection_mod_base
