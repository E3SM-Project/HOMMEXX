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
  use fvm_control_volume_mod, only        : fvm_struct
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

    if  (use_semi_lagrange_transport) then
       call initghostbuffer3D(ghostbuf_tr,nlev*qsize,np)
       call interpolate_tracers_init()
    endif

  end subroutine Prim_Advec_Init1

  subroutine Prim_Advec_Init_deriv(hybrid,fvm_corners, fvm_points)

    use kinds,          only : longdouble_kind
    use dimensions_mod, only : nc
    use derivative_mod, only : derivinit
    implicit none
    type (hybrid_t), intent(in) :: hybrid
    real(kind=longdouble_kind), intent(in) :: fvm_corners(nc+1)
    real(kind=longdouble_kind), intent(in) :: fvm_points(nc)

    ! ==================================
    ! Initialize derivative structure
    ! ==================================
    call derivinit(deriv(hybrid%ithr),fvm_corners, fvm_points)
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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! fvm driver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine Prim_Advec_Tracers_fvm(elem, fvm, deriv,hvcoord,hybrid,&
        dt,tl,nets,nete)
    use perf_mod, only : t_startf, t_stopf            ! _EXTERNAL
    use vertremap_mod, only: remap1_nofilter  ! _EXTERNAL (actually INTERNAL)
    use fvm_mod, only : cslam_runairdensity, cslam_runflux, edgeveloc
    use fvm_mod, only: fvm_mcgregor, fvm_mcgregordss, fvm_rkdss
    use fvm_mod, only : fvm_ideal_test, IDEAL_TEST_OFF, IDEAL_TEST_ANALYTICAL_WINDS
    use fvm_mod, only : fvm_test_type, IDEAL_TEST_BOOMERANG, IDEAL_TEST_SOLIDBODY
    use fvm_bsp_mod, only: get_boomerang_velocities_gll, get_solidbody_velocities_gll
    use fvm_control_volume_mod, only : n0_fvm,np1_fvm,fvm_supercycling
    use control_mod, only : tracer_transport_type
    use control_mod, only : TRACERTRANSPORT_LAGRANGIAN_FVM, TRACERTRANSPORT_FLUXFORM_FVM
    use time_mod,    only : time_at

    implicit none
    type (element_t), intent(inout)   :: elem(:)
    type (fvm_struct), intent(inout)   :: fvm(:)
    type (derivative_t), intent(in)   :: deriv
    type (hvcoord_t)                  :: hvcoord
    type (hybrid_t),     intent(in):: hybrid
    type (TimeLevel_t)                :: tl

    real(kind=real_kind) , intent(in) :: dt
    integer,intent(in)                :: nets,nete


    real (kind=real_kind), dimension(np,np,nlev)    :: dp_star
    real (kind=real_kind), dimension(np,np,nlev)    :: dp
    real (kind=real_kind)                           :: eta_dot_dpdn(np,np,nlevp)
    integer :: np1,ie,k

    real (kind=real_kind)  :: vstar(np,np,2)
    real (kind=real_kind)  :: vhat(np,np,2)
    real (kind=real_kind), dimension(np, np) :: v1, v2


    call t_barrierf('sync_prim_advec_tracers_fvm', hybrid%par%comm)
    call t_startf('prim_advec_tracers_fvm')
    np1 = tl%np1

    ! departure algorithm requires two velocities:
    !
    ! fvm%v0:        velocity at beginning of tracer timestep (time n0_qdp)
    !                this was saved before the (possibly many) dynamics steps
    ! elem%derived%vstar:
    !                velocity at end of tracer timestep (time np1 = np1_qdp)
    !                for lagrangian dynamics, this is on lagrangian levels
    !                for eulerian dynamcis, this is on reference levels
    !                and it should be interpolated.
    !
    do ie=nets,nete
       elem(ie)%derived%vstar(:,:,:,:)=elem(ie)%state%v(:,:,:,:,np1)
    enddo


    if (rsplit==0) then
       ! interpolate t+1 velocity from reference levels to lagrangian levels
       ! For rsplit=0, we need to first compute lagrangian levels based on vertical velocity
       ! which requires we first DSS mean vertical velocity from dynamics
       ! note: we introduce a local eta_dot_dpdn() variable instead of DSSing elem%eta_dot_dpdn
       ! so as to preserve BFB results in some HOMME regression tests
       do ie=nets,nete
          do k=1,nlevp
             eta_dot_dpdn(:,:,k) = elem(ie)%derived%eta_dot_dpdn(:,:,k)*elem(ie)%spheremp(:,:)
          enddo
          ! eta_dot_dpdn at nlevp is zero, so we dont boundary exchange it:
          call edgeVpack(edgeAdv1,eta_dot_dpdn(:,:,1:nlev),nlev,0,ie)
       enddo

       call t_startf('pat_fvm_bexchV')
       call bndry_exchangeV(hybrid,edgeAdv1)
       call t_stopf('pat_fvm_bexchV')

       do ie=nets,nete
          ! restor interior values.  we could avoid this if we created a global array for eta_dot_dpdn
          do k=1,nlevp
             eta_dot_dpdn(:,:,k) = elem(ie)%derived%eta_dot_dpdn(:,:,k)*elem(ie)%spheremp(:,:)
          enddo
          ! unpack DSSed edge data
          call edgeVunpack(edgeAdv1,eta_dot_dpdn(:,:,1:nlev),nlev,0,ie)
          do k=1,nlevp
             eta_dot_dpdn(:,:,k) = eta_dot_dpdn(:,:,k)*elem(ie)%rspheremp(:,:)
          enddo


          ! SET VERTICAL VELOCITY TO ZERO FOR DEBUGGING
          !        elem(ie)%derived%eta_dot_dpdn(:,:,:)=0
          ! elem%state%u(np1)  = velocity at time t+1 on reference levels
          ! elem%derived%vstar = velocity at t+1 on floating levels (computed below)
!           call remap_UV_ref2lagrange(np1,dt,elem,hvcoord,ie)
          do k=1,nlev
             dp(:,:,k) = ( hvcoord%hyai(k+1) - hvcoord%hyai(k) )*hvcoord%ps0 + &
                  ( hvcoord%hybi(k+1) - hvcoord%hybi(k) )*elem(ie)%state%ps_v(:,:,np1)
             dp_star(:,:,k) = dp(:,:,k) + dt*(eta_dot_dpdn(:,:,k+1) -   eta_dot_dpdn(:,:,k))
             if (fvm_ideal_test == IDEAL_TEST_ANALYTICAL_WINDS) then
               if (fvm_test_type == IDEAL_TEST_BOOMERANG) then
                 elem(ie)%derived%vstar(:,:,:,k)=get_boomerang_velocities_gll(elem(ie), time_at(np1))
               else if (fvm_test_type == IDEAL_TEST_SOLIDBODY) then
                 elem(ie)%derived%vstar(:,:,:,k)=get_solidbody_velocities_gll(elem(ie), time_at(np1))
               else
                 call abortmp('Bad fvm_test_type in prim_step')
               end if
             end if
          enddo
          call remap1_nofilter(elem(ie)%derived%vstar,np,1,dp,dp_star)
       end do
    else
       ! do nothing
       ! for rsplit>0:  dynamics is also vertically Lagrangian, so we do not need
       ! to interpolate v(np1).
    endif


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! 2D advection step
    !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!------------------------------------------------------------------------------------
!     call t_startf('fvm_depalg')

!     call fvm_mcgregordss(elem,fvm,nets,nete, hybrid, deriv, dt*fvm_supercycling, 3)
    call fvm_rkdss(elem,fvm,nets,nete, hybrid, deriv, dt*fvm_supercycling, 3)
    write(*,*) "fvm_rkdss dt ",dt*fvm_supercycling
!     call t_stopf('fvm_depalg')

!------------------------------------------------------------------------------------

    ! fvm departure calcluation should use vstar.
    ! from c(n0) compute c(np1):
    if (tracer_transport_type == TRACERTRANSPORT_FLUXFORM_FVM) then
      call cslam_runflux(elem,fvm,hybrid,deriv,dt,tl,nets,nete,hvcoord%hyai(1)*hvcoord%ps0)
    else if (tracer_transport_type == TRACERTRANSPORT_LAGRANGIAN_FVM) then
      call cslam_runairdensity(elem,fvm,hybrid,deriv,dt,tl,nets,nete,hvcoord%hyai(1)*hvcoord%ps0)
    else
      call abortmp('Bad tracer_transport_type in Prim_Advec_Tracers_fvm')
    end if

    call t_stopf('prim_advec_tracers_fvm')
  end subroutine Prim_Advec_Tracers_fvm

end module prim_advection_mod_base
