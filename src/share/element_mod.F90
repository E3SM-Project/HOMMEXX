#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module element_mod

  use kinds,                  only: real_kind, long_kind, int_kind
  use coordinate_systems_mod, only: spherical_polar_t, cartesian2D_t, cartesian3D_t, distance
  use dimensions_mod,         only: np, npsq, nlev, nlevp, qsize_d, max_neigh_edges
  use edgetype_mod,           only: edgedescriptor_t
  use gridgraph_mod,          only: gridvertex_t

  implicit none
  private
  integer, public, parameter :: timelevels = 3

  real (kind=real_kind),    allocatable, target, public :: elem_Dinv      (:,:,:,:,:) ! (np,np,2,2,nelemd)
  real (kind=real_kind),    allocatable, target, public :: elem_D         (:,:,:,:,:) ! (np,np,2,2,nelemd)
  real (kind=real_kind),    allocatable, target, public :: elem_metinv    (:,:,:,:,:) ! (np,np,2,2,nelemd)
  real (kind=real_kind),    allocatable, target, public :: elem_metdet    (:,:,:)     ! (np,np,nelemd)
  real (kind=real_kind),    allocatable, target, public :: elem_rmetdet   (:,:,:)     ! (np,np,nelemd)
  real (kind=real_kind),    allocatable, target, public :: elem_fcor      (:,:,:)     ! (np,np,nelemd)

  real (kind=real_kind),    allocatable, target, public :: elem_mp        (:,:,:)     ! (np,np,nelemd)
  real (kind=real_kind),    allocatable, target, public :: elem_rmp       (:,:,:)     ! (np,np,nelemd)

  type (spherical_polar_t), allocatable, target, public :: elem_spherep   (:,:,:)     ! Spherical coords of GLL points
  real (kind=real_kind),    allocatable, target, public :: elem_spheremp  (:,:,:)     ! (np,np,nelemd)
  real (kind=real_kind),    allocatable, target, public :: elem_rspheremp (:,:,:)     ! (np,np,nelemd)

  real (kind=real_kind),    allocatable, target, public :: elem_sub_elem_mass_flux (:,:,:,:,:)    ! (nc,nc,4,nlev,nelemd)

  real (kind=real_kind),    allocatable, target, public :: elem_vec_sph2cart(:,:,:,:,:) ! (np,np,3,2,nelemd)
  real (kind=real_kind),    allocatable, target, public :: elem_tensorvisc     (:,:,:,:,:) ! (np,np,2,2,nelemd)
#ifdef _PRIM

  public :: setup_element_pointers
  real (kind=real_kind), allocatable, target, public :: state_Qdp                (:,:,:,:,:,:)    ! (np,np,nlev,qsize_d,2,nelemd)
  real (kind=real_kind), allocatable, target, public :: derived_vn0              (:,:,:,:,:)      ! (np,np,2,nlev,nelemd)                   velocity for SE tracer advection
  real (kind=real_kind), allocatable, target, public :: derived_divdp            (:,:,:,:)        ! (np,np,nlev,nelemd)                     divergence of dp
  real (kind=real_kind), allocatable, target, public :: derived_divdp_proj       (:,:,:,:)        ! (np,np,nlev,nelemd)                     DSSed divdp

  ! elem_state_t arrays
  real (kind=real_kind), allocatable, target, public :: elem_state_v    (:,:,:,:,:,:)            ! velocity                           1
  real (kind=real_kind), allocatable, target, public :: elem_state_Temp (:,:,:,:,:)              ! temperature                        2
  real (kind=real_kind), allocatable, target, public :: elem_state_dp3d (:,:,:,:,:)              ! delta p on levels                  8
  real (kind=real_kind), allocatable, target, public :: elem_state_ps_v (:,:,:,:)                ! surface pressure                   4
  real (kind=real_kind), allocatable, target, public :: elem_state_phis (:,:,:)                  ! surface geopotential (prescribed)  5
  real (kind=real_kind), allocatable, target, public :: elem_state_Q    (:,:,:,:,:)              ! Tracer concentration               6
  real (kind=real_kind), allocatable, target, public :: elem_state_Qdp  (:,:,:,:,:,:)            ! Tracer mass                        7

  ! derived_state_t arrays
  real (kind=real_kind), allocatable, target, public :: elem_derived_vn0              (:,:,:,:,:)   ! velocity for SE tracer advection
  real (kind=real_kind), allocatable, target, public :: elem_derived_vstar            (:,:,:,:,:)   ! velocity on Lagrangian surfaces
  real (kind=real_kind), allocatable, target, public :: elem_derived_dpdiss_biharmonic(:,:,:,:)     ! mean dp dissipation tendency, if nu_p>0
  real (kind=real_kind), allocatable, target, public :: elem_derived_dpdiss_ave       (:,:,:,:)     ! mean dp used to compute psdiss_tens
  real (kind=real_kind), allocatable, target, public :: elem_derived_phi              (:,:,:,:)     ! geopotential
  real (kind=real_kind), allocatable, target, public :: elem_derived_omega_p          (:,:,:,:)     ! vertical tendency (derived)
  real (kind=real_kind), allocatable, target, public :: elem_derived_eta_dot_dpdn     (:,:,:,:)     ! mean vertical flux from dynamics
  real (kind=real_kind), allocatable, target, public :: elem_derived_eta_dot_dpdn_prescribed(:,:,:,:)     ! mean vertical flux from dynamics
  real (kind=real_kind), allocatable, target, public :: elem_derived_zeta             (:,:,:,:)     ! relative vorticity
  real (kind=real_kind), allocatable, target, public :: elem_derived_div              (:,:,:,:,:)   ! divergence
  real (kind=real_kind), allocatable, target, public :: elem_derived_dp               (:,:,:,:)     ! for dp_tracers at physics timestep
  real (kind=real_kind), allocatable, target, public :: elem_derived_divdp            (:,:,:,:)     ! divergence of dp
  real (kind=real_kind), allocatable, target, public :: elem_derived_divdp_proj       (:,:,:,:)     ! DSSed divdp
  real (kind=real_kind), allocatable, target, public :: elem_derived_FQ               (:,:,:,:,:)   ! tracer forcing
  real (kind=real_kind), allocatable, target, public :: elem_derived_FM               (:,:,:,:,:)   ! momentum forcing
  real (kind=real_kind), allocatable, target, public :: elem_derived_FT               (:,:,:,:)     ! temperature forcing
  real (kind=real_kind), allocatable, target, public :: elem_derived_FQps             (:,:,:,:)     ! forcing of FQ on ps_v

  ! elem_accum_t arrays
  real (kind=real_kind), allocatable, target, public :: elem_accum_KEvert1    (:,:,:)     ! term from continuity equ
  real (kind=real_kind), allocatable, target, public :: elem_accum_KEvert2    (:,:,:)     ! term from momentum equ
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEvert1    (:,:,:)     ! term from continuity equ
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEvert2    (:,:,:)     ! term from T equ
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEvert1_wet(:,:,:)     ! wet term from continuity equ
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEvert2_wet(:,:,:)     ! wet term from T equ
  real (kind=real_kind), allocatable, target, public :: elem_accum_KEhorz1    (:,:,:)     ! at time t
  real (kind=real_kind), allocatable, target, public :: elem_accum_KEhorz2    (:,:,:)     ! after calling time_advance, these will be at time t-1
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEhorz1    (:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEhorz2    (:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEhorz1_wet(:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEhorz2_wet(:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_T1         (:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_T2         (:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_T2_s       (:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_S1         (:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_S1_wet     (:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_S2         (:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_DIFF       (:,:,:,:,:) ! net hypervis term
  real (kind=real_kind), allocatable, target, public :: elem_accum_DIFFT      (:,:,:,:)   ! net hypervis term
  real (kind=real_kind), allocatable, target, public :: elem_accum_CONV       (:,:,:,:,:) ! dpdn u dot CONV = T1 + T2
  real (kind=real_kind), allocatable, target, public :: elem_accum_KEner      (:,:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_PEner      (:,:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEner      (:,:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_IEner_wet  (:,:,:,:)
  real (kind=real_kind), allocatable, target, public :: elem_accum_Qvar       (:,:,:,:,:) ! Q variance at half time levels
  real (kind=real_kind), allocatable, target, public :: elem_accum_Qmass      (:,:,:,:,:) ! Q mass at half time levels
  real (kind=real_kind), allocatable, target, public :: elem_accum_Q1mass     (:,:,:,:)   ! Q mass at full time levels

#if USE_OPENACC

  type, public :: elem_state_t
    ! prognostic variables for preqx solver
    ! prognostics must match those in prim_restart_mod.F90
    ! vertically-lagrangian code advects dp3d instead of ps_v
    ! tracers Q, Qdp always use 2 level time scheme
    real (kind=real_kind) :: v   (np,np,2,nlev,timelevels)            ! velocity                           1
    real (kind=real_kind) :: T   (np,np,nlev,timelevels)              ! temperature                        2
    real (kind=real_kind) :: dp3d(np,np,nlev,timelevels)              ! delta p on levels                  8
    real (kind=real_kind) :: ps_v(np,np,timelevels)                   ! surface pressure                   4
    real (kind=real_kind) :: phis(np,np)                              ! surface geopotential (prescribed)  5
    real (kind=real_kind) :: Q   (np,np,nlev,qsize_d)                 ! Tracer concentration               6
    real (kind=real_kind), pointer :: Qdp (:,:,:,:,:)  ! Tracer mass                        7  (np,np,nlev,qsize,2)
  end type elem_state_t

  integer(kind=int_kind),public,parameter::StateComponents=8  ! num prognistics variables (for prim_restart_mod.F90)
  type, public :: derived_state_t
    ! diagnostic variables for preqx solver
    ! storage for subcycling tracers/dynamics
    ! if (compute_mean_flux==1) vn0=time_avg(U*dp) else vn0=U at tracer-time t
  real (kind=real_kind), pointer :: vn0              (:,:,:,:)       ! (np,np,2,nlev)                  velocity for SE tracer advection
    real (kind=real_kind) :: vstar(np,np,2,nlev)                      ! velocity on Lagrangian surfaces
    real (kind=real_kind) :: dpdiss_biharmonic(np,np,nlev)            ! mean dp dissipation tendency, if nu_p>0
    real (kind=real_kind) :: dpdiss_ave(np,np,nlev)                   ! mean dp used to compute psdiss_tens

    ! diagnostics for explicit timestep
    real (kind=real_kind) :: phi(np,np,nlev)                          ! geopotential
    real (kind=real_kind) :: omega_p(np,np,nlev)                      ! vertical tendency (derived)
    real (kind=real_kind) :: eta_dot_dpdn(np,np,nlevp)                ! mean vertical flux from dynamics

    ! semi-implicit diagnostics: computed in explict-component, reused in Helmholtz-component.
    real (kind=real_kind) :: zeta(np,np,nlev)                         ! relative vorticity
    real (kind=real_kind) :: div(np,np,nlev,timelevels)               ! divergence

    ! tracer advection fields used for consistency and limiters
    real (kind=real_kind) :: dp(np,np,nlev)                           ! for dp_tracers at physics timestep
    real (kind=real_kind), pointer :: divdp            (:,:,:)         ! (np,np,nlev)                    divergence of dp
    real (kind=real_kind), pointer :: divdp_proj       (:,:,:)         ! (np,np,nlev)                    DSSed divdp

#ifdef CAM
    ! forcing terms for CAM
    real (kind=real_kind) :: FQ(np,np,nlev,qsize_d)                   ! tracer forcing
    real (kind=real_kind) :: FM(np,np,2,nlev)                         ! momentum forcing
    real (kind=real_kind) :: FT(np,np,nlev)                           ! temperature forcing
    real (kind=real_kind) :: etadot_prescribed(np,np,nlevp)           ! prescribed vertical tendency
    real (kind=real_kind) :: u_met(np,np,nlev)                        ! zonal component of prescribed meteorology winds
    real (kind=real_kind) :: dudt_met(np,np,nlev)                     ! rate of change of zonal component of prescribed meteorology winds
    real (kind=real_kind) :: v_met(np,np,nlev)                        ! meridional component of prescribed meteorology winds
    real (kind=real_kind) :: dvdt_met(np,np,nlev)                     ! rate of change of meridional component of prescribed meteorology winds
    real (kind=real_kind) :: T_met(np,np,nlev)                        ! prescribed meteorology temperature
    real (kind=real_kind) :: dTdt_met(np,np,nlev)                     ! rate of change of prescribed meteorology temperature
    real (kind=real_kind) :: ps_met(np,np)                            ! surface pressure of prescribed meteorology
    real (kind=real_kind) :: dpsdt_met(np,np)                         ! rate of change of surface pressure of prescribed meteorology
    real (kind=real_kind) :: nudge_factor(np,np,nlev)                 ! nudging factor (prescribed)
    real (kind=real_kind) :: Utnd(npsq,nlev)                          ! accumulated U tendency due to nudging towards prescribed met
    real (kind=real_kind) :: Vtnd(npsq,nlev)                          ! accumulated V tendency due to nudging towards prescribed met
    real (kind=real_kind) :: Ttnd(npsq,nlev)                          ! accumulated T tendency due to nudging towards prescribed met
#else
    ! forcing terms for HOMME
    real (kind=real_kind) :: FQ(np,np,nlev,qsize_d)                   ! tracer forcing
    real (kind=real_kind) :: FM(np,np,2,nlev)                         ! momentum forcing
    real (kind=real_kind) :: FT(np,np,nlev)                           ! temperature forcing
#endif

    ! forcing terms for both CAM and HOMME
    ! FQps for conserving dry mass in the presence of precipitation

    real (kind=real_kind) :: FQps(np,np,timelevels)                   ! forcing of FQ on ps_v
  end type derived_state_t

!else for USE_OPENACC if
#else

! =========== PRIMITIVE-EQUATION DATA-STRUCTURES =====================

  type, public :: elem_state_t

    ! prognostic variables for preqx solver

    ! prognostics must match those in prim_restart_mod.F90
    ! vertically-lagrangian code advects dp3d instead of ps_v
    ! tracers Q, Qdp always use 2 level time scheme
#ifdef HOMME_USE_FLAT_ARRAYS
    real (kind=real_kind), pointer :: v   (:,:,:,:,:)            ! velocity                           1
    real (kind=real_kind), pointer :: T   (:,:,:,:)              ! temperature                        2
    real (kind=real_kind), pointer :: dp3d(:,:,:,:)              ! delta p on levels                  8
    real (kind=real_kind), pointer :: ps_v(:,:,:)                ! surface pressure                   4
    real (kind=real_kind), pointer :: phis(:,:)                  ! surface geopotential (prescribed)  5
    real (kind=real_kind), pointer :: Q   (:,:,:,:)              ! Tracer concentration               6
    real (kind=real_kind), pointer :: Qdp (:,:,:,:,:)            ! Tracer mass                        7
! #if HOMME_USE_FLAT_ARRAYS
#else
    real (kind=real_kind) :: v   (np,np,2,nlev,timelevels)            ! velocity                           1
    real (kind=real_kind) :: T   (np,np,nlev,timelevels)              ! temperature                        2
    real (kind=real_kind) :: dp3d(np,np,nlev,timelevels)              ! delta p on levels                  8
    real (kind=real_kind) :: ps_v(np,np,timelevels)                   ! surface pressure                   4
    real (kind=real_kind) :: phis(np,np)                              ! surface geopotential (prescribed)  5
    real (kind=real_kind) :: Q   (np,np,nlev,qsize_d)                 ! Tracer concentration               6
    real (kind=real_kind) :: Qdp (np,np,nlev,qsize_d,2)               ! Tracer mass                        7
! #if HOMME_USE_FLAT_ARRAYS
#endif
  end type elem_state_t

  integer(kind=int_kind),public,parameter::StateComponents=8! num prognistics variables (for prim_restart_mod.F90)

  !___________________________________________________________________
  type, public :: derived_state_t

    ! diagnostic variables for preqx solver

    ! storage for subcycling tracers/dynamics

#ifdef HOMME_USE_FLAT_ARRAYS
    real (kind=real_kind), pointer :: vn0  (:,:,:,:)                      ! velocity for SE tracer advection
    real (kind=real_kind), pointer :: vstar(:,:,:,:)                      ! velocity on Lagrangian surfaces
    real (kind=real_kind), pointer :: dpdiss_biharmonic(:,:,:)            ! mean dp dissipation tendency, if nu_p>0
    real (kind=real_kind), pointer :: dpdiss_ave(:,:,:)                   ! mean dp used to compute psdiss_tens

    ! diagnostics for explicit timestep
    real (kind=real_kind), pointer :: phi(:,:,:)                          ! geopotential
    real (kind=real_kind), pointer :: omega_p(:,:,:)                      ! vertical tendency (derived)
    real (kind=real_kind), pointer :: eta_dot_dpdn(:,:,:)                 ! mean vertical flux from dynamics
    real (kind=real_kind), pointer :: eta_dot_dpdn_prescribed(:,:,:)      ! prescribed wind test cases

    ! semi-implicit diagnostics: computed in explict-component, reused in Helmholtz-component.
    real (kind=real_kind), pointer :: zeta(:,:,:)                         ! relative vorticity
    real (kind=real_kind), pointer :: div(:,:,:,:)                        ! divergence

    ! tracer advection fields used for consistency and limiters
    real (kind=real_kind), pointer :: dp(:,:,:)                           ! for dp_tracers at physics timestep
    real (kind=real_kind), pointer :: divdp(:,:,:)                        ! divergence of dp
    real (kind=real_kind), pointer :: divdp_proj(:,:,:)                   ! DSSed divdp
! #if HOMME_USE_FLAT_ARRAYS
#else
    real (kind=real_kind) :: vn0  (np,np,2,nlev)                      ! velocity for SE tracer advection
    real (kind=real_kind) :: vstar(np,np,2,nlev)                      ! velocity on Lagrangian surfaces
    real (kind=real_kind) :: dpdiss_biharmonic(np,np,nlev)            ! mean dp dissipation tendency, if nu_p>0
    real (kind=real_kind) :: dpdiss_ave(np,np,nlev)                   ! mean dp used to compute psdiss_tens

    ! diagnostics for explicit timestep
    real (kind=real_kind) :: phi(np,np,nlev)                          ! geopotential
    real (kind=real_kind) :: omega_p(np,np,nlev)                      ! vertical tendency (derived)
    real (kind=real_kind) :: eta_dot_dpdn(np,np,nlevp)                ! mean vertical flux from dynamics
    real (kind=real_kind) :: eta_dot_dpdn_prescribed(np,np,nlevp)     ! prescribed wind test cases

    ! semi-implicit diagnostics: computed in explict-component, reused in Helmholtz-component.
    real (kind=real_kind) :: zeta(np,np,nlev)                         ! relative vorticity
    real (kind=real_kind) :: div(np,np,nlev,timelevels)               ! divergence

    ! tracer advection fields used for consistency and limiters
    real (kind=real_kind) :: dp(np,np,nlev)                           ! for dp_tracers at physics timestep
    real (kind=real_kind) :: divdp(np,np,nlev)                        ! divergence of dp
    real (kind=real_kind) :: divdp_proj(np,np,nlev)                   ! DSSed divdp
! #if HOMME_USE_FLAT_ARRAYS
#endif

#ifdef CAM
    ! forcing terms for CAM
    real (kind=real_kind) :: FQ(np,np,nlev,qsize_d)                   ! tracer forcing
    real (kind=real_kind) :: FM(np,np,2,nlev)                         ! momentum forcing
    real (kind=real_kind) :: FT(np,np,nlev)                           ! temperature forcing
    real (kind=real_kind) :: etadot_prescribed(np,np,nlevp)           ! prescribed vertical tendency
    real (kind=real_kind) :: u_met(np,np,nlev)                        ! zonal component of prescribed meteorology winds
    real (kind=real_kind) :: dudt_met(np,np,nlev)                     ! rate of change of zonal component of prescribed meteorology winds
    real (kind=real_kind) :: v_met(np,np,nlev)                        ! meridional component of prescribed meteorology winds
    real (kind=real_kind) :: dvdt_met(np,np,nlev)                     ! rate of change of meridional component of prescribed meteorology winds
    real (kind=real_kind) :: T_met(np,np,nlev)                        ! prescribed meteorology temperature
    real (kind=real_kind) :: dTdt_met(np,np,nlev)                     ! rate of change of prescribed meteorology temperature
    real (kind=real_kind) :: ps_met(np,np)                            ! surface pressure of prescribed meteorology
    real (kind=real_kind) :: dpsdt_met(np,np)                         ! rate of change of surface pressure of prescribed meteorology
    real (kind=real_kind) :: nudge_factor(np,np,nlev)                 ! nudging factor (prescribed)
    real (kind=real_kind) :: Utnd(npsq,nlev)                          ! accumulated U tendency due to nudging towards prescribed met
    real (kind=real_kind) :: Vtnd(npsq,nlev)                          ! accumulated V tendency due to nudging towards prescribed met
    real (kind=real_kind) :: Ttnd(npsq,nlev)                          ! accumulated T tendency due to nudging towards prescribed met
! #ifdef CAM
#else
#ifdef HOMME_USE_FLAT_ARRAYS
    ! forcing terms for HOMME
    real (kind=real_kind), pointer :: FQ(:,:,:,:)                     ! tracer forcing
    real (kind=real_kind), pointer :: FM(:,:,:,:)                     ! momentum forcing
    real (kind=real_kind), pointer :: FT(:,:,:)                       ! temperature forcing
#else
    ! forcing terms for HOMME
    real (kind=real_kind) :: FQ(np,np,nlev,qsize_d)                   ! tracer forcing
    real (kind=real_kind) :: FM(np,np,2,nlev)                         ! momentum forcing
    real (kind=real_kind) :: FT(np,np,nlev)                           ! temperature forcing
#endif
! #ifdef CAM
#endif

    ! forcing terms for both CAM and HOMME
    ! FQps for conserving dry mass in the presence of precipitation

#ifdef HOMME_USE_FLAT_ARRAYS
    real (kind=real_kind), pointer :: FQps(:,:,:)                     ! forcing of FQ on ps_v
#else
    real (kind=real_kind) :: FQps(np,np,timelevels)                   ! forcing of FQ on ps_v
#endif

  end type derived_state_t

!ending USE_OPENACC if
#endif

  !___________________________________________________________________
  type, public :: elem_accum_t

#ifdef ENERGY_DIAGNOSTICS

    ! Energy equation:
    ! KE_t  = T1 + T2  + D1   + Err   +  vertical & horizontal advection terms
    ! IE_t  = S1 + D2                 +  vertical & horizontal advection terms
    ! PE_t  = S2
    !
    ! KEvert*  =  KE net vertical advection    (should be zero)
    ! KEhoriz* =  KE net horizonatl advection  (should be zero)
    ! IEvert*  =  IE net vertical advection    (should be zero)
    ! IEhoriz* =  IE net horizonatl advection  (should be zero)
    !
    ! With leapfrog, energy equations are all exact except KE
    ! (has an Err term that goes to zero as dt**2)
    !
    ! Transfer terms:
    ! T1   = -< dp/dn u, RT_v/p grad_p >     KE<->IE:   T1 + T2-T2_s = S1
    ! T2   = -< dp/dn u, grad_phi >          KE<->PE:   T2_s         = S2
    ! T2_s = -< dp/dn u, grad_phis >
    ! S1   = < Cp_star dp/dn , RT omega_p/Cp_star >
    ! S2   = -< div (u dp/dn), phis >

#ifdef HOMME_USE_FLAT_ARRAYS
    real (kind=real_kind), pointer :: KEvert1(:,:)                           ! term from continuity equ
    real (kind=real_kind), pointer :: KEvert2(:,:)                           ! term from momentum equ
    real (kind=real_kind), pointer :: IEvert1(:,:)                           ! term from continuity equ
    real (kind=real_kind), pointer :: IEvert2(:,:)                           ! term from T equ
    real (kind=real_kind), pointer :: IEvert1_wet(:,:)                       ! wet term from continuity equ
    real (kind=real_kind), pointer :: IEvert2_wet(:,:)                       ! wet term from T equ

    real (kind=real_kind), pointer :: KEhorz1(:,:)                           ! at time t
    real (kind=real_kind), pointer :: KEhorz2(:,:)                           ! after calling time_advance, these will be at time t-1
    real (kind=real_kind), pointer :: IEhorz1(:,:)
    real (kind=real_kind), pointer :: IEhorz2(:,:)
    real (kind=real_kind), pointer :: IEhorz1_wet(:,:)
    real (kind=real_kind), pointer :: IEhorz2_wet(:,:)

    real (kind=real_kind), pointer :: T1(:,:)
    real (kind=real_kind), pointer :: T2(:,:)
    real (kind=real_kind), pointer :: T2_s(:,:)
    real (kind=real_kind), pointer :: S1(:,:)
    real (kind=real_kind), pointer :: S1_wet(:,:)
    real (kind=real_kind), pointer :: S2(:,:)

    ! the KE conversion term and diffusion term
    real (kind=real_kind), pointer :: DIFF(:,:,:,:)                       ! net hypervis term
    real (kind=real_kind), pointer :: DIFFT(:,:,:)                        ! net hypervis term
    real (kind=real_kind), pointer :: CONV(:,:,:,:)                       ! dpdn u dot CONV = T1 + T2
! #if HOMME_USE_FLAT_ARRAYS
#else
    real (kind=real_kind) :: KEvert1(np,np)                           ! term from continuity equ
    real (kind=real_kind) :: KEvert2(np,np)                           ! term from momentum equ
    real (kind=real_kind) :: IEvert1(np,np)                           ! term from continuity equ
    real (kind=real_kind) :: IEvert2(np,np)                           ! term from T equ
    real (kind=real_kind) :: IEvert1_wet(np,np)                       ! wet term from continuity equ
    real (kind=real_kind) :: IEvert2_wet(np,np)                       ! wet term from T equ

    real (kind=real_kind) :: KEhorz1(np,np)                           ! at time t
    real (kind=real_kind) :: KEhorz2(np,np)                           ! after calling time_advance, these will be at time t-1
    real (kind=real_kind) :: IEhorz1(np,np)
    real (kind=real_kind) :: IEhorz2(np,np)
    real (kind=real_kind) :: IEhorz1_wet(np,np)
    real (kind=real_kind) :: IEhorz2_wet(np,np)

    real (kind=real_kind) :: T1(np,np)
    real (kind=real_kind) :: T2(np,np)
    real (kind=real_kind) :: T2_s(np,np)
    real (kind=real_kind) :: S1(np,np)
    real (kind=real_kind) :: S1_wet(np,np)
    real (kind=real_kind) :: S2(np,np)

    ! the KE conversion term and diffusion term
    real (kind=real_kind) :: DIFF(np,np,2,nlev)                       ! net hypervis term
    real (kind=real_kind) :: DIFFT(np,np,nlev)                        ! net hypervis term
    real (kind=real_kind) :: CONV(np,np,2,nlev)                       ! dpdn u dot CONV = T1 + T2
! #if HOMME_USE_FLAT_ARRAYS
#endif
! ENERGY_DIAGNOSTICS
#endif

    ! the "4" timelevels represents data computed at:
    !  1  t-.5
    !  2  t+.5   after dynamics
    !  3  t+.5   after forcing
    !  4  t+.5   after Robert
    ! after calling TimeLevelUpdate, all times above decrease by 1.0

#ifdef HOMME_USE_FLAT_ARRAYS
    real (kind=real_kind), pointer :: KEner(:,:,:)
    real (kind=real_kind), pointer :: PEner(:,:,:)
    real (kind=real_kind), pointer :: IEner(:,:,:)
    real (kind=real_kind), pointer :: IEner_wet(:,:,:)
    real (kind=real_kind), pointer :: Qvar(:,:,:,:)                    ! Q variance at half time levels
    real (kind=real_kind), pointer :: Qmass(:,:,:,:)                   ! Q mass at half time levels
    real (kind=real_kind), pointer :: Q1mass(:,:,:)                    ! Q mass at full time levels
#else
    real (kind=real_kind) :: KEner(np,np,4)
    real (kind=real_kind) :: PEner(np,np,4)
    real (kind=real_kind) :: IEner(np,np,4)
    real (kind=real_kind) :: IEner_wet(np,np,4)
    real (kind=real_kind) :: Qvar(np,np,qsize_d,4)                    ! Q variance at half time levels
    real (kind=real_kind) :: Qmass(np,np,qsize_d,4)                   ! Q mass at half time levels
    real (kind=real_kind) :: Q1mass(np,np,qsize_d)                    ! Q mass at full time levels
#endif
  end type elem_accum_t

! if not _PRIM
#else
! ================== SHALLOW-WATER DATA-STRUCTURES ===================

  public :: setup_element_pointers_sw
  real (kind=real_kind), allocatable, target, public :: elem_state_p (:,:,:,:,:)    ! (np,np,nlevel,timelevels,nelemd)
  real (kind=real_kind), allocatable, target, public :: elem_state_ps (:,:,:)       ! (np,np,nelemd)
  real (kind=real_kind), allocatable, target, public :: elem_state_v  (:,:,:,:,:,:) ! (np,np,2,nlev,timelevels,nelemd)


  type, public :: elem_state_t

    ! prognostic variables for shallow-water solver
     real (kind=real_kind) :: gradps(np,np,2)                         ! gradient of surface geopotential
#ifdef HOMME_USE_FLAT_ARRAYS
     real (kind=real_kind), pointer   :: p(:,:,:,:)
     real (kind=real_kind), pointer   :: ps(:,:)                      ! surface geopotential
     real (kind=real_kind), pointer   :: v(:,:,:,:,:)                 ! contravarient comp
#else
     real (kind=real_kind) :: p(np,np,nlev,timelevels)
     real (kind=real_kind) :: ps(np,np)                               ! surface geopotential
     real (kind=real_kind) :: v(np,np,2,nlev,timelevels)              ! contravarient comp
#endif

  end type elem_state_t

  !___________________________________________________________________
  type, public :: derived_state_t
     real (kind=real_kind) :: vstar(np,np,2,nlev)                     ! velocity on Lagrangian surfaces
  end type derived_state_t

! #ifdef _PRIM
#endif

! ============= DATA-STRUCTURES COMMON TO ALL SOLVERS ================

  type, public :: index_t
     integer(kind=int_kind) :: ia(npsq),ja(npsq)
     integer(kind=int_kind) :: is,ie
     integer(kind=int_kind) :: NumUniquePts
     integer(kind=int_kind) :: UniquePtOffset
  end type index_t

  !___________________________________________________________________
  type, public :: element_t
     integer(kind=int_kind) :: LocalId
     integer(kind=int_kind) :: GlobalId

     ! Equ-angular gnomonic projection coordinates
     type (cartesian2D_t)     :: cartp(np,np)                         ! gnomonic coords of GLL points
     type (cartesian2D_t)     :: corners(4)                           ! gnomonic coords of element corners

     ! 3D cartesian coordinates
     type (cartesian3D_t)     :: corners3D(4)

     ! Element diagnostics
     real (kind=real_kind)    :: area                                 ! Area of element
     real (kind=real_kind)    :: normDinv                             ! some type of norm of Dinv used for CFL
     real (kind=real_kind)    :: dx_short                             ! short length scale in km
     real (kind=real_kind)    :: dx_long                              ! long length scale in km

     real (kind=real_kind)    :: variable_hyperviscosity(np,np)       ! hyperviscosity based on above
     real (kind=real_kind)    :: hv_courant                           ! hyperviscosity courant number
#ifdef HOMME_USE_FLAT_ARRAYS
     real (kind=real_kind), pointer    :: tensorVisc(:,:,:,:)
#else
     real (kind=real_kind)    :: tensorVisc(np,np,2,2)                !og, matrix V for tensor viscosity
#endif

     ! Edge connectivity information
!     integer(kind=int_kind)   :: node_numbers(4)
!     integer(kind=int_kind)   :: node_multiplicity(4)                 ! number of elements sharing corner node

     type (GridVertex_t)      :: vertex                               ! element grid vertex information
     type (EdgeDescriptor_t)  :: desc

     type (elem_state_t)      :: state

     type (derived_state_t)   :: derived
#if defined _PRIM
     type (elem_accum_t)       :: accum
#endif
     ! Metric terms
     real (kind=real_kind)    :: met(np,np,2,2)                       ! metric tensor on velocity and pressure grid

#ifdef HOMME_USE_FLAT_ARRAYS
     real (kind=real_kind), pointer    :: metinv(:,:,:,:)
     real (kind=real_kind), pointer    :: metdet(:,:)
     real (kind=real_kind), pointer    :: rmetdet(:,:)
     real (kind=real_kind), pointer    :: Dinv(:,:,:,:)
     real (kind=real_kind), pointer    :: D(:,:,:,:)
     real (kind=real_kind), pointer    :: spheremp(:,:)
     real (kind=real_kind), pointer    :: rspheremp(:,:)
     real (kind=real_kind), pointer    :: fcor(:,:)

     real (kind=real_kind), pointer    :: mp(:,:)                            ! mass matrix on v and p grid
     real (kind=real_kind), pointer    :: rmp(:,:)                           ! inverse mass matrix on v and p grid

     type (spherical_polar_t), pointer :: spherep(:,:)                       ! Spherical coords of GLL points

     real (kind=real_kind), pointer    :: sub_elem_mass_flux(:,:,:,:)
     real (kind=real_kind), pointer    :: vec_sphere2cart(:,:,:,:)
#else
     real (kind=real_kind)    :: metinv(np,np,2,2)
     real (kind=real_kind)    :: metdet(np,np)                        ! g = SQRT(det(g_ij)) on velocity and pressure grid
     real (kind=real_kind)    :: rmetdet(np,np)                       ! 1/metdet on velocity pressure grid
     real (kind=real_kind)    :: Dinv(np,np,2,2)                      ! Map vector field on the sphere to covariant v on cube
     real (kind=real_kind)    :: D(np,np,2,2)                         ! Map covariant field on cube to vector field on the sphere

     ! Mass matrix terms for an element on a cube face
     real (kind=real_kind)    :: mp(np,np)                            ! mass matrix on v and p grid
     real (kind=real_kind)    :: rmp(np,np)                           ! inverse mass matrix on v and p grid

     ! Mass matrix terms for an element on the sphere
     ! This mass matrix is used when solving the equations in weak form
     ! with the natural (surface area of the sphere) inner product

     ! Coordinate values of element points
     type (spherical_polar_t) :: spherep(np,np)                       ! Spherical coords of GLL points

     real (kind=real_kind)    :: spheremp(np,np)                      ! mass matrix on v and p grid
     real (kind=real_kind)    :: rspheremp(np,np)                     ! inverse mass matrix on v and p grid
     real (kind=real_kind)    :: fcor(np,np)                          ! Coreolis term

     ! Mass flux across the sides of each sub-element.
     ! The storage is redundent since the mass across shared sides
     ! must be equal in magnitude and opposite in sign.
     ! The layout is like:
     !   --------------------------------------------------------------
     ! ^|    (1,4,3)     |                |              |    (4,4,3) |
     ! ||                |                |              |            |
     ! ||(1,4,4)         |                |              |(4,4,4)     |
     ! ||         (1,4,2)|                |              |     (4,4,2)|
     ! ||                |                |              |            |
     ! ||   (1,4,1)      |                |              |  (4,4,1)   |
     ! |---------------------------------------------------------------
     ! S|                |                |              |            |
     ! e|                |                |              |            |
     ! c|                |                |              |            |
     ! o|                |                |              |            |
     ! n|                |                |              |            |
     ! d|                |                |              |            |
     !  ---------------------------------------------------------------
     ! C|                |                |              |            |
     ! o|                |                |              |            |
     ! o|                |                |              |            |
     ! r|                |                |              |            |
     ! d|                |                |              |            |
     ! i|                |                |              |            |
     ! n---------------------------------------------------------------
     ! a|    (1,1,3)     |                |              |    (4,1,3) |
     ! t|                |                |              |(4,1,4)     |
     ! e|(1,1,4)         |                |              |            |
     !  |         (1,1,2)|                |              |     (4,1,2)|
     !  |                |                |              |            |
     !  |    (1,1,1)     |                |              |  (4,1,1)   |
     !  ---------------------------------------------------------------
     !          First Coordinate ------->

     ! real (kind=real_kind) :: sub_elem_mass_flux(nc,nc,4,nlev)

     ! Convert vector fields from spherical to rectangular components
     ! The transpose of this operation is its pseudoinverse.
     real (kind=real_kind)    :: vec_sphere2cart(np,np,3,2)
#endif

     integer(kind=long_kind)  :: gdofP(np,np)                         ! global degree of freedom (P-grid)

     type (index_t) :: idxP
     type (index_t),pointer :: idxV
     integer :: FaceNum

     ! force element_t to be a multiple of 8 bytes.
     ! on BGP, code will crash (signal 7, or signal 15) if 8 byte alignment is off
     ! check core file for:
     ! core.63:Generated by interrupt..(Alignment Exception DEAR=0xa1ef671c ESR=0x01800000 CCR0=0x4800a002)
     integer :: dummy
  end type element_t

  !___________________________________________________________________
  public :: element_coordinates
  public :: element_var_coordinates
  public :: element_var_coordinates3D
  public :: GetColumnIdP,GetColumnIdV
  public :: allocate_element_desc
  public :: PrintElem

contains

  subroutine PrintElem(arr)

    real(kind=real_kind) :: arr(:,:)
    integer :: i,j

      do j=np,1,-1
         write(6,*) (arr(i,j), i=1,np)
      enddo

  end subroutine PrintElem
! ===================== ELEMENT_MOD METHODS ==========================

  function GetColumnIdP(elem,i,j) result(col_id)

    ! Get unique identifier for a Physics column on the P-grid

    type(element_t), intent(in) :: elem
    integer, intent(in) :: i,j
    integer :: col_id
    col_id = elem%gdofP(i,j)
  end function GetColumnIdP

  !___________________________________________________________________
  function GetColumnIdV(elem,i,j) result(col_id)

    !  Get unique identifier for a Physics column on the V-grid

    type(element_t), intent(in) :: elem
    integer, intent(in) :: i,j
    integer :: col_id
    col_id = elem%gdofP(i,j)
  end function GetColumnIdV

  !___________________________________________________________________
  function element_coordinates(start,end,points) result(cart)

    ! Initialize 2D rectilinear element colocation points

    use kinds, only : longdouble_kind
    type (cartesian2D_t), intent(in) :: start
    type (cartesian2D_t), intent(in) :: end
    real (kind=longdouble_kind), intent(in) :: points(:)

    type (cartesian2D_t) :: cart(SIZE(points),SIZE(points))
    type (cartesian2D_t) :: length, centroid
    real (kind=longdouble_kind) :: y
    integer i,j

    length%x   = 0.50D0*(end%x-start%x)
    length%y   = 0.50D0*(end%y-start%y)
    centroid%x = 0.50D0*(end%x+start%x)
    centroid%y = 0.50D0*(end%y+start%y)
    do j=1,SIZE(points)
       y = centroid%y + length%y*points(j)
       do i=1,SIZE(points)
          cart(i,j)%x = centroid%x + length%x*points(i)
          cart(i,j)%y = y
       end do
    end do
  end function element_coordinates

  !___________________________________________________________________
  function element_var_coordinates(c,points) result(cart)

    use kinds, only : longdouble_kind
    type (cartesian2D_t), intent(in) :: c(4)
    real (kind=longdouble_kind), intent(in) :: points(:)
    type (cartesian2D_t) :: cart(SIZE(points),SIZE(points))

    real (kind=longdouble_kind) :: p(size(points))
    real (kind=longdouble_kind) :: q(size(points))
    integer i,j

    p(:) = (1.0D0-points(:))/2.0D0
    q(:) = (1.0D0+points(:))/2.0D0

    do j=1,SIZE(points)
       do i=1,SIZE(points)
          cart(i,j)%x = p(i)*p(j)*c(1)%x &
                      + q(i)*p(j)*c(2)%x &
                      + q(i)*q(j)*c(3)%x &
                      + p(i)*q(j)*c(4)%x
          cart(i,j)%y = p(i)*p(j)*c(1)%y &
                      + q(i)*p(j)*c(2)%y &
                      + q(i)*q(j)*c(3)%y &
                      + p(i)*q(j)*c(4)%y
       end do
    end do
  end function element_var_coordinates

  !___________________________________________________________________
  function element_var_coordinates3d(c,points) result(cart)

    use kinds, only : longdouble_kind
    type (cartesian3D_t), intent(in) :: c(4)
    real (kind=longdouble_kind), intent(in) :: points(:)
    type (cartesian3D_t) :: cart(SIZE(points),SIZE(points))

    real (kind=longdouble_kind) :: p(size(points))
    real (kind=longdouble_kind) :: q(size(points)),r
    integer i,j

    p(:) = (1.0D0-points(:))/2.0D0
    q(:) = (1.0D0+points(:))/2.0D0

    do j=1,SIZE(points)
       do i=1,SIZE(points)
          cart(i,j)%x = p(i)*p(j)*c(1)%x &
                      + q(i)*p(j)*c(2)%x &
                      + q(i)*q(j)*c(3)%x &
                      + p(i)*q(j)*c(4)%x
          cart(i,j)%y = p(i)*p(j)*c(1)%y &
                      + q(i)*p(j)*c(2)%y &
                      + q(i)*q(j)*c(3)%y &
                      + p(i)*q(j)*c(4)%y
          cart(i,j)%z = p(i)*p(j)*c(1)%z &
                      + q(i)*p(j)*c(2)%z &
                      + q(i)*q(j)*c(3)%z &
                      + p(i)*q(j)*c(4)%z

          ! project back to sphere:
          r = distance(cart(i,j))
          cart(i,j)%x = cart(i,j)%x/r
          cart(i,j)%y = cart(i,j)%y/r
          cart(i,j)%z = cart(i,j)%z/r
       end do
    end do
  end function element_var_coordinates3d

  !___________________________________________________________________
  subroutine allocate_element_desc(elem)

    type (element_t), intent(inout)   :: elem(:)
    integer                           :: num, j,i

    num = SIZE(elem)

    do j=1,num
       do i=1,max_neigh_edges
          elem(j)%desc%loc2buf(i)=i
          elem(j)%desc%globalID(i)=-1
       enddo

    end do
  end subroutine allocate_element_desc


#ifdef _PRIM
  !___________________________________________________________________
  subroutine setup_element_pointers(elem)
    use dimensions_mod, only: nelemd, qsize
    implicit none
    type(element_t), intent(inout) :: elem(:)
#if USE_OPENACC
    integer :: ie
    allocate( state_Qdp                (np,np,nlev,qsize,2,nelemd)            )
    allocate( derived_vn0              (np,np,2,nlev,nelemd)                  )
    allocate( derived_divdp            (np,np,nlev,nelemd)                    )
    allocate( derived_divdp_proj       (np,np,nlev,nelemd)                    )
    do ie = 1 , nelemd
      elem(ie)%state%Qdp                 => state_Qdp                (:,:,:,:,:,ie)
      elem(ie)%derived%vn0               => derived_vn0              (:,:,:,:,ie)
      elem(ie)%derived%divdp             => derived_divdp            (:,:,:,ie)
      elem(ie)%derived%divdp_proj        => derived_divdp_proj       (:,:,:,ie)
    enddo
#else
#ifdef HOMME_USE_FLAT_ARRAYS
    integer :: ie

    allocate( elem_D                   (np,np,2,2,nelemd) )
    allocate( elem_Dinv                (np,np,2,2,nelemd) )
    allocate( elem_metdet              (np,np,nelemd)     )
    allocate( elem_rmetdet             (np,np,nelemd)     )
    allocate( elem_metinv              (np,np,2,2,nelemd) )
    allocate( elem_fcor                (np,np, nelemd)    )
    allocate( elem_mp                  (np,np, nelemd)    )
    allocate( elem_rmp                 (np,np, nelemd)    )
    allocate( elem_spherep             (np,np, nelemd)    )
    allocate( elem_spheremp            (np,np, nelemd)    )
    allocate( elem_rspheremp           (np,np, nelemd)    )
    ! allocate( elem_sub_elem_mass_flux  (nc,nc,4,nlev,nelemd))
    ! allocate( elem_vec_sph2cart     (np,np,3,2,nelemd) )
    allocate( elem_vec_sphere2cart     (np,np,3,2,nelemd) )
    allocate( elem_tensorVisc          (np,np,2,2,nelemd) )

    do ie = 1 , nelemd
      elem(ie)%D                      => elem_D(:,:,:,:,ie)
      elem(ie)%metinv                 => elem_metinv(:,:,:,:,ie)
      elem(ie)%Dinv                   => elem_Dinv(:,:,:,:,ie)
      elem(ie)%metdet                 => elem_metdet(:,:,ie)
      elem(ie)%rmetdet                => elem_rmetdet(:,:,ie)
      elem(ie)%fcor                   => elem_fcor(:,:,ie)
      elem(ie)%mp                     => elem_mp(:,:,ie)
      elem(ie)%rmp                    => elem_rmp(:,:,ie)
      elem(ie)%spherep                => elem_spherep(:,:,ie)
      elem(ie)%spheremp               => elem_spheremp(:,:,ie)
      elem(ie)%rspheremp              => elem_rspheremp(:,:,ie)
      elem(ie)%sub_elem_mass_flux     => elem_sub_elem_mass_flux(:,:,:,:,ie)
      elem(ie)%vec_sphere2cart        => elem_vec_sph2cart(:,:,:,:,ie)
      elem(ie)%tensorVisc             => elem_tensorvisc(:,:,:,:,ie)
    enddo

    ! elem_state_t arrays
    allocate( elem_state_v    (np,np,2,nlev,timelevels,nelemd) )
    allocate( elem_state_Temp (np,np,nlev,timelevels,nelemd)   )
    allocate( elem_state_dp3d (np,np,nlev,timelevels,nelemd)   )
    allocate( elem_state_ps_v (np,np,timelevels,nelemd)        )
    allocate( elem_state_phis (np,np,nelemd)                   )
    allocate( elem_state_Q    (np,np,nlev,qsize_d,nelemd)      )
    allocate( elem_state_Qdp  (np,np,nlev,qsize_d,2,nelemd)    )

    do ie = 1 , nelemd
      elem(ie)%state%v        => elem_state_v    (:,:,:,:,:,ie)
      elem(ie)%state%T        => elem_state_Temp (:,:,:,:,ie)
      elem(ie)%state%dp3d     => elem_state_dp3d (:,:,:,:,ie)
      elem(ie)%state%ps_v     => elem_state_ps_v (:,:,:,ie)
      elem(ie)%state%phis     => elem_state_phis (:,:,ie)
      elem(ie)%state%Q        => elem_state_Q    (:,:,:,:,ie)
      elem(ie)%state%Qdp      => elem_state_Qdp  (:,:,:,:,:,ie)
    enddo

    ! derived_state_t arrays
    allocate( elem_derived_vn0              (np,np,2,nlev,nelemd)                  )
    allocate( elem_derived_vstar            (np,np,2,nlev,nelemd)                  )
    allocate( elem_derived_dpdiss_biharmonic(np,np,nlev,nelemd)                    )
    allocate( elem_derived_dpdiss_ave       (np,np,nlev,nelemd)                    )
    allocate( elem_derived_phi              (np,np,nlev,nelemd)                    )
    allocate( elem_derived_omega_p          (np,np,nlev,nelemd)                    )
    allocate( elem_derived_eta_dot_dpdn     (np,np,nlevp,nelemd)                   )
    allocate( elem_derived_eta_dot_dpdn_prescribed     (np,np,nlevp,nelemd)                   )
    allocate( elem_derived_zeta             (np,np,nlev,nelemd)                    )
    allocate( elem_derived_div              (np,np,nlev,timelevels,nelemd)         )
    allocate( elem_derived_dp               (np,np,nlev,nelemd)                    )
    allocate( elem_derived_divdp            (np,np,nlev,nelemd)                    )
    allocate( elem_derived_divdp_proj       (np,np,nlev,nelemd)                    )
#ifndef USE_KOKKOS_KERNELS
    allocate( elem_derived_FQ               (np,np,nlev,qsize_d,nelemd) )
#endif
    allocate( elem_derived_FM               (np,np,2,nlev,nelemd)       )
    allocate( elem_derived_FT               (np,np,nlev,nelemd)         )
    allocate( elem_derived_FQps             (np,np,timelevels,nelemd)              )

    do ie = 1 , nelemd
      elem(ie)%derived%vn0                  => elem_derived_vn0              (:,:,:,:,ie)
      elem(ie)%derived%vstar                => elem_derived_vstar            (:,:,:,:,ie)
      elem(ie)%derived%dpdiss_biharmonic    => elem_derived_dpdiss_biharmonic(:,:,:,ie)
      elem(ie)%derived%dpdiss_ave           => elem_derived_dpdiss_ave       (:,:,:,ie)
      elem(ie)%derived%phi                  => elem_derived_phi              (:,:,:,ie)
      elem(ie)%derived%omega_p              => elem_derived_omega_p          (:,:,:,ie)
      elem(ie)%derived%eta_dot_dpdn         => elem_derived_eta_dot_dpdn     (:,:,:,ie)
      elem(ie)%derived%eta_dot_dpdn_prescribed=>elem_derived_eta_dot_dpdn_prescribed(:,:,:,ie)
      elem(ie)%derived%zeta                 => elem_derived_zeta             (:,:,:,ie)
      elem(ie)%derived%div                  => elem_derived_div              (:,:,:,:,ie)
      elem(ie)%derived%dp                   => elem_derived_dp               (:,:,:,ie)
      elem(ie)%derived%divdp                => elem_derived_divdp            (:,:,:,ie)
      elem(ie)%derived%divdp_proj           => elem_derived_divdp_proj       (:,:,:,ie)
      elem(ie)%derived%FQ                   => elem_derived_FQ               (:,:,:,:,ie)
      elem(ie)%derived%FM                   => elem_derived_FM               (:,:,:,:,ie)
      elem(ie)%derived%FT                   => elem_derived_FT               (:,:,:,ie)
      elem(ie)%derived%FQps                 => elem_derived_FQps             (:,:,:,ie)
    enddo

    ! elem_accum_t arrays
    allocate( elem_accum_KEvert1    (np,np,nelemd)           )
    allocate( elem_accum_KEvert2    (np,np,nelemd)           )
    allocate( elem_accum_IEvert1    (np,np,nelemd)           )
    allocate( elem_accum_IEvert2    (np,np,nelemd)           )
    allocate( elem_accum_IEvert1_wet(np,np,nelemd)           )
    allocate( elem_accum_IEvert2_wet(np,np,nelemd)           )
    allocate( elem_accum_KEhorz1    (np,np,nelemd)           )
    allocate( elem_accum_KEhorz2    (np,np,nelemd)           )
    allocate( elem_accum_IEhorz1    (np,np,nelemd)           )
    allocate( elem_accum_IEhorz2    (np,np,nelemd)           )
    allocate( elem_accum_IEhorz1_wet(np,np,nelemd)           )
    allocate( elem_accum_IEhorz2_wet(np,np,nelemd)           )
    allocate( elem_accum_T1         (np,np,nelemd)           )
    allocate( elem_accum_T2         (np,np,nelemd)           )
    allocate( elem_accum_T2_s       (np,np,nelemd)           )
    allocate( elem_accum_S1         (np,np,nelemd)           )
    allocate( elem_accum_S1_wet     (np,np,nelemd)           )
    allocate( elem_accum_S2         (np,np,nelemd)           )
    allocate( elem_accum_DIFF       (np,np,2,nlev,nelemd)    )
    allocate( elem_accum_DIFFT      (np,np,nlev,nelemd)      )
    allocate( elem_accum_CONV       (np,np,2,nlev,nelemd)    )
    allocate( elem_accum_KEner      (np,np,4,nelemd)         )
    allocate( elem_accum_PEner      (np,np,4,nelemd)         )
    allocate( elem_accum_IEner      (np,np,4,nelemd)         )
    allocate( elem_accum_IEner_wet  (np,np,4,nelemd)         )
    allocate( elem_accum_Qvar       (np,np,qsize_d,4,nelemd) )
    allocate( elem_accum_Qmass      (np,np,qsize_d,4,nelemd) )
    allocate( elem_accum_Q1mass     (np,np,qsize_d,nelemd)   )

    do ie = 1 , nelemd
#ifdef ENERGY_DIAGNOSTICS
      elem(ie)%accum%KEvert1        => elem_accum_KEvert1    (:,:,ie)
      elem(ie)%accum%KEvert2        => elem_accum_KEvert2    (:,:,ie)
      elem(ie)%accum%IEvert1        => elem_accum_IEvert1    (:,:,ie)
      elem(ie)%accum%IEvert2        => elem_accum_IEvert2    (:,:,ie)
      elem(ie)%accum%IEvert1_wet    => elem_accum_IEvert1_wet(:,:,ie)
      elem(ie)%accum%IEvert2_wet    => elem_accum_IEvert2_wet(:,:,ie)
      elem(ie)%accum%KEhorz1        => elem_accum_KEhorz1    (:,:,ie)
      elem(ie)%accum%KEhorz2        => elem_accum_KEhorz2    (:,:,ie)
      elem(ie)%accum%IEhorz1        => elem_accum_IEhorz1    (:,:,ie)
      elem(ie)%accum%IEhorz2        => elem_accum_IEhorz2    (:,:,ie)
      elem(ie)%accum%IEhorz1_wet    => elem_accum_IEhorz1_wet(:,:,ie)
      elem(ie)%accum%IEhorz2_wet    => elem_accum_IEhorz2_wet(:,:,ie)
      elem(ie)%accum%T1             => elem_accum_T1         (:,:,ie)
      elem(ie)%accum%T2             => elem_accum_T2         (:,:,ie)
      elem(ie)%accum%T2_s           => elem_accum_T2_s       (:,:,ie)
      elem(ie)%accum%S1             => elem_accum_S1         (:,:,ie)
      elem(ie)%accum%S1_wet         => elem_accum_S1_wet     (:,:,ie)
      elem(ie)%accum%S2             => elem_accum_S2         (:,:,ie)
      elem(ie)%accum%DIFF           => elem_accum_DIFF       (:,:,:,:,ie)
      elem(ie)%accum%DIFFT          => elem_accum_DIFFT      (:,:,:,ie)
      elem(ie)%accum%CONV           => elem_accum_CONV       (:,:,:,:,ie)
#endif
      elem(ie)%accum%KEner          => elem_accum_KEner      (:,:,:,ie)
      elem(ie)%accum%PEner          => elem_accum_PEner      (:,:,:,ie)
      elem(ie)%accum%IEner          => elem_accum_IEner      (:,:,:,ie)
      elem(ie)%accum%IEner_wet      => elem_accum_IEner_wet  (:,:,:,ie)
      elem(ie)%accum%Qvar           => elem_accum_Qvar       (:,:,:,:,ie)
      elem(ie)%accum%Qmass          => elem_accum_Qmass      (:,:,:,:,ie)
      elem(ie)%accum%Q1mass         => elem_accum_Q1mass     (:,:,:,ie)
    enddo
! #if HOMME_USE_FLAT_ARRAYS
#endif
! #if USE_OPENACC
#endif
  end subroutine setup_element_pointers

! #ifdef _PRIM
#else

  !___________________________________________________________________
  subroutine setup_element_pointers_sw(elem)
    use dimensions_mod, only: nelemd, qsize
    implicit none
    type(element_t), intent(inout) :: elem(:)
#ifdef HOMME_USE_FLAT_ARRAYS
    integer :: ie

    allocate( elem_metdet              (np,np,nelemd)            )
    allocate( elem_rmetdet             (np,np,nelemd)            )
    allocate( elem_Dinv                (np,np,2,2,nelemd)        )
    allocate( elem_D                   (np,np,2,2,nelemd)        )
    allocate( elem_metinv              (np,np,2,2,nelemd)        )
    allocate( elem_state_p             (np,np,nlev,timelevels,nelemd) )
    allocate( elem_state_ps            (np,np,nelemd) )
    allocate( elem_state_v             (np,np,2,nlev,timelevels,nelemd) )
    allocate( elem_mp                  (np,np, nelemd)           )
    allocate( elem_rmp                 (np,np, nelemd)           )
    allocate( elem_spheremp            (np,np, nelemd)           )
    allocate( elem_rspheremp           (np,np, nelemd)          )
    allocate( elem_spherep             (np,np, nelemd)          )
    allocate( elem_fcor                (np,np, nelemd)          )
    allocate( elem_vec_sph2cart        (np,np,3,2,nelemd)        )
    allocate( elem_tensorvisc          (np,np,2,2,nelemd)        )
    do ie = 1 , nelemd
      elem(ie)%metdet                 => elem_metdet(:,:,ie)
      elem(ie)%rmetdet                => elem_rmetdet(:,:,ie)
      elem(ie)%Dinv                   => elem_Dinv(:,:,:,:,ie)
      elem(ie)%D                      => elem_D(:,:,:,:,ie)
      elem(ie)%metinv                 => elem_metinv(:,:,:,:,ie)
      elem(ie)%state%p                => elem_state_p(:,:,:,:,ie)
      elem(ie)%state%ps               => elem_state_ps(:,:,ie)
      elem(ie)%state%v                => elem_state_v(:,:,:,:,:,ie)
      elem(ie)%mp                     => elem_mp(:,:,ie)
      elem(ie)%rmp                    => elem_rmp(:,:,ie)
      elem(ie)%spheremp               => elem_spheremp(:,:,ie)
      elem(ie)%rspheremp              => elem_rspheremp(:,:,ie)
      elem(ie)%spherep                => elem_spherep(:,:,ie)
      elem(ie)%fcor                   => elem_fcor(:,:,ie)
      elem(ie)%vec_sphere2cart        => elem_vec_sph2cart(:,:,:,:,ie)
      elem(ie)%tensorvisc             => elem_tensorvisc(:,:,:,:,ie)
    enddo
#endif
  end subroutine setup_element_pointers_sw

! #ifdef _PRIM
#endif

end module element_mod
