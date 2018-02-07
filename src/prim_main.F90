#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

program prim_main
#ifdef _PRIM
  use prim_driver_mod,  only: prim_init1, prim_init2, prim_finalize
#ifndef USE_KOKKOS_KERNELS
  use prim_driver_mod,  only: leapfrog_bootstrap, prim_run, prim_run_subcycle
#endif
  use hybvcoord_mod,    only: hvcoord_t, hvcoord_init
#endif

  use parallel_mod,     only: parallel_t, initmp, syncmp, haltmp, abortmp
  use hybrid_mod,       only: hybrid_t
  use thread_mod,       only: nthreads, vert_num_threads, omp_get_thread_num, &
                              omp_set_num_threads, omp_get_nested, &
                              omp_get_num_threads, omp_get_max_threads
  use time_mod,         only: tstep, nendstep, timelevel_t, TimeLevel_init
  use dimensions_mod,   only: nelemd, qsize, ntrac
  use control_mod,      only: restartfreq, vfile_mid, vfile_int, runtype, integration, statefreq, tstep_type
#ifdef USE_KOKKOS_KERNELS
  use control_mod,      only: qsplit, rsplit
#endif
  use domain_mod,       only: domain1d_t, decompose
  use element_mod,      only: element_t
  use fvm_mod,          only: fvm_init3
  use common_io_mod,    only: output_dir
  use common_movie_mod, only: nextoutputstep
  use perf_mod,         only: t_initf, t_prf, t_finalizef, t_startf, t_stopf ! _EXTERNAL
  use restart_io_mod ,  only: restartheader_t, writerestart
  use hybrid_mod,       only: hybrid_create
  use fvm_control_volume_mod, only: fvm_struct
  use fvm_control_volume_mod, only: n0_fvm

#ifdef USE_KOKKOS_KERNELS
  use prim_cxx_driver_mod, only: cleanup_cxx_structures
  use element_mod,         only: elem_state_v, elem_state_temp, elem_state_dp3d, &
                                 elem_state_Qdp, elem_state_ps_v
  use iso_c_binding,       only: c_ptr, c_loc, c_int
#endif

#ifdef _REFSOLN
  use prim_state_mod, only : prim_printstate_par
#endif

#ifdef PIO_INTERP
  use interp_movie_mod, only : interp_movie_output, interp_movie_finish, interp_movie_init
  use interpolate_driver_mod, only : interpolate_driver
#else
  use prim_movie_mod,   only : prim_movie_output, prim_movie_finish,prim_movie_init
#endif

  implicit none
#ifdef USE_KOKKOS_KERNELS
  interface
    subroutine initialize_hommexx_session() bind(c)
    end subroutine initialize_hommexx_session

    subroutine init_hvcoord_c (ps0,hybrid_am_ptr,hybrid_ai_ptr,hybrid_bm_ptr,hybrid_bi_ptr) bind(c)
      use iso_c_binding , only : c_ptr, c_double
      !
      ! Inputs
      !
      real (kind=c_double),  intent(in) :: ps0
      type (c_ptr),          intent(in) :: hybrid_am_ptr, hybrid_ai_ptr
      type (c_ptr),          intent(in) :: hybrid_bm_ptr, hybrid_bi_ptr
    end subroutine init_hvcoord_c

    subroutine init_time_level_c(nm1,n0,np1,nstep,nstep0) bind(c)
      use iso_c_binding, only: c_int
      !
      ! Inputs
      !
      integer(kind=c_int), intent(in) :: nm1, n0, np1, nstep, nstep0
    end subroutine init_time_level_c

    subroutine prim_run_subcycle_c(tstep,nstep,nm1,n0,np1) bind(c)
      use iso_c_binding, only: c_int, c_double
      !
      ! Inputs
      !
      integer(kind=c_int),  intent(in) :: nstep, nm1, n0, np1
      real (kind=c_double), intent(in) :: tstep
    end subroutine prim_run_subcycle_c


    subroutine cxx_push_results_to_f90(elem_state_v_ptr, elem_state_temp_ptr, elem_state_dp3d_ptr, &
                                       elem_state_Qdp_ptr, elem_state_ps_v_ptr) bind(c)
      use iso_c_binding , only : c_ptr
      !
      ! Inputs
      !
      type (c_ptr),          intent(in) :: elem_state_v_ptr, elem_state_temp_ptr, elem_state_dp3d_ptr
      type (c_ptr),          intent(in) :: elem_state_Qdp_ptr, elem_state_ps_v_ptr
    end subroutine cxx_push_results_to_f90

    subroutine finalize_hommexx_session() bind(c)
    end subroutine finalize_hommexx_session
  end interface
#endif

  type (element_t),  pointer  :: elem(:)
  type (fvm_struct), pointer  :: fvm(:)
  type (hybrid_t)             :: hybrid         ! parallel structure for shared memory/distributed memory
  type (parallel_t)           :: par            ! parallel structure for distributed memory programming
  type (domain1d_t), pointer  :: dom_mt(:)
  type (RestartHeader_t)      :: RestartHeader
  type (TimeLevel_t)          :: tl             ! Main time level struct
  type (hvcoord_t), target    :: hvcoord        ! hybrid vertical coordinate struct

#ifdef USE_KOKKOS_KERNELS
  type (c_ptr) :: hybrid_am_ptr, hybrid_ai_ptr, hybrid_bm_ptr, hybrid_bi_ptr
  type (c_ptr) :: elem_state_v_ptr, elem_state_temp_ptr, elem_state_dp3d_ptr
  type (c_ptr) :: elem_state_Qdp_ptr, elem_state_ps_v_ptr
  integer (kind=c_int) :: nstep_c, nm1_c, n0_c, np1_c
#endif

  real*8 timeit, et, st
  integer nets,nete
  integer ithr
  integer ierr
  integer nstep

  character (len=20) :: numproc_char
  character (len=20) :: numtrac_char

  logical :: dir_e ! boolean existence of directory where output netcdf goes

  ! =====================================================
  ! Begin executable code set distributed memory world...
  ! =====================================================
  par=initmp()

#ifdef USE_KOKKOS_KERNELS
  ! Do this right away, but AFTER MPI initialization
  call initialize_hommexx_session()
#endif

  ! =====================================
  ! Set number of threads...
  ! =====================================
  call t_initf('input.nl',LogPrint=par%masterproc, &
  Mpicom=par%comm, MasterTask=par%masterproc)
  call t_startf('Total')
  call t_startf('prim_init1')
  call prim_init1(elem,  fvm, par,dom_mt,tl)
  call t_stopf('prim_init1')

  ! =====================================
  ! Begin threaded region so each thread can print info
  ! =====================================
#if (defined HORIZ_OPENMP && defined COLUMN_OPENMP)
   call omp_set_nested(.true.)
#ifndef __bg__
   if (.not. omp_get_nested()) then
     call haltmp("Nested threading required but not available. Set OMP_NESTED=true")
   endif
#endif
#endif

  ! =====================================
  ! Begin threaded region so each thread can print info
  ! =====================================
#if (defined HORIZ_OPENMP)
  !$OMP PARALLEL NUM_THREADS(nthreads), DEFAULT(SHARED), PRIVATE(ithr,nets,nete,hybrid)
  call omp_set_num_threads(vert_num_threads)
#endif
  ithr=omp_get_thread_num()
  nets=dom_mt(ithr)%start
  nete=dom_mt(ithr)%end
  ! ================================================
  ! Initialize thread decomposition
  ! Note: The OMP Critical is required for threading since the Fortran
  !   standard prohibits multiple I/O operations on the same unit.
  ! ================================================
#if (defined HORIZ_OPENMP)
  !$OMP CRITICAL
#endif
  if (par%rank<100) then
     write(6,9) par%rank,ithr,omp_get_max_threads(),nets,nete
  endif
9 format("process: ",i2,1x,"horiz thread id: ",i2,1x,"# vert threads: ",i2,1x,&
       "element limits: ",i5,"-",i5)
#if (defined HORIZ_OPENMP)
  !$OMP END CRITICAL
  !$OMP END PARALLEL
#endif

  ! setup fake threading so we can call routines that require 'hybrid'
  ithr=omp_get_thread_num()
  hybrid = hybrid_create(par,ithr,1)
  nets=1
  nete=nelemd


  ! ==================================
  ! Initialize the vertical coordinate  (cam initializes hvcoord externally)
  ! ==================================
  hvcoord = hvcoord_init(vfile_mid, vfile_int, .true., hybrid%masterthread, ierr)
  if (ierr /= 0) then
     call haltmp("error in hvcoord_init")
  end if

#ifdef USE_KOKKOS_KERNELS
  hybrid_am_ptr = c_loc(hvcoord%hyam)
  hybrid_ai_ptr = c_loc(hvcoord%hyai)
  hybrid_bm_ptr = c_loc(hvcoord%hybm)
  hybrid_bi_ptr = c_loc(hvcoord%hybi)
  call init_hvcoord_c (hvcoord%ps0,hybrid_am_ptr,hybrid_ai_ptr,hybrid_bm_ptr,hybrid_bi_ptr)
#endif

#ifdef PIO_INTERP
  if(runtype<0) then
     ! Interpolate a netcdf file from one grid to another
     call interpolate_driver(elem, hybrid)
     call haltmp('interpolation complete')
  end if
#endif

  if(par%masterproc) print *,"Primitive Equation Initialization..."
#if (defined HORIZ_OPENMP)
  !$OMP PARALLEL NUM_THREADS(nthreads), DEFAULT(SHARED), PRIVATE(ithr,nets,nete,hybrid)
  call omp_set_num_threads(vert_num_threads)
#endif
  ithr=omp_get_thread_num()
  hybrid = hybrid_create(par,ithr,nthreads)
  nets=dom_mt(ithr)%start
  nete=dom_mt(ithr)%end

  call t_startf('prim_init2')
  call prim_init2(elem, fvm,  hybrid,nets,nete,tl, hvcoord)
  call t_stopf('prim_init2')
#if (defined HORIZ_OPENMP)
  !$OMP END PARALLEL
#endif
  ! setup fake threading so we can call routines that require 'hybrid'
  ithr=omp_get_thread_num()
  hybrid = hybrid_create(par,ithr,1)
  nets=1
  nete=nelemd




  ! Here we get sure the directory specified
  ! in the input namelist file in the
  ! variable 'output_dir' does exist.
  ! this avoids a abort deep within the PIO
  ! library (SIGABRT:signal 6) which in most
  ! architectures produces a core dump.
  if (par%masterproc) then
     open(unit=447,file=trim(output_dir) // "/output_dir_test",iostat=ierr)
     if ( ierr==0 ) then
        print *,'Directory ',trim(output_dir), ' does exist: initialing IO'
        close(447)
     else
        print *,'Error creating file in directory ',trim(output_dir)
        call haltmp("Please be sure the directory exist or specify 'output_dir' in the namelist.")
     end if
  endif
#if 0
  this ALWAYS fails on lustre filesystems.  replaced with the check above
  inquire( file=output_dir, exist=dir_e )
  if ( dir_e ) then
     if(par%masterproc) print *,'Directory ',output_dir, ' does exist: initialing IO'
  else
     if(par%masterproc) print *,'Directory ',output_dir, ' does not exist: stopping'
     call haltmp("Please get sure the directory exist or specify one via output_dir in the namelist file.")
  end if
#endif


#ifdef PIO_INTERP
  ! initialize history files.  filename constructed with restart time
  ! so we have to do this after ReadRestart in prim_init2 above
  call interp_movie_init( elem, par,  hvcoord, tl )
#else
  call prim_movie_init( elem, par, hvcoord, tl )
#endif


  ! output initial state for NEW runs (not restarts or branch runs)
  if (runtype == 0 ) then
#ifdef PIO_INTERP
     call interp_movie_output(elem, tl, par, 0d0, fvm=fvm, hvcoord=hvcoord)
#else
     call prim_movie_output(elem, tl, hvcoord, par, fvm)
#endif
  endif


  ! advance_si not yet upgraded to be self-starting.  use leapfrog bootstrap procedure:
  if(integration == 'semi_imp') then
     if (runtype /= 1 ) then
#ifdef USE_KOKKOS_KERNELS
        call abortmp("Error! Cannot use this option in Kokkos build")
#else
        if(par%masterproc) print *,"Leapfrog bootstrap initialization..."
        call leapfrog_bootstrap(elem, hybrid,1,nelemd,tstep,tl,hvcoord)
#endif
     endif
  endif

#ifdef USE_KOKKOS_KERNELS
  call init_time_level_c(tl%nm1,tl%n0,tl%np1, tl%nstep, tl%nstep0)
#endif

  if(par%masterproc) print *,"Entering main timestepping loop"
  call t_startf('prim_main_loop')
  do while(tl%nstep < nEndStep)
#if (defined HORIZ_OPENMP)
     !$OMP PARALLEL NUM_THREADS(nthreads), DEFAULT(SHARED), PRIVATE(ithr,nets,nete,hybrid)
     call omp_set_num_threads(vert_num_threads)
#endif
     ithr=omp_get_thread_num()
     hybrid = hybrid_create(par,ithr,nthreads)
     nets=dom_mt(ithr)%start
     nete=dom_mt(ithr)%end

     nstep = nextoutputstep(tl)
!JMD     call vprof_start()
     do while(tl%nstep<nstep)
        call t_startf('prim_run')
        if (tstep_type>0) then  ! forward in time subcycled methods
#ifdef USE_KOKKOS_KERNELS
          if (nets/=1 .or. nete/=nelemd) then
            call abortmp ("We don't allow to call C routines from a horizontally threaded region")
          endif
          call prim_run_subcycle_c(tstep,nstep_c,nm1_c,n0_c,np1_c)
          tl%nstep = nstep_c
          tl%nm1   = nm1_c + 1
          tl%n0    = n0_c  + 1
          tl%np1   = np1_c + 1
#else
          call prim_run_subcycle(elem, fvm, hybrid,nets,nete, tstep, tl, hvcoord,1)
#endif
        else  ! leapfrog
#ifdef USE_KOKKOS_KERNELS
           call abortmp ("Error! Functionality not available in Kokkos build")
#else
           call prim_run(elem, hybrid,nets,nete, tstep, tl, hvcoord, "leapfrog")
#endif
        endif
        call t_stopf('prim_run')
     end do
!JMD     call vprof_stop()
#if (defined HORIZ_OPENMP)
     !$OMP END PARALLEL
#endif
     ! setup fake threading so we can call routines that require 'hybrid'
     ithr=omp_get_thread_num()
     hybrid = hybrid_create(par,ithr,1)
     nets=1
     nete=nelemd

#ifdef USE_KOKKOS_KERNELS
     elem_state_v_ptr         = c_loc(elem_state_v)
     elem_state_temp_ptr      = c_loc(elem_state_temp)
     elem_state_dp3d_ptr      = c_loc(elem_state_dp3d)
     elem_state_Qdp_ptr       = c_loc(elem_state_Qdp)
     elem_state_ps_v_ptr      = c_loc(elem_state_ps_v)
     call cxx_push_results_to_f90(elem_state_v_ptr, elem_state_temp_ptr, elem_state_dp3d_ptr, &
                                  elem_state_Qdp_ptr, elem_state_ps_v_ptr)
#endif

#ifdef PIO_INTERP
     if (ntrac>0) call fvm_init3(elem,fvm,hybrid,nets,nete,n0_fvm)
     call interp_movie_output(elem, tl, par, 0d0,fvm=fvm, hvcoord=hvcoord)
#else
     call prim_movie_output(elem, tl, hvcoord, par, fvm)
#endif

#ifdef _REFSOLN
     call prim_printstate_par(elem, tl,hybrid,hvcoord,nets,nete, par)
#endif

     ! ============================================================
     ! Write restart files if required
     ! ============================================================
     if((restartfreq > 0) .and. (MODULO(tl%nstep,restartfreq) ==0)) then
        call WriteRestart(elem, ithr,1,nelemd,tl)
     endif
  end do
  call t_stopf('prim_main_loop')

  if(par%masterproc) print *,"Finished main timestepping loop",tl%nstep
  call prim_finalize()
  if(par%masterproc) print *,"closing history files"
#ifdef PIO_INTERP
  call interp_movie_finish
#else
  call prim_movie_finish
#endif

  call t_stopf('Total')
  if(par%masterproc) print *,"writing timing data"
!   write(numproc_char,*) par%nprocs
!   write(numtrac_char,*) ntrac
!   call system('mkdir -p '//'time/'//trim(adjustl(numproc_char))//'-'//trim(adjustl(numtrac_char)))
!   call t_prf('time/HommeFVMTime-'//trim(adjustl(numproc_char))//'-'//trim(adjustl(numtrac_char)),par%comm)
  call t_prf('HommeTime', par%comm)
  if(par%masterproc) print *,"calling t_finalizef"
  call t_finalizef()
#ifdef USE_KOKKOS_KERNELS
  call cleanup_cxx_structures ()
  call finalize_hommexx_session()
#endif
  call haltmp("exiting program...")
end program prim_main
