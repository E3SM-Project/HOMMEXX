&ctl_nl
NThreads=-1
vthreads = 1
partmethod    = 4
topology      = "cube"
test_case     = "jw_baroclinic"
u_perturb = 1
rotate_grid = 0
ne=2
!ndays=1
nmax       = 5000
qsize = 40
statefreq=9999
disable_diagnostics = .true.
restartfreq   = 43200
restartfile   = "./R0001"
runtype       = 0
mesh_file='/dev/null'
tstep=40      ! ne30: 300  ne120: 75
rsplit=3       ! ne30: 3   ne120:  2
qsplit = 1
tstep_type = 5
integration   = "explicit"
theta_hydrostatic_mode=.false.
nu=4e15
nu_div = -1 ! nu_div=1e16 
!in e3sm there are no ne20 settings 
!but the rule for nu_div is the same as for nu,
!C(dx)^3.2 with nu_div=2.5e15 for ne30
!so, approx (1.5)^3.2\simeq 4, 2.5e15*4=1e16
nu_p=4e15
nu_q=4e15
nu_s=4e15
nu_top = 2.5e5
se_ftype     = 0
limiter_option = 8
vert_remap_q_alg = 1
hypervis_scaling=0
hypervis_order = 2
hypervis_subcycle=3    ! ne30: 3  ne120: 4
/
&vert_nl
vform         = "ccm"
vfile_mid = './acme-72m.ascii'
vfile_int = './acme-72i.ascii'
/

&prof_inparm
profile_outpe_num = 100
profile_single_file		= .true.
/

&analysis_nl
! disabled
 output_timeunits=1,1
 output_frequency=0,0
 output_start_time=0,0
 output_end_time=30000,30000
 output_varnames1='ps','zeta','T','geo'
 output_varnames2='Q','Q2','Q3','Q4','Q5'
 output_prefix='homme-ne20-'
 io_stride=8
 output_type = 'netcdf' 
/

