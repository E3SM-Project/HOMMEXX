&ctl_nl
NThreads=-1
partmethod    = 4
topology      = "cube"
test_case     = "jw_baroclinic"
u_perturb = 1
rotate_grid = 0
ne=NE
qsize = 40
nmax=NMAX
statefreq=360
disable_diagnostics = .true.
restartfreq   = 43200
restartfile   = "./R0001"
runtype       = 0
mesh_file='/dev/null'
tstep=TSTEP      ! ne30: 300  ne120: 75
rsplit=6       ! ne30: 3   ne120:  2
qsplit = 6
transport_alg=12
semi_lagrange_cdr_alg=20
tstep_type = 5
integration   = "explicit"
nu=1e13
nu_div=1e13
nu_p=1e13
nu_q=1e13
nu_s=1e13
nu_top = 2.5e5
se_ftype     = 0
limiter_option = 9
vert_remap_q_alg = 1
hypervis_scaling=0
hypervis_order = 2
hypervis_subcycle=4    ! ne30: 3  ne120: 4
/
&vert_nl
vform         = "ccm"
vfile_mid = './vcoord/acme-72m.ascii'
vfile_int = './vcoord/acme-72i.ascii'
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
 io_stride=8
 output_type = 'netcdf' 
/

