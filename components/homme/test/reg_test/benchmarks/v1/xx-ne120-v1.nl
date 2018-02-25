&ctl_nl
NThreads=1
vert_num_threads = 1
partmethod    = 4
topology      = "cube"
test_case     = "jw_baroclinic"
u_perturb = 1
rotate_grid = 0
ne=120
qsize = 40
ndays=1
!nmax = 1152 ! one day
statefreq=9999
disable_diagnostics = .true.
restartfreq   = 43200
restartfile   = "./R0001"
runtype       = 0
mesh_file='/dev/null'
tstep=75      ! ne30: 300  ne120: 75
rsplit=2       ! ne30: 3   ne120:  2
qsplit = 1
tstep_type = 5
integration   = "explicit"
nu=1e13
nu_div = -1 ! nu_div=2.5e13
nu_p=1e13
nu_q=1e13
nu_s=1e13
nu_top = 2.5e5
se_ftype     = 0
limiter_option = 8
vert_remap_q_alg = 1
hypervis_scaling=0
hypervis_order = 2
hypervis_subcycle=4    ! ne30: 3  ne120: 4
/
&solver_nl
precon_method = "identity"
maxits        = 500
tol           = 1.e-9
/
&filter_nl
filter_type   = "taylor"
transfer_type = "bv"
filter_freq   = 0
filter_mu     = 0.04D0
p_bv          = 12.0D0
s_bv          = .666666666666666666D0
wght_fm       = 0.10D0
kcut_fm       = 2
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
 output_frequency=-1,-1
 output_start_time=0,0
 output_end_time=30000,30000
 output_varnames1='ps','zeta','T','geo'
 output_varnames2='Q','Q2','Q3','Q4','Q5'
 output_prefix='xx-ne120-'
 io_stride=8
 output_type = 'netcdf' 
/

