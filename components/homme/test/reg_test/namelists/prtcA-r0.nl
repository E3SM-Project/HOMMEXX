&ctl_nl
vert_num_threads  = 1
NThreads          = 1
partmethod        = 4
topology          = "cube"
test_case         = "jw_baroclinic"
u_perturb         = 1
rotate_grid       = 0
ne                = 4
qsize             = 4
ndays             = 1
statefreq         = 72
restartfreq       = 43200
restartfile       = "./R0001"
runtype           = 0
mesh_file         = '/dev/null'
tstep             = 600
rsplit            = 0
qsplit            = 1
tstep_type        = 5
energy_fixer      = -1
integration       = "explicit"
smooth            = 0
nu                = 7e15
nu_div            = 7e15
nu_p              = 7e15
nu_q              = 7e15
nu_s              =-1
nu_top            = 2.5e5
se_ftype          = 0
limiter_option    = 8
vert_remap_q_alg  = 1
hypervis_scaling  =0
hypervis_order    = 2
hypervis_subcycle =3
hypervis_subcycle =3
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
vform     = "ccm"
vfile_mid = './vcoord/camm-26.ascii'
vfile_int = './vcoord/cami-26.ascii'
/

&prof_inparm
profile_outpe_num   = 100
profile_single_file = .true.
/

!  timunits: 0= steps, 1=days, 2=hours			
&analysis_nl
 interp_gridtype   = 2
 output_timeunits  = 1,1
 output_frequency  = 1,1
 output_start_time = 1,1
 output_end_time   = 30000,30000
 output_varnames1  = 'ps','zeta','u','v','T'
 output_varnames2  = 'Q','Q2','Q3','Q4'
 io_stride         = 8
 output_type       = 'netcdf'
/
