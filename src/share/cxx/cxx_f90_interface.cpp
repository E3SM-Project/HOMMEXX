#include "Derivative.hpp"
#include "Elements.hpp"
#include "Context.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "HommexxEnums.hpp"
#include "EulerStepFunctor.hpp"
#include "HyperviscosityFunctor.hpp"
#include "CaarFunctor.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/BuffersManager.hpp"
#include "ErrorDefs.hpp"

#include "utilities/SyncUtils.hpp"

#include "profiling.hpp"

namespace Homme
{

extern "C"
{

void init_simulation_params_c (const int& remap_alg, const int& limiter_option, const int& rsplit, const int& qsplit,
                               const int& time_step_type, const int& energy_fixer, const int& qsize, const int& state_frequency,
                               const Real& nu, const Real& nu_p, const Real& nu_q, const Real& nu_s, const Real& nu_div, const Real& nu_top,
                               const int& hypervis_order, const int& hypervis_subcycle, const double& hypervis_scaling,
                               const bool& prescribed_wind, const bool& moisture, const bool& disable_diagnostics, const bool& use_semi_lagrangian_transport)
{
  // Check that the simulation options are supported. This helps us in the future, since we
  // are currently 'assuming' some option have/not have certain values. As we support for more
  // options in the C++ build, we will remove some checks
  Errors::check_option("init_simulation_params_c","vert_remap_q_alg",remap_alg,{1,2});
  Errors::check_option("init_simulation_params_c","prescribed_wind",prescribed_wind,{false});
  Errors::check_option("init_simulation_params_c","hypervis_order",hypervis_order,{2});
  Errors::check_option("init_simulation_params_c","hypervis_scaling",hypervis_scaling,{0.0});
  Errors::check_option("init_simulation_params_c","use_semi_lagrangian_transport",use_semi_lagrangian_transport,{false});
  Errors::check_option("init_simulation_params_c","time_step_type",time_step_type,{5});
  Errors::check_option("init_simulation_params_c","limiter_option",limiter_option,{8});
  Errors::check_option("init_simulation_params_c","nu_p",nu_p,0.0,Errors::ComparisonOp::GT);
  Errors::check_option("init_simulation_params_c","nu",nu,0.0,Errors::ComparisonOp::GT);
  Errors::check_option("init_simulation_params_c","nu_div",nu_div,0.0,Errors::ComparisonOp::GT);
  Errors::check_options_relation("init_simulation_params_c","nu_div","nu",nu_div,nu,Errors::ComparisonOp::EQ);

  // Get the simulation params struct
  SimulationParams& params = Context::singleton().get_simulation_params();

  if (remap_alg==1) {
    params.remap_alg = RemapAlg::PPM_MIRRORED;
  } else if (remap_alg==2) {
    params.remap_alg = RemapAlg::PPM_FIXED;
  }

  params.limiter_option                = limiter_option;
  params.rsplit                        = rsplit;
  params.qsplit                        = qsplit;
  params.time_step_type                = time_step_type;
  params.prescribed_wind               = prescribed_wind;
  params.energy_fixer                  = (energy_fixer>0);
  params.state_frequency               = state_frequency;
  params.qsize                         = qsize;
  params.nu                            = nu;
  params.nu_p                          = nu_p;
  params.nu_q                          = nu_q;
  params.nu_s                          = nu_s;
  params.nu_div                        = nu_div;
  params.nu_top                        = nu_top;
  params.hypervis_order                = hypervis_order;
  params.hypervis_subcycle             = hypervis_subcycle;
  params.disable_diagnostics           = disable_diagnostics;
  params.moisture                      = (moisture ? MoistDry::MOIST : MoistDry::DRY);
  params.use_semi_lagrangian_transport = use_semi_lagrangian_transport;

  // TODO Parse a fortran string and set this properly. For now, our code does
  // not depend on this except to throw an error in apply_test_forcing.
  params.test_case = TestCase::JW_BAROCLINIC;

  // Now this structure can be used safely
  params.params_set = true;
}

void init_hvcoord_c (const Real& ps0, CRCPtr& hybrid_am_ptr, CRCPtr& hybrid_ai_ptr,
                                      CRCPtr& hybrid_bm_ptr, CRCPtr& hybrid_bi_ptr)
{
  HybridVCoord& hvcoord = Context::singleton().get_hvcoord();
  hvcoord.init(ps0,hybrid_am_ptr,hybrid_ai_ptr,hybrid_bm_ptr,hybrid_bi_ptr);
}

void cxx_push_results_to_f90(F90Ptr& elem_state_v_ptr,   F90Ptr& elem_state_temp_ptr, F90Ptr& elem_state_dp3d_ptr,
                             F90Ptr& elem_state_Qdp_ptr, F90Ptr& elem_Q_ptr, F90Ptr& elem_state_ps_v_ptr,
                             F90Ptr& elem_derived_omega_p_ptr)
{
  Elements& elements = Context::singleton().get_elements ();
  elements.push_4d(elem_state_v_ptr,elem_state_temp_ptr,elem_state_dp3d_ptr);
  elements.push_qdp(elem_state_Qdp_ptr);

  // F90 ptrs to arrays (np,np,num_time_levels,nelemd) can be stuffed directly in an unmanaged view
  // with scalar Real*[NUM_TIME_LEVELS][NP][NP] (with runtime dimension nelemd)
  HostViewUnmanaged<Real*[NUM_TIME_LEVELS][NP][NP]> ps_v_f90(elem_state_ps_v_ptr,elements.num_elems());

  decltype(elements.m_ps_v)::HostMirror ps_v_host = Kokkos::create_mirror_view(elements.m_ps_v);

  Kokkos::deep_copy(ps_v_host,elements.m_ps_v);
  Kokkos::deep_copy(ps_v_f90,ps_v_host);

  sync_to_host(elements.m_omega_p,HostViewUnmanaged<Real *[NUM_PHYSICAL_LEV][NP][NP]>(elem_derived_omega_p_ptr,elements.num_elems()));
  sync_to_host(elements.m_Q,HostViewUnmanaged<Real*[QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>(elem_Q_ptr,elements.num_elems()));
}

void init_derivative_c (CF90Ptr& dvv)
{
  Derivative& deriv = Context::singleton().get_derivative ();
  deriv.init(dvv);
}

void init_time_level_c (const int& nm1, const int& n0, const int& np1,
                        const int& nstep, const int& nstep0)
{
  TimeLevel& tl = Context::singleton().get_time_level ();
  tl.nm1    = nm1-1;
  tl.n0     = n0-1;
  tl.np1    = np1-1;
  tl.nstep  = nstep;
  tl.nstep0 = nstep0;
}

void init_elements_2d_c (const int& num_elems, CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor,
                         CF90Ptr& mp, CF90Ptr& spheremp, CF90Ptr& rspheremp,
                         CF90Ptr& metdet, CF90Ptr& metinv, CF90Ptr& phis)
{
  Elements& r = Context::singleton().get_elements ();
  r.init (num_elems);
  r.init_2d(D,Dinv,fcor,mp,spheremp,rspheremp,metdet,metinv,phis);
}

void init_elements_states_c (CF90Ptr& elem_state_v_ptr,   CF90Ptr& elem_state_temp_ptr, CF90Ptr& elem_state_dp3d_ptr,
                             CF90Ptr& elem_state_Qdp_ptr, CF90Ptr& elem_state_ps_v_ptr)
{
  Elements& elements = Context::singleton().get_elements ();
  elements.pull_4d(elem_state_v_ptr,elem_state_temp_ptr,elem_state_dp3d_ptr);
  elements.pull_qdp(elem_state_Qdp_ptr);

  // F90 ptrs to arrays (np,np,num_time_levels,nelemd) can be stuffed directly in an unmanaged view
  // with scalar Real*[NUM_TIME_LEVELS][NP][NP] (with runtime dimension nelemd)
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NP][NP]> ps_v_f90(elem_state_ps_v_ptr,elements.num_elems());

  decltype(elements.m_ps_v)::HostMirror ps_v_host = Kokkos::create_mirror_view(elements.m_ps_v);

  Kokkos::deep_copy(ps_v_host,ps_v_f90);
  Kokkos::deep_copy(elements.m_ps_v,ps_v_host);
}

void init_boundary_exchanges_c ()
{
  SimulationParams& params = Context::singleton().get_simulation_params();

  // Euler BEs
  auto& esf = Context::singleton().get_euler_step_functor();
  esf.reset(params);
  esf.init_boundary_exchanges();

  // RK stages BE's
  auto& cf = Context::singleton().get_caar_functor();
  cf.init_boundary_exchanges(Context::singleton().get_buffers_manager(MPI_EXCHANGE));

  // HyperviscosityFunctor's BE's
  auto& hvf = Context::singleton().get_hyperviscosity_functor();
  hvf.init_boundary_exchanges();
}

} // extern "C"

} // namespace Homme
