#include "BoundaryExchange.hpp"
#include "BuffersManager.hpp"
#include "Context.hpp"
#include "Control.hpp"
#include "Derivative.hpp"
#include "Elements.hpp"
#include "TimeLevel.hpp"
#include "SimulationParams.hpp"

namespace Homme
{

extern "C"
{

void init_simulation_params_c (const int& remap_alg, const int& limiter_option, const int& rsplit, const int& qsplit,
                               const int& time_step_type, const int& prescribed_wind, const int& energy_fixer,
                               const int& qsize, const int& state_frequency,
                               const Real& nu, const Real& nu_p, const Real& nu_s,
                               const int& hypervis_order, const int& hypervis_subcycle,
                               const bool& disable_diagnostics, const bool& moisture, const bool& use_semi_lagrangian_transport)
{
  // Get the simulation params struct
  SimulationParams& params = Context::singleton().get_simulation_params();

  if (remap_alg==1) {
    params.remap_alg = RemapAlg::PPM_MIRRORED;
  } else if (remap_alg==2) {
    params.remap_alg = RemapAlg::PPM_FIXED;
  } else {
    Errors::runtime_abort("Error in init_simulation_params_c: unknown remap algorithm.\n",
                           Errors::unknown_option);
  }

  params.limiter_option                = limiter_option;
  params.rsplit                        = rsplit;
  params.qsplit                        = qsplit;
  params.time_step_type                = time_step_type;
  params.prescribed_wind               = (prescribed_wind>0);
  params.energy_fixer                  = (energy_fixer>0);
  params.state_frequency               = state_frequency;
  params.qsize                         = qsize;
  params.nu                            = nu;
  params.nu_p                          = nu_p;
  params.nu_s                          = nu_s;
  params.hypervis_order                = hypervis_order;
  params.hypervis_subcycle             = hypervis_subcycle;
  params.disable_diagnostics           = disable_diagnostics;
  params.moisture                      = (moisture ? MoistDry::MOIST : MoistDry::DRY);
  params.use_semi_lagrangian_transport = use_semi_lagrangian_transport;

  // Now this structure can be used safely
  params.params_set = true;
}

void init_hvcoord_c (const Real& ps0, CF90Ptr& hyai_ptr, CF90Ptr& hybi_ptr)
{
  Control& data = Context::singleton().get_control();
  data.init_hvcoord(ps0,hyai_ptr,hybi_ptr);
}

void init_time_levels_c (const int& nm1, const int& n0, const int& np1)
{
  TimeLevel& tl = Context::singleton().get_time_level();
  tl.nm1 = nm1-1;
  tl.n0  = n0-1;
  tl.np1 = np1-1;
}

void init_derivative_c (CF90Ptr& dvv)
{
  Derivative& deriv = Context::singleton().get_derivative ();
  deriv.init(dvv);
}

void init_elements_2d_c (const int& num_elems, CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor,
                         CF90Ptr& spheremp, CF90Ptr& rspheremp, CF90Ptr& metdet, CF90Ptr& phis)
{
  Elements& r = Context::singleton().get_elements ();
  r.init (num_elems);
  r.init_2d(D,Dinv,fcor,spheremp,rspheremp,metdet,phis);

  Control& data = Context::singleton().get_control();
  data.num_elems = num_elems;
}

void init_elements_states_c (CF90Ptr& elem_state_v_ptr,   CF90Ptr& elem_state_temp_ptr, CF90Ptr& elem_state_dp3d_ptr,
                             CF90Ptr& elem_state_Qdp_ptr, CF90Ptr& elem_state_ps_v_ptr, CF90Ptr& elem_state_lnps_ptr)
{
  Elements& elements = Context::singleton().get_elements ();
  elements.pull_4d(elem_state_v_ptr,elem_state_temp_ptr,elem_state_dp3d_ptr);
  elements.pull_qdp(elem_state_Qdp_ptr);

  // F90 ptrs to arrays (np,np,num_time_levels,nelemd) can be stuffed directly in an unmanaged view
  // with scalar Real*[NUM_TIME_LEVELS][NP][NP] (with runtime dimension nelemd)
  Control& data = Context::singleton().get_control();
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NP][NP]> ps_v_f90(elem_state_ps_v_ptr,data.num_elems);
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NP][NP]> lnps_f90(elem_state_lnps_ptr,data.num_elems);

  decltype(elements.m_ps_v)::HostMirror ps_v_host = Kokkos::create_mirror_view(elements.m_ps_v);
  decltype(elements.m_ps_v)::HostMirror lnps_host = Kokkos::create_mirror_view(elements.m_lnps);

  Kokkos::deep_copy(ps_v_host,ps_v_f90);
  Kokkos::deep_copy(elements.m_ps_v,ps_v_host);
  Kokkos::deep_copy(lnps_host,lnps_f90);
  Kokkos::deep_copy(elements.m_lnps,lnps_host);
}

void init_boundary_exchanges_c ()
{
  SimulationParams& params = Context::singleton().get_simulation_params();
  Elements& elements = Context::singleton().get_elements();
  std::shared_ptr<BuffersManager> bm_exchange        = Context::singleton().get_buffers_manager(MPI_EXCHANGE);
  std::shared_ptr<BuffersManager> bm_exchange_minmax = Context::singleton().get_buffers_manager(MPI_EXCHANGE_MIN_MAX);

  // Euler minmax BE's
  {
    // Get the BE
    BoundaryExchange& be = *Context::singleton().get_boundary_exchange("min max Euler");

    // Safety check (do not call this routine twice!)
    assert (!be.is_registration_completed());

    // Setup the BE
    be.set_buffers_manager(bm_exchange_minmax);
    be.set_num_fields(params.qsize,0,0);
    be.register_min_max_fields(elements.buffers.qlim,params.qsize,0);
    be.registration_completed();

  }

  // Euler qdp (and dss var) BE
  {
    std::map<std::string,decltype(elements.m_omega_p)> dss_var_map;
    dss_var_map["eta"]         = elements.m_eta_dot_dpdn;
    dss_var_map["omega"]       = elements.m_eta_dot_dpdn;
    dss_var_map["div_vdp_ave"] = elements.m_eta_dot_dpdn;
    for (auto it : dss_var_map) {
      for (int np1_qdp=0; np1_qdp<Q_NUM_TIME_LEVELS; ++np1_qdp) {
        std::stringstream ss;
        ss << "exchange qdp " << it.first << " " << np1_qdp;

        // Get the BE
        BoundaryExchange& be = *Context::singleton().get_boundary_exchange(ss.str());

        // Safety check (do not call this routine twice!)
        assert (!be.is_registration_completed());

        // Setup the BE
        be.set_buffers_manager(bm_exchange);
        be.set_num_fields(0,0,params.qsize+1);
        be.register_field(elements.m_qdp,np1_qdp,params.qsize,0);
        be.register_field(it.second);
        be.registration_completed();
      }
    }
  }

  // RK stages BE's
  {
    std::shared_ptr<BuffersManager> buffers_manager = Context::singleton().get_buffers_manager(MPI_EXCHANGE);
    for (int tl=0; tl<NUM_TIME_LEVELS; ++tl) {
      std::stringstream ss;
      ss << "caar tl " << tl;

      // Get the BE
      BoundaryExchange& be = *Context::singleton().get_boundary_exchange(ss.str());

      // Safety check (do not call this routine twice!)
      assert (!be.is_registration_completed());

      // If it was not yet created, create it and set it up
      be.set_buffers_manager(bm_exchange);

      // Set the views of this time level into this time level's boundary exchange
      be.set_num_fields(0,0,4);
      be.register_field(elements.m_u,1,tl);
      be.register_field(elements.m_v,1,tl);
      be.register_field(elements.m_t,1,tl);
      be.register_field(elements.m_dp3d,1,tl);
      be.registration_completed();
    }
  }
}

void cxx_push_results_to_f90(F90Ptr& elem_state_v_ptr,   F90Ptr& elem_state_temp_ptr, F90Ptr& elem_state_dp3d_ptr,
                             F90Ptr& elem_state_Qdp_ptr, F90Ptr& elem_state_ps_v_ptr)
{
  Elements& elements = Context::singleton().get_elements ();
  elements.push_4d(elem_state_v_ptr,elem_state_temp_ptr,elem_state_dp3d_ptr);
  elements.push_qdp(elem_state_Qdp_ptr);

  // F90 ptrs to arrays (np,np,num_time_levels,nelemd) can be stuffed directly in an unmanaged view
  // with scalar Real*[NUM_TIME_LEVELS][NP][NP] (with runtime dimension nelemd)
  Control& data = Context::singleton().get_control();
  HostViewUnmanaged<Real*[NUM_TIME_LEVELS][NP][NP]> ps_v_f90(elem_state_ps_v_ptr,data.num_elems);

  decltype(elements.m_ps_v)::HostMirror ps_v_host = Kokkos::create_mirror_view(elements.m_ps_v);

  Kokkos::deep_copy(ps_v_host,elements.m_ps_v);
  Kokkos::deep_copy(ps_v_f90,ps_v_host);
}

} // extern "C"

} // namespace Homme
