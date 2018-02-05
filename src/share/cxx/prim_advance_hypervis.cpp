#include "BoundaryExchange.hpp"
#include "BuffersManager.hpp"
#include "Context.hpp"
#include "Control.hpp"
#include "Elements.hpp"
#include "SimulationParams.hpp"
#include "HyperviscosityFunctor.hpp"

#include "Types.hpp"

namespace Homme
{

//void pull_hypervis_data_c (CF90Ptr& elem_state_v_ptr, CF90Ptr& elem_state_t_ptr, CF90Ptr& elem_state_dp3d_ptr,
//                           CF90Ptr& elem_derived_dpdiss_ave_ptr, CF90Ptr& elem_derived_dpdiss_biharmonic_ptr)
//{
//  // Get control and elements
//  Control&  data     = Context::singleton().get_control();
//  Elements& elements = Context::singleton().get_elements();

//  // Forward states from pointers to Elements
//  elements.pull_4d(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr);

//  // Forward derived quantities from pointers to Elements
//  HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]> dpdiss_ave_f90        (elem_derived_dpdiss_ave_ptr,        data.num_elems);
//  HostViewUnmanaged<const Real*[NUM_PHYSICAL_LEV][NP][NP]> dpdiss_biharmonic_f90 (elem_derived_dpdiss_biharmonic_ptr, data.num_elems);

//  sync_to_device(dpdiss_ave_f90,        elements.m_derived_dpdiss_ave       );
//  sync_to_device(dpdiss_biharmonic_f90, elements.m_derived_dpdiss_biharmonic);
//}

void advance_hypervis_dp (const int np1, const Real dt, const Real eta_ave_w)
{
  // Get simulation parameters, control, elements and derivative
  SimulationParams& params   = Context::singleton().get_simulation_params();
  Control&          data     = Context::singleton().get_control();
  Elements&         elements = Context::singleton().get_elements();
  Derivative&       deriv    = Context::singleton().get_derivative();

  // Set elements some inputs into the control
  data.np1       = np1;
  data.dt        = dt/params.hypervis_subcycle;
  data.eta_ave_w = eta_ave_w;
  data.nu_ratio  = 1.0;

  // Make sure all the BE needed by Hyperviscosity
  std::string be_name = "HyperviscosityFunctor";
  BoundaryExchange& be = *Context::singleton().get_boundary_exchange(be_name);
  if (!be.is_registration_completed())
  {
    std::shared_ptr<BuffersManager> buffers_manager = Context::singleton().get_buffers_manager(MPI_EXCHANGE);
    be.set_buffers_manager(buffers_manager);

    // Set the views into the BE: dptens (one 3d field), ttens (one 3d field), vtens (two 3d fields)
    be.set_num_fields(0,0,4);
    be.register_field(elements.buffers.vtens,2,0);
    be.register_field(elements.buffers.ttens);
    be.register_field(elements.buffers.dptens);
    be.registration_completed();
  }

  HyperviscosityFunctor functor(data,elements,deriv);
  functor.run(params.hypervis_subcycle);
}

//void push_hypervis_results_c (F90Ptr& elem_state_v_ptr, F90Ptr& elem_state_t_ptr, F90Ptr& elem_state_dp3d_ptr,
//                              F90Ptr& elem_derived_dpdiss_ave_ptr, F90Ptr& elem_derived_dpdiss_biharmonic_ptr)
//{
//  // Get control and elements
//  Control&  data     = Context::singleton().get_control();
//  Elements& elements = Context::singleton().get_elements();

//  // Forward states from Elements to pointers
//  elements.push_4d(elem_state_v_ptr,elem_state_t_ptr,elem_state_dp3d_ptr);

//  // Forward derived quantities from Elements to pointers
//  HostViewUnmanaged<Real*[NUM_PHYSICAL_LEV][NP][NP]> dpdiss_ave_f90        (elem_derived_dpdiss_ave_ptr,        data.num_elems);
//  HostViewUnmanaged<Real*[NUM_PHYSICAL_LEV][NP][NP]> dpdiss_biharmonic_f90 (elem_derived_dpdiss_biharmonic_ptr, data.num_elems);

//  sync_to_host(elements.m_derived_dpdiss_ave,        dpdiss_ave_f90       );
//  sync_to_host(elements.m_derived_dpdiss_biharmonic, dpdiss_biharmonic_f90);
//}

} // namespace Homme
