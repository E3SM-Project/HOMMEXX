#ifndef HOMME_REGION_HPP
#define HOMME_REGION_HPP

#include "Dimensions.hpp"

#include "Types.hpp"

#include "CaarControl.hpp"
#include <Kokkos_Core.hpp>

#include "vector/KokkosKernels_Vector.hpp"

#include <random>

namespace Homme {

/* Per element data - specific velocity, temperature, pressure, etc. */
class CaarRegion {
public:
  // Coriolis term
  ExecViewManaged<Real * [NP][NP]> m_fcor;
  // Differential geometry things
  ExecViewManaged<Real * [NP][NP]> m_spheremp;
  ExecViewManaged<Real * [NP][NP]> m_metdet;
  // Prescrived surface geopotential height at eta = 1
  ExecViewManaged<Real * [NP][NP]> m_phis;

  // Differential geometry tensors
  ExecViewManaged<Real * [2][2][NP][NP]> m_d;
  ExecViewManaged<Real * [2][2][NP][NP]> m_dinv;

  // Omega is the pressure vertical velocity
  ExecViewManaged<Real * [NP][NP][NUM_LEV]> m_omega_p;
  // ???
  ExecViewManaged<Real * [NP][NP][NUM_LEV]> m_pecnd;
  // Geopotential height field
  ExecViewManaged<Real * [NP][NP][NUM_LEV]> m_phi;
  // ???
  ExecViewManaged<Real * [NP][NP][NUM_LEV]> m_derived_un0;
  // ???
  ExecViewManaged<Real * [NP][NP][NUM_LEV]> m_derived_vn0;

  // Lateral Velocity
  ExecViewManaged<Real * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> m_u;
  ExecViewManaged<Real * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> m_v;
  // Temperature
  ExecViewManaged<Real * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> m_t;
  // ???
  ExecViewManaged<Real * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> m_dp3d;

  // q is the specific humidity
  ExecViewManaged<Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]> m_qdp;
  // eta is the vertical coordinate
  // eta dot is the flux through the vertical level interface
  //    (note there are NUM_LEV_P of them)
  // dpdn is the derivative of pressure with respect to eta
  ExecViewManaged<Real * [NP][NP][NUM_LEV_P]> m_eta_dot_dpdn;

  struct BufferViews {
    BufferViews() = default;
    void init(const int num_elems);
    ExecViewManaged<Real * [NP][NP][NUM_LEV]> pressure;
    ExecViewManaged<Real * [2][NP][NP][NUM_LEV]> pressure_grad;
    ExecViewManaged<Real * [NP][NP][NUM_LEV]> temperature_virt;
    ExecViewManaged<Real * [2][NP][NP][NUM_LEV]> temperature_grad;
    ExecViewManaged<Real * [NP][NP][NUM_LEV]> omega_p;
    ExecViewManaged<Real * [2][NP][NP][NUM_LEV]> vdp;
    ExecViewManaged<Real * [NP][NP][NUM_LEV]> div_vdp;
    ExecViewManaged<Real * [NP][NP][NUM_LEV]> ephi;
    ExecViewManaged<Real * [2][NP][NP][NUM_LEV]> energy_grad;
    ExecViewManaged<Real * [NP][NP][NUM_LEV]> vorticity;
  } buffers;

  CaarRegion() = default;

  void init(const int num_elems);

  void random_init(int num_elems, std::mt19937_64 &engine);

  int num_elems() const { return m_num_elems; }

  // Fill the exec space views with data coming from F90 pointers
  void init_2d(CF90Ptr &D, CF90Ptr &Dinv, CF90Ptr &fcor, CF90Ptr &spheremp,
               CF90Ptr &metdet, CF90Ptr &phis);

  // Fill the exec space views with data coming from F90 pointers
  void pull_from_f90_pointers(CF90Ptr &state_v, CF90Ptr &state_t,
                              CF90Ptr &state_dp3d, CF90Ptr &derived_phi,
                              CF90Ptr &derived_pecnd, CF90Ptr &derived_omega_p,
                              CF90Ptr &derived_v, CF90Ptr &derived_eta_dot_dpdn,
                              CF90Ptr &state_qdp);
  void pull_3d(CF90Ptr &derived_phi, CF90Ptr &derived_pecnd,
               CF90Ptr &derived_omega_p, CF90Ptr &derived_v,
               CF90Ptr &derived_eta_dot_dpdn);
  void pull_4d(CF90Ptr &state_v, CF90Ptr &state_t, CF90Ptr &state_dp3d);
  void pull_extra(CF90Ptr &derived_eta_dot_dpdn, CF90Ptr &state_qdp);

  // Push the results from the exec space views to the F90 pointers
  void push_to_f90_pointers(F90Ptr &state_v, F90Ptr &state_t, F90Ptr &state_dp,
                            F90Ptr &derived_phi, F90Ptr &derived_pecnd,
                            F90Ptr &derived_omega_p, F90Ptr &derived_v,
                            F90Ptr &derived_eta_dot_dpdn,
                            F90Ptr &state_qdp) const;
  void push_3d(F90Ptr &derived_phi, F90Ptr &derived_pecnd,
               F90Ptr &derived_omega_p, F90Ptr &derived_v,
               F90Ptr &derived_eta_dot_dpdn) const;
  void push_4d(F90Ptr &state_v, F90Ptr &state_t, F90Ptr &state_dp3d) const;
  void push_extra(F90Ptr &derived_eta_dot_dpdn, F90Ptr &state_qdp) const;

  void d(Real *d_ptr, int ie) const;
  void dinv(Real *dinv_ptr, int ie) const;

private:
  int m_num_elems;
};

// TODO: DON'T USE SINGLETONS
CaarRegion &get_region();

} // Homme

#endif // HOMME_REGION_HPP
