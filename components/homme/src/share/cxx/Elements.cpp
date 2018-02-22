#include "Elements.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/TestUtils.hpp"
#include "HybridVCoord.hpp"
#include "Context.hpp"

#include <limits>
#include <random>
#include <assert.h>

namespace Homme {

void Element::init(Real* buffer) {

  m_phis = ExecViewUnmanaged<Real [NP][NP]>(buffer);
  buffer += NP*NP;

  m_fcor = ExecViewUnmanaged<Real [NP][NP]>(buffer);
  buffer += NP*NP;

  m_mp = ExecViewUnmanaged<Real [NP][NP]>(buffer);
  buffer += NP*NP;
  m_spheremp = ExecViewUnmanaged<Real [NP][NP]>(buffer);
  buffer += NP*NP;
  m_rspheremp = ExecViewUnmanaged<Real [NP][NP]>(buffer);
  buffer += NP*NP;

  m_metdet = ExecViewUnmanaged<Real [NP][NP]>(buffer);
  buffer += NP*NP;
  m_metinv = ExecViewUnmanaged<Real [2][2][NP][NP]>(buffer);
  buffer += 2*2*NP*NP;

  m_d    = ExecViewUnmanaged<Real [2][2][NP][NP]>(buffer);
  buffer += 2*2*NP*NP;
  m_dinv = ExecViewUnmanaged<Real [2][2][NP][NP]>(buffer);
  buffer += 2*2*NP*NP;

  Scalar* sbuf = reinterpret_cast<Scalar*>(buffer);

  m_omega_p = ExecViewUnmanaged<Scalar [NP][NP][NUM_LEV]>(sbuf);
  sbuf += NP*NP*NUM_LEV;
  m_phi = ExecViewUnmanaged<Scalar [NP][NP][NUM_LEV]>(sbuf);
  sbuf += NP*NP*NUM_LEV;
  m_derived_vn0 = ExecViewUnmanaged<Scalar [2][NP][NP][NUM_LEV]>(sbuf);
  sbuf += 2*NP*NP*NUM_LEV;

  m_v = ExecViewUnmanaged<Scalar [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV]>(sbuf);
  sbuf += 2*NUM_TIME_LEVELS*NP*NP*NUM_LEV;
  m_t = ExecViewUnmanaged<Scalar [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(sbuf);
  sbuf += NUM_TIME_LEVELS*NP*NP*NUM_LEV;
  m_dp3d = ExecViewUnmanaged<Scalar [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(sbuf);
  sbuf += NUM_TIME_LEVELS*NP*NP*NUM_LEV;

  m_eta_dot_dpdn = ExecViewUnmanaged<Scalar [NP][NP][NUM_LEV]>(sbuf);
  sbuf += NP*NP*NUM_LEV;
  m_derived_dp = ExecViewUnmanaged<Scalar [NP][NP][NUM_LEV]>(sbuf);
  sbuf += NP*NP*NUM_LEV;
  m_derived_dpdiss_ave = ExecViewUnmanaged<Scalar [NP][NP][NUM_LEV]>(sbuf);
  sbuf += NP*NP*NUM_LEV;
  m_derived_divdp = ExecViewUnmanaged<Scalar [NP][NP][NUM_LEV]>(sbuf);
  sbuf += NP*NP*NUM_LEV;
  m_derived_divdp_proj = ExecViewUnmanaged<Scalar [NP][NP][NUM_LEV]>(sbuf);
  sbuf += NP*NP*NUM_LEV;
  m_derived_dpdiss_biharmonic = ExecViewUnmanaged<Scalar [NP][NP][NUM_LEV]>(sbuf);
  sbuf += NP*NP*NUM_LEV;

  buffer = reinterpret_cast<Real*>(sbuf);
  ExecViewUnmanaged<Real [NUM_TIME_LEVELS][NP][NP]> m_ps_v;
}

void Elements::init(const int num_elems) {
  m_num_elems = num_elems;

  // Allocating the internal buffer
  const size_t elem_size = Element::size();
  m_internal_buffer = ExecViewManaged<Real*>("",elem_size*num_elems);

  // Creating empty view of Element's
  m_elements = ExecViewManaged<Element*>("elements", m_num_elems);

  // Init-ing a host copy of m_elements, and then deep_copy-ing it to the device
  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Real* data = m_internal_buffer.data();
  for (int ie=0; ie<num_elems; ++ie) {
    h_elements(ie).init(data);
    data += elem_size;
  }
  Kokkos::deep_copy(m_elements,h_elements);

  buffers.init(num_elems);
}

void Elements::init_2d(CF90Ptr &D, CF90Ptr &Dinv, CF90Ptr &fcor,
                       CF90Ptr &mp, CF90Ptr &spheremp, CF90Ptr &rspheremp,
                       CF90Ptr &metdet, CF90Ptr &metinv, CF90Ptr &phis) {

  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);

  HostViewUnmanaged<const Real *[2][2][NP][NP]> h_D_f90         (D,         m_num_elems);
  HostViewUnmanaged<const Real *[2][2][NP][NP]> h_Dinv_f90      (Dinv,      m_num_elems);
  HostViewUnmanaged<const Real *      [NP][NP]> h_fcor_f90      (fcor,      m_num_elems);
  HostViewUnmanaged<const Real *      [NP][NP]> h_mp_f90        (mp,        m_num_elems);
  HostViewUnmanaged<const Real *      [NP][NP]> h_spheremp_f90  (spheremp,  m_num_elems);
  HostViewUnmanaged<const Real *      [NP][NP]> h_rspheremp_f90 (rspheremp, m_num_elems);
  HostViewUnmanaged<const Real *      [NP][NP]> h_metdet_f90    (metdet,    m_num_elems);
  HostViewUnmanaged<const Real *[2][2][NP][NP]> h_metinv_f90    (metinv,    m_num_elems);
  HostViewUnmanaged<const Real *      [NP][NP]> h_phis_f90      (phis,      m_num_elems);

  for (int ie=0; ie<m_num_elems; ++ie) {
    const Element& elem = h_elements(ie);
    sync_to_device(Homme::subview(h_D_f90,ie)        , elem.m_d        );
    sync_to_device(Homme::subview(h_Dinv_f90,ie)     , elem.m_dinv     );
    sync_to_device(Homme::subview(h_fcor_f90,ie)     , elem.m_fcor     );
    sync_to_device(Homme::subview(h_metinv_f90,ie)   , elem.m_metinv   );
    sync_to_device(Homme::subview(h_mp_f90,ie)       , elem.m_mp       );
    sync_to_device(Homme::subview(h_spheremp_f90,ie) , elem.m_spheremp );
    sync_to_device(Homme::subview(h_rspheremp_f90,ie), elem.m_rspheremp);
    sync_to_device(Homme::subview(h_metdet_f90,ie)   , elem.m_metdet   );
    sync_to_device(Homme::subview(h_metinv_f90,ie)   , elem.m_metinv   );
    sync_to_device(Homme::subview(h_phis_f90,ie)     , elem.m_phis     );
  }

  // NOTE: there is no need to copy h_elements into m_elements. They both store
  //       the same device views, so the deep_copy's above already copied the
  //       data from f90 ptrs into the device views of the given element
}

void Elements::random_init(const int num_elems, const Real max_pressure) {
  // arbitrary minimum value to generate and minimum determinant allowed
  constexpr const Real min_value = 0.015625;
  init(num_elems);
  std::random_device rd;
  std::mt19937_64 engine(rd());
  std::uniform_real_distribution<Real> random_dist(min_value, 1.0 / min_value);

  // Note: make sure you init hvcoord before calling this method!
  const auto& hvcoord = Context::singleton().get_hvcoord();

  // This ensures the pressure in a single column is monotonically increasing
  // and has fixed upper and lower values
  const auto make_pressure_partition = [=](
      HostViewUnmanaged<Scalar[NUM_LEV]> pt_pressure) {
    // Put in monotonic order
    std::sort(
        reinterpret_cast<Real *>(pt_pressure.data()),
        reinterpret_cast<Real *>(pt_pressure.data() + pt_pressure.size()));
    // Ensure none of the values are repeated
    for (int level = NUM_PHYSICAL_LEV - 1; level > 0; --level) {
      const int prev_ilev = (level - 1) / VECTOR_SIZE;
      const int prev_vlev = (level - 1) % VECTOR_SIZE;
      const int cur_ilev = level / VECTOR_SIZE;
      const int cur_vlev = level % VECTOR_SIZE;
      // Need to try again if these are the same or if the thickness is too
      // small
      if (pt_pressure(cur_ilev)[cur_vlev] <=
          pt_pressure(prev_ilev)[prev_vlev] +
              min_value * std::numeric_limits<Real>::epsilon()) {
        return false;
      }
    }
    // We know the minimum thickness of a layer is min_value * epsilon
    // (due to floating point), so set the bottom layer thickness to that,
    // and subtract that from the top layer
    // This ensures that the total sum is max_pressure
    pt_pressure(0)[0] = min_value * std::numeric_limits<Real>::epsilon();
    const int top_ilev = (NUM_PHYSICAL_LEV - 1) / VECTOR_SIZE;
    const int top_vlev = (NUM_PHYSICAL_LEV - 1) % VECTOR_SIZE;
    // Note that this may not actually change the top level pressure
    // This is okay, because we only need to approximately sum to max_pressure
    pt_pressure(top_ilev)[top_vlev] = max_pressure - pt_pressure(0)[0];
    for (int e_vlev = top_vlev + 1; e_vlev < VECTOR_SIZE; ++e_vlev) {
      pt_pressure(top_ilev)[e_vlev] = std::numeric_limits<Real>::quiet_NaN();
    }
    // Now compute the interval thicknesses
    for (int level = NUM_PHYSICAL_LEV - 1; level > 0; --level) {
      const int prev_ilev = (level - 1) / VECTOR_SIZE;
      const int prev_vlev = (level - 1) % VECTOR_SIZE;
      const int cur_ilev = level / VECTOR_SIZE;
      const int cur_vlev = level % VECTOR_SIZE;
      pt_pressure(cur_ilev)[cur_vlev] -= pt_pressure(prev_ilev)[prev_vlev];
    }
    return true;
  };

  std::uniform_real_distribution<Real> pressure_pdf(min_value, max_pressure);

  // Lambdas used to constrain the metric tensor and its inverse
  const auto compute_det = [](HostViewUnmanaged<Real[2][2]> mtx) {
    return mtx(0, 0) * mtx(1, 1) - mtx(0, 1) * mtx(1, 0);
  };

  const auto constrain_det = [=](HostViewUnmanaged<Real[2][2]> mtx) {
    Real determinant = compute_det(mtx);
    // We want to ensure both the metric tensor and its inverse have reasonable
    // determinants
    if (determinant > min_value && determinant < 1.0 / min_value) {
      return true;
    } else {
      return false;
    }
  };

  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  HostViewManaged<Real[2][2]> h_matrix("single host metric matrix");

  for (int ie=0; ie<m_num_elems; ++ie) {
    Element& elem = h_elements(ie);
    genRandArray(elem.m_fcor, engine, random_dist);
    genRandArray(elem.m_mp, engine, random_dist);
    genRandArray(elem.m_spheremp, engine, random_dist);
    genRandArray(elem.m_rspheremp, engine, random_dist);
    genRandArray(elem.m_metdet, engine, random_dist);
    genRandArray(elem.m_metinv, engine, random_dist);
    genRandArray(elem.m_phis, engine, random_dist);

    genRandArray(elem.m_omega_p, engine, random_dist);
    genRandArray(elem.m_phi, engine, random_dist);
    genRandArray(elem.m_derived_vn0, engine, random_dist);

    genRandArray(elem.m_v, engine, random_dist);
    genRandArray(elem.m_t, engine, random_dist);

    // Generate ps_v so that it is >> ps0.
    genRandArray(elem.m_ps_v, engine, std::uniform_real_distribution<Real>(100*hvcoord.ps0,1000*hvcoord.ps0));


    // 2d tensors
    // Generating lots of matrices with reasonable determinants can be difficult
    // So instead of generating them all at once and verifying they're correct,
    // generate them one at a time, verifying them individually

    ExecViewManaged<Real [2][2][NP][NP]>::HostMirror h_d =
        Kokkos::create_mirror_view(elem.m_d);
    ExecViewManaged<Real [2][2][NP][NP]>::HostMirror h_dinv =
        Kokkos::create_mirror_view(elem.m_dinv);

    Real dp3d_min = std::numeric_limits<Real>::max();
    // Because this constraint is difficult to satisfy for all of the tensors,
    // incrementally generate the view
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
          ExecViewUnmanaged<Scalar[NUM_LEV]> pt_dp3d =
              Homme::subview(elem.m_dp3d, tl, igp, jgp);
          genRandArray(pt_dp3d, engine, pressure_pdf, make_pressure_partition);
          auto h_dp3d = Kokkos::create_mirror_view(pt_dp3d);
          Kokkos::deep_copy(h_dp3d,pt_dp3d);
          for (int ilev=0; ilev<NUM_LEV; ++ilev) {
            for (int iv=0; iv<VECTOR_SIZE; ++iv) {
              dp3d_min = std::min(dp3d_min,h_dp3d(ilev)[iv]);
            }
          }
        }
        genRandArray(h_matrix, engine, random_dist, constrain_det);
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j) {
            h_d(ie, i, j, igp, jgp) = h_matrix(i, j);
          }
        }
        const Real determinant = compute_det(h_matrix);
        h_dinv(ie, 0, 0, igp, jgp) = h_matrix(1, 1) / determinant;
        h_dinv(ie, 1, 0, igp, jgp) = -h_matrix(1, 0) / determinant;
        h_dinv(ie, 0, 1, igp, jgp) = -h_matrix(0, 1) / determinant;
        h_dinv(ie, 1, 1, igp, jgp) = h_matrix(0, 0) / determinant;
      }
    }
    Kokkos::deep_copy(elem.m_d, h_d);
    Kokkos::deep_copy(elem.m_dinv, h_dinv);

    // Generate eta_dot_dpdn so that it is << dp3d
    genRandArray(elem.m_eta_dot_dpdn, engine, std::uniform_real_distribution<Real>(0.01*dp3d_min,0.1*dp3d_min));
  }
}

void Elements::pull_from_f90_pointers(
    CF90Ptr &state_v, CF90Ptr &state_t, CF90Ptr &state_dp3d,
    CF90Ptr &derived_phi, CF90Ptr &derived_omega_p,
    CF90Ptr &derived_v, CF90Ptr &derived_eta_dot_dpdn) {
  pull_3d(derived_phi, derived_omega_p, derived_v);
  pull_4d(state_v, state_t, state_dp3d);
  pull_eta_dot(derived_eta_dot_dpdn);
}

void Elements::pull_3d(CF90Ptr &derived_phi, CF90Ptr &derived_omega_p, CF90Ptr &derived_v) {
  HostViewUnmanaged<const Real *[NUM_PHYSICAL_LEV]   [NP][NP]> derived_phi_f90(derived_phi,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_PHYSICAL_LEV]   [NP][NP]> derived_omega_p_f90(derived_omega_p,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_PHYSICAL_LEV][2][NP][NP]> derived_v_f90(derived_v,m_num_elems);

  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Kokkos::deep_copy(h_elements,m_elements);
  for (int ie=0; ie<m_num_elems; ++ie) {
    sync_to_device(Homme::subview(derived_phi_f90,ie),     h_elements(ie).m_phi);
    sync_to_device(Homme::subview(derived_omega_p_f90,ie), h_elements(ie).m_omega_p);
    sync_to_device(Homme::subview(derived_v_f90,ie),       h_elements(ie).m_derived_vn0);
  }
}

void Elements::pull_4d(CF90Ptr &state_v, CF90Ptr &state_t, CF90Ptr &state_dp3d) {
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV]   [NP][NP]> state_t_f90    (state_t,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV]   [NP][NP]> state_dp3d_f90 (state_dp3d,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]> state_v_f90    (state_v,m_num_elems);

  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Kokkos::deep_copy(h_elements,m_elements);
  for (int ie=0; ie<m_num_elems; ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      sync_to_device(Homme::subview(state_t_f90,ie,tl),    Homme::subview(h_elements(ie).m_t,tl));
      sync_to_device(Homme::subview(state_dp3d_f90,ie,tl), Homme::subview(h_elements(ie).m_dp3d,tl));
      sync_to_device(Homme::subview(state_v_f90,ie,tl),    Homme::subview(h_elements(ie).m_v,tl));
    }
  }
}

void Elements::pull_eta_dot(CF90Ptr &derived_eta_dot_dpdn) {
  HostViewUnmanaged<const Real *[NUM_INTERFACE_LEV][NP][NP]> eta_dot_dpdn_f90(derived_eta_dot_dpdn,m_num_elems);
  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Kokkos::deep_copy(h_elements,m_elements);
  for (int ie=0; ie<m_num_elems; ++ie) {
    sync_to_device_i2p(Homme::subview(eta_dot_dpdn_f90,ie), h_elements(ie).m_eta_dot_dpdn);
  }
}

void Elements::push_to_f90_pointers(F90Ptr &state_v, F90Ptr &state_t,
                                    F90Ptr &state_dp3d, F90Ptr &derived_phi,
                                    F90Ptr &derived_omega_p, F90Ptr &derived_v,
                                    F90Ptr &derived_eta_dot_dpdn) const {
  push_3d(derived_phi, derived_omega_p, derived_v);
  push_4d(state_v, state_t, state_dp3d);
  push_eta_dot(derived_eta_dot_dpdn);
}

void Elements::push_3d(F90Ptr &derived_phi, F90Ptr &derived_omega_p, F90Ptr &derived_v) const {
  HostViewUnmanaged<Real *[NUM_PHYSICAL_LEV]   [NP][NP]> derived_phi_f90(derived_phi,m_num_elems);
  HostViewUnmanaged<Real *[NUM_PHYSICAL_LEV]   [NP][NP]> derived_omega_p_f90(derived_omega_p,m_num_elems);
  HostViewUnmanaged<Real *[NUM_PHYSICAL_LEV][2][NP][NP]> derived_v_f90(derived_v,m_num_elems);

  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Kokkos::deep_copy(h_elements,m_elements);
  for (int ie=0; ie<m_num_elems; ++ie) {
    sync_to_host(h_elements(ie).m_phi,         Homme::subview(derived_phi_f90,ie));
    sync_to_host(h_elements(ie).m_omega_p,     Homme::subview(derived_omega_p_f90,ie));
    sync_to_host(h_elements(ie).m_derived_vn0, Homme::subview(derived_v_f90,ie));
  }
}

void Elements::push_4d(F90Ptr &state_v, F90Ptr &state_t, F90Ptr &state_dp3d) const {
  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV]   [NP][NP]> state_t_f90    (state_t,m_num_elems);
  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV]   [NP][NP]> state_dp3d_f90 (state_dp3d,m_num_elems);
  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]> state_v_f90    (state_v,m_num_elems);

  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Kokkos::deep_copy(h_elements,m_elements);
  for (int ie=0; ie<m_num_elems; ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      sync_to_host(Homme::subview(h_elements(ie).m_t,tl),    Homme::subview(state_t_f90,ie,tl));
      sync_to_host(Homme::subview(h_elements(ie).m_dp3d,tl), Homme::subview(state_dp3d_f90,ie,tl));
      sync_to_host(Homme::subview(h_elements(ie).m_v,tl),    Homme::subview(state_v_f90,ie,tl));
    }
  }
}

void Elements::push_eta_dot(F90Ptr &derived_eta_dot_dpdn) const {
  HostViewUnmanaged<Real *[NUM_INTERFACE_LEV][NP][NP]> eta_dot_dpdn_f90(derived_eta_dot_dpdn,m_num_elems);
  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Kokkos::deep_copy(h_elements,m_elements);
  for (int ie=0; ie<m_num_elems; ++ie) {
    sync_to_host_p2i(h_elements(ie).m_eta_dot_dpdn,Homme::subview(eta_dot_dpdn_f90,ie));
  }
}

void Elements::d(Real *d_ptr, int ie) const {
  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Kokkos::deep_copy(h_elements,m_elements);

  HostViewUnmanaged<Real[2][2][NP][NP]> d_wrapper(d_ptr);
  sync_to_host(h_elements(ie).m_d,d_wrapper);
}

void Elements::dinv(Real *dinv_ptr, int ie) const {
  ExecViewManaged<Element*>::HostMirror h_elements = Kokkos::create_mirror_view(m_elements);
  Kokkos::deep_copy(h_elements,m_elements);

  HostViewUnmanaged<Real[2][2][NP][NP]> dinv_wrapper(dinv_ptr);
  sync_to_host(h_elements(ie).m_dinv,dinv_wrapper);
}

void Elements::BufferViews::init(const int num_elems) {
  pressure =
      ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("Pressure buffer", num_elems);
  pressure_grad = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>(
      "Gradient of pressure", num_elems);
  temperature_virt = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Virtual Temperature", num_elems);
  temperature_grad = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>(
      "Gradient of temperature", num_elems);
  omega_p = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Omega_P = omega/pressure = (Dp/Dt)/pressure", num_elems);
  vdp = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("(u,v)*dp", num_elems);
  div_vdp = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Divergence of dp3d * (u,v)", num_elems);
  ephi = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Kinetic Energy + Geopotential Energy", num_elems);
  energy_grad = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>(
      "Gradient of ephi", num_elems);
  vorticity =
      ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("Vorticity", num_elems);

  ttens  = ExecViewManaged<Scalar*    [NP][NP][NUM_LEV]>("Temporary for temperature",num_elems);
  dptens = ExecViewManaged<Scalar*    [NP][NP][NUM_LEV]>("Temporary for dp3d",num_elems);
  vtens  = ExecViewManaged<Scalar* [2][NP][NP][NUM_LEV]>("Temporary for velocity",num_elems);

  vstar = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("buffer for (flux v)/dp",
       num_elems);
  qwrk      = ExecViewManaged<Scalar * [QSIZE_D][2][NP][NP][NUM_LEV]>(
      "work buffer for tracers", num_elems);
  dpdissk = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "dpdissk", num_elems);

  preq_buf = ExecViewManaged<Real * [NP][NP]>("Preq Buffer", num_elems);

  sdot_sum = ExecViewManaged<Real * [NP][NP]>("Sdot sum buffer", num_elems);

  div_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("Divergence Buffer",
                                                           num_elems);

  lapl_buf_1 = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("Scalar laplacian Buffer", num_elems);
  lapl_buf_2 = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("Scalar laplacian Buffer", num_elems);
  v_vadv_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("v_vadv buffer",
                                                              num_elems);
  t_vadv_buf = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("t_vadv buffer",
                                                           num_elems);
  eta_dot_dpdn_buf = ExecViewManaged<Scalar * [NP][NP][NUM_LEV_P]>("eta_dot_dpdpn buffer",
                                                                   num_elems);

  kernel_start_times = ExecViewManaged<clock_t *>("Start Times", num_elems);
  kernel_end_times = ExecViewManaged<clock_t *>("End Times", num_elems);
}

} // namespace Homme
