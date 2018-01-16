#include "Elements.hpp"
#include "Utility.hpp"

#include <limits>
#include <random>
#include <assert.h>

namespace Homme {

void Elements::init(const int num_elems, const bool rsplit0) {
  m_num_elems = num_elems;

  buffers.init(num_elems, rsplit0);

  m_fcor = ExecViewManaged<Real * [NP][NP]>("FCOR", m_num_elems);
  m_spheremp = ExecViewManaged<Real * [NP][NP]>("SPHEREMP", m_num_elems);
  m_rspheremp = ExecViewManaged<Real * [NP][NP]>("RSPHEREMP", m_num_elems);
  m_metdet = ExecViewManaged<Real * [NP][NP]>("METDET", m_num_elems);
  m_phis = ExecViewManaged<Real * [NP][NP]>("PHIS", m_num_elems);

  //D is not a metric tensor, D^tD is
  m_d =
      ExecViewManaged<Real * [2][2][NP][NP]>("matrix D", m_num_elems);
  m_dinv = ExecViewManaged<Real * [2][2][NP][NP]>(
      "DInv - inverse of matrix D", m_num_elems);

  m_omega_p =
      ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("Omega P", m_num_elems);
  m_phi = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("PHI", m_num_elems);
  m_derived_un0 = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Flux for u", m_num_elems);
  m_derived_vn0 = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Flux for v", m_num_elems);

  m_u = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(
      "Velocity u", m_num_elems);
  m_v = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(
      "Velocity v", m_num_elems);
  m_t = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(
      "Temperature", m_num_elems);
  m_dp3d = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(
      "DP3D", m_num_elems);

  m_qdp =
      ExecViewManaged<Scalar * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>(
          "qdp", m_num_elems);
  m_eta_dot_dpdn = ExecViewManaged<Scalar * [NP][NP][NUM_LEV_P]>("eta_dot_dpdn",
                                                                 m_num_elems);
}

void Elements::init_2d(CF90Ptr &D, CF90Ptr &Dinv, CF90Ptr &fcor, CF90Ptr &spheremp,
                       CF90Ptr &rspheremp, CF90Ptr &metdet, CF90Ptr &phis) {
  int k_scalars = 0;
  int k_tensors = 0;
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_fcor =
      Kokkos::create_mirror_view(m_fcor);
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_metdet =
      Kokkos::create_mirror_view(m_metdet);
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_spheremp =
      Kokkos::create_mirror_view(m_spheremp);
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_rspheremp =
      Kokkos::create_mirror_view(m_rspheremp);
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_phis =
      Kokkos::create_mirror_view(m_phis);

  ExecViewManaged<Real *[2][2][NP][NP]>::HostMirror h_d =
      Kokkos::create_mirror_view(m_d);
  ExecViewManaged<Real *[2][2][NP][NP]>::HostMirror h_dinv =
      Kokkos::create_mirror_view(m_dinv);

  // 2d scalars
  for (int ie = 0; ie < m_num_elems; ++ie) {
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp, ++k_scalars) {
        h_fcor(ie, igp, jgp) = fcor[k_scalars];
        h_spheremp(ie, igp, jgp) = spheremp[k_scalars];
        h_rspheremp(ie, igp, jgp) = rspheremp[k_scalars];
        h_metdet(ie, igp, jgp) = metdet[k_scalars];
        h_phis(ie, igp, jgp) = phis[k_scalars];
      }
    }
  }

  // 2d tensors
  for (int ie = 0; ie < m_num_elems; ++ie) {
    for (int idim = 0; idim < 2; ++idim) {
      for (int jdim = 0; jdim < 2; ++jdim) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp, ++k_tensors) {
            h_d(ie, idim, jdim, igp, jgp) = D[k_tensors];
            h_dinv(ie, idim, jdim, igp, jgp) = Dinv[k_tensors];
          }
        }
      }
    }
  }

  Kokkos::deep_copy(m_fcor, h_fcor);
  Kokkos::deep_copy(m_metdet, h_metdet);
  Kokkos::deep_copy(m_spheremp, h_spheremp);
  Kokkos::deep_copy(m_rspheremp, h_rspheremp);
  Kokkos::deep_copy(m_phis, h_phis);

  Kokkos::deep_copy(m_d, h_d);
  Kokkos::deep_copy(m_dinv, h_dinv);
}

void Elements::random_init(const int num_elems, const Real max_pressure) {
  // arbitrary minimum value to generate and minimum determinant allowed
  constexpr const Real min_value = 0.015625;
  init(num_elems, true);
  std::random_device rd;
  std::mt19937_64 engine(rd());
  std::uniform_real_distribution<Real> random_dist(min_value, 1.0 / min_value);

  genRandArray(m_fcor, engine, random_dist);
  genRandArray(m_spheremp, engine, random_dist);
  genRandArray(m_rspheremp, engine, random_dist);
  genRandArray(m_metdet, engine, random_dist);
  genRandArray(m_phis, engine, random_dist);

  genRandArray(m_omega_p, engine, random_dist);
  genRandArray(m_phi, engine, random_dist);
  genRandArray(m_derived_un0, engine, random_dist);
  genRandArray(m_derived_vn0, engine, random_dist);

  genRandArray(m_u, engine, random_dist);
  genRandArray(m_v, engine, random_dist);
  genRandArray(m_t, engine, random_dist);

  genRandArray(m_qdp, engine, random_dist);
  genRandArray(m_eta_dot_dpdn, engine, random_dist);

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

  // 2d tensors
  // Generating lots of matrices with reasonable determinants can be difficult
  // So instead of generating them all at once and verifying they're correct,
  // generate them one at a time, verifying them individually
  HostViewManaged<Real[2][2]> h_matrix("single host metric matrix");

  ExecViewManaged<Real *[2][2][NP][NP]>::HostMirror h_d =
      Kokkos::create_mirror_view(m_d);
  ExecViewManaged<Real *[2][2][NP][NP]>::HostMirror h_dinv =
      Kokkos::create_mirror_view(m_dinv);

  for (int ie = 0; ie < m_num_elems; ++ie) {
    // Because this constraint is difficult to satisfy for all of the tensors,
    // incrementally generate the view
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
          ExecViewUnmanaged<Scalar[NUM_LEV]> pt_dp3d =
              Homme::subview(m_dp3d, ie, tl, igp, jgp);
          genRandArray(pt_dp3d, engine, pressure_pdf, make_pressure_partition);
        }
        genRandArray(h_matrix, engine, random_dist, constrain_det);
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j) {
            h_d(ie, i, j, igp, jgp) = h_matrix(i, j);
          }
        }

//do quiet_nan for the tail of eta_
//are there any other fields that are on interfaces?
//       for (int ilevel = NUM_PHYSICAL_LEV+1; ilevel < NUM_LEV_P*VECTOR_SIZE; ilevel++){ 
//         int ilev = ilevel / VECTOR_SIZE;
//         int ivector = ilevel % VECTOR_SIZE;
//         h_eta_dot_dpdn(ie, igp, jgp, ilev)[ivector] = std::numeric_limits<Real>::quiet_NaN();
//       }

        const Real determinant = compute_det(h_matrix);
        h_dinv(ie, 0, 0, igp, jgp) = h_matrix(1, 1) / determinant;
        h_dinv(ie, 1, 0, igp, jgp) = -h_matrix(1, 0) / determinant;
        h_dinv(ie, 0, 1, igp, jgp) = -h_matrix(0, 1) / determinant;
        h_dinv(ie, 1, 1, igp, jgp) = h_matrix(0, 0) / determinant;
      }
    }
  }

  Kokkos::deep_copy(m_d, h_d);
  Kokkos::deep_copy(m_dinv, h_dinv);
  return;
}

void Elements::pull_from_f90_pointers(
    CF90Ptr &state_v, CF90Ptr &state_t, CF90Ptr &state_dp3d,
    CF90Ptr &derived_phi, CF90Ptr &derived_omega_p,
    CF90Ptr &derived_v, CF90Ptr &derived_eta_dot_dpdn, CF90Ptr &state_qdp) {
  pull_3d(derived_phi, derived_omega_p, derived_v);
  pull_4d(state_v, state_t, state_dp3d);
  pull_eta_dot(derived_eta_dot_dpdn);
  pull_qdp(state_qdp);
}

void Elements::pull_3d(CF90Ptr &derived_phi, 
                       CF90Ptr &derived_omega_p, CF90Ptr &derived_v) {
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_omega_p =
      Kokkos::create_mirror_view(m_omega_p);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_phi =
      Kokkos::create_mirror_view(m_phi);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_derived_un0 =
      Kokkos::create_mirror_view(m_derived_un0);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_derived_vn0 =
      Kokkos::create_mirror_view(m_derived_vn0);
  for (int ie = 0, k_3d_scalars = 0, k_3d_vectors = 0; ie < m_num_elems; ++ie) {
    for (int ilevel = 0; ilevel < NUM_PHYSICAL_LEV; ++ilevel) {
      int ilev = ilevel / VECTOR_SIZE;
      int ivector = ilevel % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp, ++k_3d_scalars) {
          h_omega_p(ie, igp, jgp, ilev)[ivector] =
              derived_omega_p[k_3d_scalars];
          h_phi(ie, igp, jgp, ilev)[ivector] = derived_phi[k_3d_scalars];
        }
      }

      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp, ++k_3d_vectors) {
          h_derived_un0(ie, igp, jgp, ilev)[ivector] = derived_v[k_3d_vectors];
        }
      }
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp, ++k_3d_vectors) {
          h_derived_vn0(ie, igp, jgp, ilev)[ivector] = derived_v[k_3d_vectors];
        }
      }
    }
  }
  Kokkos::deep_copy(m_omega_p, h_omega_p);
  Kokkos::deep_copy(m_phi, h_phi);
  Kokkos::deep_copy(m_derived_un0, h_derived_un0);
  Kokkos::deep_copy(m_derived_vn0, h_derived_vn0);
}

void Elements::pull_4d(CF90Ptr &state_v, CF90Ptr &state_t,
                       CF90Ptr &state_dp3d) {
  ExecViewManaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror h_u =
      Kokkos::create_mirror_view(m_u);
  ExecViewManaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror h_v =
      Kokkos::create_mirror_view(m_v);
  ExecViewManaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror h_t =
      Kokkos::create_mirror_view(m_t);
  ExecViewManaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror
  h_dp3d = Kokkos::create_mirror_view(m_dp3d);
  for (int ie = 0, k_4d_scalars = 0, k_4d_vectors = 0; ie < m_num_elems; ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int ilevel = 0; ilevel < NUM_PHYSICAL_LEV; ++ilevel) {
        int ilev = ilevel / VECTOR_SIZE;
        int ivector = ilevel % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp, ++k_4d_scalars) {
            h_dp3d(ie, tl, igp, jgp, ilev)[ivector] = state_dp3d[k_4d_scalars];
            h_t(ie, tl, igp, jgp, ilev)[ivector] = state_t[k_4d_scalars];
          }
        }

        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp, ++k_4d_vectors) {
            h_u(ie, tl, igp, jgp, ilev)[ivector] = state_v[k_4d_vectors];
          }
        }
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp, ++k_4d_vectors) {
            h_v(ie, tl, igp, jgp, ilev)[ivector] = state_v[k_4d_vectors];
          }
        }
      }
    }
  }
  Kokkos::deep_copy(m_u, h_u);
  Kokkos::deep_copy(m_v, h_v);
  Kokkos::deep_copy(m_t, h_t);
  Kokkos::deep_copy(m_dp3d, h_dp3d);
}

void Elements::pull_eta_dot(CF90Ptr &derived_eta_dot_dpdn) {

  ExecViewManaged<Scalar *[NP][NP][NUM_LEV_P]>::HostMirror h_eta_dot_dpdn =
      Kokkos::create_mirror_view(m_eta_dot_dpdn);
  for (int ie = 0, k_eta_dot_dp_dn = 0; ie < m_num_elems; ++ie) {
    // Note: we must process only NUM_PHYSICAL_LEV, since the F90
    //       ptr has that size. If we looped on levels packs (0 to NUM_LEV_P)
    //       and on vector length, we would have to treat the last pack with
    // care
    for (int ilevel = 0; ilevel < NUM_INTERFACE_LEV; ++ilevel) {
      int ilev = ilevel / VECTOR_SIZE;
      int ivector = ilevel % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp, ++k_eta_dot_dp_dn) {
          h_eta_dot_dpdn(ie, igp, jgp, ilev)[ivector] =
              derived_eta_dot_dpdn[k_eta_dot_dp_dn];
        }
      }
    }
  }
  Kokkos::deep_copy(m_eta_dot_dpdn, h_eta_dot_dpdn);
}

void Elements::pull_qdp(CF90Ptr &state_qdp) {
  ExecViewManaged<
      Scalar *[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>::HostMirror h_qdp =
      Kokkos::create_mirror_view(m_qdp);
  for (int ie = 0, k_qdp = 0; ie < m_num_elems; ++ie) {
    for (int qni = 0; qni < Q_NUM_TIME_LEVELS; ++qni) {
      for (int iq = 0; iq < QSIZE_D; ++iq) {
        for (int ilevel = 0; ilevel < NUM_PHYSICAL_LEV; ++ilevel) {
          int ilev = ilevel / VECTOR_SIZE;
          int ivector = ilevel % VECTOR_SIZE;
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp, ++k_qdp) {
              h_qdp(ie, qni, iq, igp, jgp, ilev)[ivector] = state_qdp[k_qdp];
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy(m_qdp, h_qdp);
}

void Elements::push_to_f90_pointers(F90Ptr &state_v, F90Ptr &state_t,
                                    F90Ptr &state_dp3d, F90Ptr &derived_phi,
                                    F90Ptr &derived_omega_p, F90Ptr &derived_v,
                                    F90Ptr &derived_eta_dot_dpdn,
                                    F90Ptr &state_qdp) const {
  push_3d(derived_phi, derived_omega_p, derived_v);
  push_4d(state_v, state_t, state_dp3d);
  push_eta_dot(derived_eta_dot_dpdn);
  push_qdp(state_qdp);
}

void Elements::push_3d(F90Ptr &derived_phi, 
                       F90Ptr &derived_omega_p, F90Ptr &derived_v) const {
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_omega_p =
      Kokkos::create_mirror_view(m_omega_p);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_phi =
      Kokkos::create_mirror_view(m_phi);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_derived_un0 =
      Kokkos::create_mirror_view(m_derived_un0);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_derived_vn0 =
      Kokkos::create_mirror_view(m_derived_vn0);

  Kokkos::deep_copy(h_omega_p, m_omega_p);
  Kokkos::deep_copy(h_phi, m_phi);
  Kokkos::deep_copy(h_derived_un0, m_derived_un0);
  Kokkos::deep_copy(h_derived_vn0, m_derived_vn0);
  for (int ie = 0, k_3d_scalars = 0, k_3d_vectors = 0; ie < m_num_elems; ++ie) {
    for (int ilevel = 0; ilevel < NUM_PHYSICAL_LEV; ++ilevel) {
      int ilev = ilevel / VECTOR_SIZE;
      int ivector = ilevel % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp, ++k_3d_scalars) {
          derived_omega_p[k_3d_scalars] =
              h_omega_p(ie, igp, jgp, ilev)[ivector];
          derived_phi[k_3d_scalars] = h_phi(ie, igp, jgp, ilev)[ivector];
        }
      }

      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp, ++k_3d_vectors) {
          derived_v[k_3d_vectors] = h_derived_un0(ie, igp, jgp, ilev)[ivector];
        }
      }
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp, ++k_3d_vectors) {
          derived_v[k_3d_vectors] = h_derived_vn0(ie, igp, jgp, ilev)[ivector];
        }
      }
    }
  }
}

void Elements::push_4d(F90Ptr &state_v, F90Ptr &state_t,
                       F90Ptr &state_dp3d) const {
  ExecViewManaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror h_u =
      Kokkos::create_mirror_view(m_u);
  ExecViewManaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror h_v =
      Kokkos::create_mirror_view(m_v);
  ExecViewManaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror h_t =
      Kokkos::create_mirror_view(m_t);
  ExecViewManaged<Scalar *[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror
  h_dp3d = Kokkos::create_mirror_view(m_dp3d);
  Kokkos::deep_copy(h_u, m_u);
  Kokkos::deep_copy(h_v, m_v);
  Kokkos::deep_copy(h_t, m_t);
  Kokkos::deep_copy(h_dp3d, m_dp3d);
  for (int ie = 0, k_4d_scalars = 0, k_4d_vectors = 0; ie < m_num_elems; ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int ilevel = 0; ilevel < NUM_PHYSICAL_LEV; ++ilevel) {
        int ilev = ilevel / VECTOR_SIZE;
        int ivector = ilevel % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp, ++k_4d_scalars) {
            state_dp3d[k_4d_scalars] = h_dp3d(ie, tl, igp, jgp, ilev)[ivector];
            state_t[k_4d_scalars] = h_t(ie, tl, igp, jgp, ilev)[ivector];
          }
        }

        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp, ++k_4d_vectors) {
            state_v[k_4d_vectors] = h_u(ie, tl, igp, jgp, ilev)[ivector];
          }
        }
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp, ++k_4d_vectors) {
            state_v[k_4d_vectors] = h_v(ie, tl, igp, jgp, ilev)[ivector];
          }
        }
      }
    }
  }
}

void Elements::push_eta_dot(F90Ptr &derived_eta_dot_dpdn) const {
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV_P]>::HostMirror h_eta_dot_dpdn =
      Kokkos::create_mirror_view(m_eta_dot_dpdn);
  Kokkos::deep_copy(h_eta_dot_dpdn, m_eta_dot_dpdn);
  int k_eta_dot_dp_dn = 0;
  for (int ie = 0; ie < m_num_elems; ++ie) {
    // Note: we must process only NUM_PHYSICAL_LEV, since the F90
    //       ptr has that size. If we looped on levels packs (0 to NUM_LEV_P)
    //       and on vector length, we would have to treat the last pack with
    // care
    for (int ilevel = 0; ilevel < NUM_INTERFACE_LEV; ++ilevel) {
      int ilev = ilevel / VECTOR_SIZE;
      int ivector = ilevel % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp, ++k_eta_dot_dp_dn) {
          derived_eta_dot_dpdn[k_eta_dot_dp_dn] =
              h_eta_dot_dpdn(ie, igp, jgp, ilev)[ivector];
        }
      }
    }
  }
}

void Elements::push_qdp(F90Ptr &state_qdp) const {
  ExecViewManaged<
      Scalar *[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>::HostMirror h_qdp =
      Kokkos::create_mirror_view(m_qdp);
  Kokkos::deep_copy(h_qdp, m_qdp);
  for (int ie = 0, k_qdp = 0; ie < m_num_elems; ++ie) {
    for (int qni = 0; qni < Q_NUM_TIME_LEVELS; ++qni) {
      for (int iq = 0; iq < QSIZE_D; ++iq) {
        for (int ilevel = 0; ilevel < NUM_PHYSICAL_LEV; ++ilevel) {
          int ilev = ilevel / VECTOR_SIZE;
          int ivector = ilevel % VECTOR_SIZE;
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp, ++k_qdp) {
              state_qdp[k_qdp] = h_qdp(ie, qni, iq, igp, jgp, ilev)[ivector];
            }
          }
        }
      }
    }
  }
}

void Elements::d(Real *d_ptr, int ie) const {
  ExecViewManaged<Real[2][2][NP][NP]> d_device = Kokkos::subview(
      m_d, ie, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  ExecViewManaged<Real[2][2][NP][NP]>::HostMirror
  d_host = Kokkos::create_mirror_view(d_device),
  d_wrapper(d_ptr);
  Kokkos::deep_copy(d_host, d_device);
  for (int m = 0; m < 2; ++m) {
    for (int n = 0; n < 2; ++n) {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          d_wrapper(m, n, jgp, igp) = d_host(n, m, igp, jgp);
        }
      }
    }
  }
}

void Elements::dinv(Real *dinv_ptr, int ie) const {
  ExecViewManaged<Real[2][2][NP][NP]> dinv_device = Kokkos::subview(
      m_dinv, ie, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  ExecViewManaged<Real[2][2][NP][NP]>::HostMirror dinv_host(dinv_ptr);
  Kokkos::deep_copy(dinv_host, dinv_device);
}

void Elements::BufferViews::init(const int num_elems, const bool rsplit0) {
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

  qtens = ExecViewManaged<Scalar * [QSIZE_D][NP][NP][NUM_LEV]>(
      "buffer for tracers", num_elems);
  vstar = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("buffer for (flux v)/dp",
                                                         num_elems);
  vstar_qdp = ExecViewManaged<Scalar * [QSIZE_D][2][NP][NP][NUM_LEV]>(
      "buffer for vstar*qdp", num_elems);
  qwrk      = ExecViewManaged<Scalar * [QSIZE_D][2][NP][NP][NUM_LEV]>(
      "work buffer for tracers", num_elems);

  preq_buf = ExecViewManaged<Real * [NP][NP]>("Preq Buffer", num_elems);

  sdot_sum = ExecViewManaged<Real * [NP][NP]>("Sdot sum buffer", num_elems);

  div_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("Divergence Buffer",
                                                           num_elems);
  grad_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("Gradient Buffer",
                                                            num_elems);
  vort_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("Vorticity Buffer",
                                                            num_elems);
  if (rsplit0) {
    v_vadv_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("v_vadv buffer",
                                                                num_elems);
    t_vadv_buf = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("t_vadv buffer",
                                                             num_elems);
    eta_dot_dpdn_buf = ExecViewManaged<Scalar * [NP][NP][NUM_LEV_P]>("eta_dot_dpdpn buffer",
                                                                     num_elems);
  }

  kernel_start_times = ExecViewManaged<clock_t *>("Start Times", num_elems);
  kernel_end_times = ExecViewManaged<clock_t *>("End Times", num_elems);
}

} // namespace Homme
