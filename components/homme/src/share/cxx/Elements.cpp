#include "Elements.hpp"
#include "Utility.hpp"

#include <random>

#include <assert.h>

namespace Homme {

void Elements::init(const int num_elems) {
  m_num_elems = num_elems;

  buffers.init(num_elems);

  m_fcor = ExecViewManaged<Real * [NP][NP]>("FCOR", m_num_elems);
  m_spheremp = ExecViewManaged<Real * [NP][NP]>("SPHEREMP", m_num_elems);
  m_rspheremp = ExecViewManaged<Real * [NP][NP]>("RSPHEREMP", m_num_elems);
  m_metdet = ExecViewManaged<Real * [NP][NP]>("METDET", m_num_elems);
  m_phis = ExecViewManaged<Real * [NP][NP]>("PHIS", m_num_elems);

  m_d =
      ExecViewManaged<Real * [2][2][NP][NP]>("D - metric tensor", m_num_elems);
  m_dinv = ExecViewManaged<Real * [2][2][NP][NP]>(
      "DInv - inverse metric tensor", m_num_elems);

  m_omega_p =
      ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("Omega P", m_num_elems);
  m_pecnd = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("PECND", m_num_elems);
  m_phi = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("PHI", m_num_elems);
  m_derived_un0 = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Derived Lateral Velocity 1", m_num_elems);
  m_derived_vn0 = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Derived Lateral Velocity 2", m_num_elems);

  m_u = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(
      "Lateral Velocity 1", m_num_elems);
  m_v = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(
      "Lateral Velocity 2", m_num_elems);
  m_t = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(
      "Temperature", m_num_elems);
  m_dp3d = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>(
      "DP3D", m_num_elems);

  m_qdp =
      ExecViewManaged<Scalar * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>(
          "qdp", m_num_elems);
  m_eta_dot_dpdn = ExecViewManaged<Scalar * [NP][NP][NUM_LEV_P]>("eta_dot_dpdn",
                                                                 m_num_elems);

  h_u    = Kokkos::create_mirror_view(m_u);
  h_v    = Kokkos::create_mirror_view(m_v);
  h_t    = Kokkos::create_mirror_view(m_t);
  h_dp3d = Kokkos::create_mirror_view(m_dp3d);
  h_qdp  = Kokkos::create_mirror_view(m_qdp);
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

void Elements::random_init(const int num_elems, const int seed) {
  std::mt19937_64 engine(seed);
  init(num_elems);
  constexpr const Real min_value = 0.015625;
  std::uniform_real_distribution<Real> random_dist(min_value, 1.0);

  ExecViewManaged<Real *[NP][NP]>::HostMirror h_fcor =
      Kokkos::create_mirror_view(m_fcor);
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_spheremp =
      Kokkos::create_mirror_view(m_spheremp);
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_rspheremp =
      Kokkos::create_mirror_view(m_rspheremp);
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_metdet =
      Kokkos::create_mirror_view(m_metdet);
  ExecViewManaged<Real *[NP][NP]>::HostMirror h_phis =
      Kokkos::create_mirror_view(m_phis);

  ExecViewManaged<Real *[2][2][NP][NP]>::HostMirror h_d =
      Kokkos::create_mirror_view(m_d);
  ExecViewManaged<Real *[2][2][NP][NP]>::HostMirror h_dinv =
      Kokkos::create_mirror_view(m_dinv);

  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_omega_p =
      Kokkos::create_mirror_view(m_omega_p);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_pecnd =
      Kokkos::create_mirror_view(m_pecnd);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_phi =
      Kokkos::create_mirror_view(m_phi);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_derived_un0 =
      Kokkos::create_mirror_view(m_derived_un0);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_derived_vn0 =
      Kokkos::create_mirror_view(m_derived_vn0);

  ExecViewManaged<Scalar *[NP][NP][NUM_LEV_P]>::HostMirror h_eta_dot_dpdn =
      Kokkos::create_mirror_view(m_eta_dot_dpdn);

  for (int ie = 0; ie < m_num_elems; ++ie) {
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        // 2d scalars
        h_fcor(ie, igp, jgp) = random_dist(engine);
        h_spheremp(ie, igp, jgp) = random_dist(engine);
        h_rspheremp(ie, igp, jgp) = random_dist(engine);
        h_metdet(ie, igp, jgp) = random_dist(engine);
        h_phis(ie, igp, jgp) = random_dist(engine);

        for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
          for (int vec = 0; vec < VECTOR_SIZE; ++vec) {
            // 3d scalars
            h_omega_p(ie, igp, jgp, ilev)[vec] = random_dist(engine);
            h_pecnd(ie, igp, jgp, ilev)[vec] = random_dist(engine);
            h_phi(ie, igp, jgp, ilev)[vec] = random_dist(engine);
            h_derived_un0(ie, igp, jgp, ilev)[vec] = random_dist(engine);
            h_derived_vn0(ie, igp, jgp, ilev)[vec] = random_dist(engine);

            // 4d scalars
            for (int timelevel = 0; timelevel < NUM_TIME_LEVELS; ++timelevel) {
              h_u(ie, timelevel, igp, jgp, ilev)[vec] = random_dist(engine);
              h_v(ie, timelevel, igp, jgp, ilev)[vec] = random_dist(engine);
              h_t(ie, timelevel, igp, jgp, ilev)[vec] = random_dist(engine);
              h_dp3d(ie, timelevel, igp, jgp, ilev)[vec] = random_dist(engine);
            }

            for (int q_timelevel = 0; q_timelevel < Q_NUM_TIME_LEVELS;
                 ++q_timelevel) {
              for (int i_q = 0; i_q < QSIZE_D; ++i_q) {
                h_qdp(ie, q_timelevel, i_q, igp, jgp, ilev)[vec] =
                    random_dist(engine);
              }
            }
          }
        }

        for (int ilev = 0; ilev < NUM_LEV_P; ++ilev) {
          for (int vec = 0; vec < VECTOR_SIZE; ++vec) {
            // 3d scalar at the interfaces of the levels
            h_eta_dot_dpdn(ie, igp, jgp, ilev)[vec] = random_dist(engine);
          }
        }

        Real determinant = 0.0;
        while (std::abs(determinant) < min_value) {
          // 2d tensors
          for (int idim = 0; idim < 2; ++idim) {
            for (int jdim = 0; jdim < 2; ++jdim) {
              h_d(ie, idim, jdim, igp, jgp) = random_dist(engine);
            }
          }
          determinant = h_d(ie, 0, 0, igp, jgp) * h_d(ie, 1, 1, igp, jgp) -
                        h_d(ie, 0, 1, igp, jgp) * h_d(ie, 1, 0, igp, jgp);
          h_dinv(ie, 0, 0, igp, jgp) = h_d(ie, 1, 1, igp, jgp) / determinant;
          h_dinv(ie, 0, 1, igp, jgp) = -h_d(ie, 1, 0, igp, jgp) / determinant;
          h_dinv(ie, 1, 0, igp, jgp) = -h_d(ie, 0, 1, igp, jgp) / determinant;
          h_dinv(ie, 1, 1, igp, jgp) = h_d(ie, 0, 0, igp, jgp) / determinant;
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

  Kokkos::deep_copy(m_omega_p, h_omega_p);
  Kokkos::deep_copy(m_pecnd, h_pecnd);
  Kokkos::deep_copy(m_phi, h_phi);
  Kokkos::deep_copy(m_derived_un0, h_derived_un0);
  Kokkos::deep_copy(m_derived_vn0, h_derived_vn0);

  Kokkos::deep_copy(m_u, h_u);
  Kokkos::deep_copy(m_v, h_v);
  Kokkos::deep_copy(m_t, h_t);
  Kokkos::deep_copy(m_dp3d, h_dp3d);

  Kokkos::deep_copy(m_eta_dot_dpdn, h_eta_dot_dpdn);
  return;
}

void Elements::pull_from_f90_pointers(
    CF90Ptr &state_v, CF90Ptr &state_t, CF90Ptr &state_dp3d,
    CF90Ptr &derived_phi, CF90Ptr &derived_pecnd, CF90Ptr &derived_omega_p,
    CF90Ptr &derived_v, CF90Ptr &derived_eta_dot_dpdn, CF90Ptr &state_qdp) {
  pull_3d(derived_phi, derived_pecnd, derived_omega_p, derived_v);
  pull_4d(state_v, state_t, state_dp3d);
  pull_eta_dot(derived_eta_dot_dpdn);
  pull_qdp(state_qdp);
}

void Elements::pull_3d(CF90Ptr &derived_phi, CF90Ptr &derived_pecnd,
                       CF90Ptr &derived_omega_p, CF90Ptr &derived_v) {
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_omega_p =
      Kokkos::create_mirror_view(m_omega_p);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_pecnd =
      Kokkos::create_mirror_view(m_pecnd);
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
          h_pecnd(ie, igp, jgp, ilev)[ivector] = derived_pecnd[k_3d_scalars];
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
  Kokkos::deep_copy(m_pecnd, h_pecnd);
  Kokkos::deep_copy(m_phi, h_phi);
  Kokkos::deep_copy(m_derived_un0, h_derived_un0);
  Kokkos::deep_copy(m_derived_vn0, h_derived_vn0);
}

void Elements::pull_4d(CF90Ptr &state_v, CF90Ptr &state_t,
                       CF90Ptr &state_dp3d) {
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
                                    F90Ptr &derived_pecnd,
                                    F90Ptr &derived_omega_p, F90Ptr &derived_v,
                                    F90Ptr &derived_eta_dot_dpdn,
                                    F90Ptr &state_qdp) const {
  push_3d(derived_phi, derived_pecnd, derived_omega_p, derived_v);
  push_4d(state_v, state_t, state_dp3d);
  push_eta_dot(derived_eta_dot_dpdn);
  push_qdp(state_qdp);
}

void Elements::push_3d(F90Ptr &derived_phi, F90Ptr &derived_pecnd,
                       F90Ptr &derived_omega_p, F90Ptr &derived_v) const {
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_omega_p =
      Kokkos::create_mirror_view(m_omega_p);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_pecnd =
      Kokkos::create_mirror_view(m_pecnd);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_phi =
      Kokkos::create_mirror_view(m_phi);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_derived_un0 =
      Kokkos::create_mirror_view(m_derived_un0);
  ExecViewManaged<Scalar *[NP][NP][NUM_LEV]>::HostMirror h_derived_vn0 =
      Kokkos::create_mirror_view(m_derived_vn0);

  Kokkos::deep_copy(h_omega_p, m_omega_p);
  Kokkos::deep_copy(h_pecnd, m_pecnd);
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
          derived_pecnd[k_3d_scalars] = h_pecnd(ie, igp, jgp, ilev)[ivector];
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

void Elements::BufferViews::init(int num_elems) {
  pressure =
      ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("Pressure buffer", num_elems);
  pressure_grad = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>(
      "Gradient of pressure", num_elems);
  temperature_virt = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Virtual Temperature", num_elems);
  temperature_grad = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>(
      "Gradient of temperature", num_elems);
  omega_p = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Omega_P why two named the same thing???", num_elems);
  vdp = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("vdp???", num_elems);
  div_vdp = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Divergence of dp3d * u", num_elems);
  ephi = ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>(
      "Kinetic Energy + Geopotential Energy", num_elems);
  energy_grad = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>(
      "Gradient of ephi", num_elems);
  vorticity =
      ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>("Vorticity", num_elems);

  qtens = ExecViewManaged<Scalar * [QSIZE_D][NP][NP][NUM_LEV]>(
      "buffer for tracers", num_elems);
  vstar = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("buffer for v/dp",
                                                         num_elems);
  vstar_qdp = ExecViewManaged<Scalar * [QSIZE_D][2][NP][NP][NUM_LEV]>(
      "buffer for vstar*qdp", num_elems);

  preq_buf = ExecViewManaged<Real * [NP][NP]>("Preq Buffer", num_elems);

  div_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("Divergence Buffer",
                                                           num_elems);
  grad_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("Gradient Buffer",
                                                            num_elems);
  vort_buf = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("Vorticity Buffer",
                                                            num_elems);

  kernel_start_times = ExecViewManaged<clock_t *>("Start Times", num_elems);
  kernel_end_times = ExecViewManaged<clock_t *>("End Times", num_elems);
}

} // namespace Homme
