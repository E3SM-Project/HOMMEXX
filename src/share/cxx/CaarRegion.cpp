#include "CaarRegion.hpp"
#include "Utility.hpp"

#include <assert.h>

namespace Homme {

void CaarRegion::init(const int num_elems)
{
  m_num_elems = num_elems;

  m_2d_scalars = ExecViewManaged<Real * [NUM_2D_SCALARS][NP][NP]>("2d scalars", m_num_elems);
  m_2d_tensors = ExecViewManaged<Real * [NUM_2D_TENSORS][2][2][NP][NP]> ("2d tensors", m_num_elems);

  m_3d_scalars = ExecViewManaged<Real * [NUM_3D_SCALARS][NUM_LEV][NP][NP]> ("3d scalars", m_num_elems);
  m_4d_scalars = ExecViewManaged<Real * [NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP]> ("4d scalars",m_num_elems);

  m_Qdp          = ExecViewManaged<Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP]> ("qdp", m_num_elems);
  m_eta_dot_dpdn = ExecViewManaged<Real * [NUM_LEV_P][NP][NP]> ("eta_dot_dpdn", m_num_elems);

  m_3d_buffers = ExecViewManaged<Real * [NUM_3D_BUFFERS][NUM_LEV][NP][NP]> ("buffers", m_num_elems);
}

void CaarRegion::init_2d (CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor, CF90Ptr& spheremp, CF90Ptr& metdet, CF90Ptr& phis)
{
  int k_scalars = 0;
  int k_tensors = 0;
  ExecViewManaged<Real * [NUM_2D_SCALARS][NP][NP]>::HostMirror h_2d_scalars = Kokkos::create_mirror_view(m_2d_scalars);
  ExecViewManaged<Real *[NUM_2D_TENSORS][2][2][NP][NP]>::HostMirror h_2d_tensors = Kokkos::create_mirror_view(m_2d_tensors);
  for (int ie=0; ie<m_num_elems; ++ie)
  {
    // 2d scalars
    for (int jgp=0; jgp<NP; ++jgp)
    {
      for (int igp=0; igp<NP; ++igp, ++k_scalars)
      {
        h_2d_scalars(ie, static_cast<int>(IDX_FCOR),     igp, jgp) = fcor[k_scalars];
        h_2d_scalars(ie, static_cast<int>(IDX_SPHEREMP), igp, jgp) = spheremp[k_scalars];
        h_2d_scalars(ie, static_cast<int>(IDX_METDET),   igp, jgp) = metdet[k_scalars];
        h_2d_scalars(ie, static_cast<int>(IDX_PHIS),     igp, jgp) = phis[k_scalars];
      }
    }

    // 2d tensors
    for (int jdim=0; jdim<2; ++jdim)
    {
      for (int idim=0; idim<2; ++idim)
      {
        for (int jgp=0; jgp<NP; ++jgp)
        {
          for (int igp=0; igp<NP; ++igp, ++k_tensors)
          {
            h_2d_tensors(ie,static_cast<int>(IDX_D),   idim,jdim,igp,jgp) = D[k_tensors];
            h_2d_tensors(ie,static_cast<int>(IDX_DINV),idim,jdim,igp,jgp) = Dinv[k_tensors];
          }
        }
      }
    }
  }

  Kokkos::deep_copy(m_2d_scalars, h_2d_scalars);
  Kokkos::deep_copy(m_2d_tensors, h_2d_tensors);
}

void CaarRegion::random_init(const int num_elems, std::mt19937_64 &engine) {
  init(num_elems);
  std::uniform_real_distribution<Real> random_dist(0.0625, 1.0);

  int k_scalars = 0;

  ExecViewManaged<Real *[NUM_2D_SCALARS][NP][NP]>::HostMirror h_2d_scalars = Kokkos::create_mirror_view(m_2d_scalars);
  ExecViewManaged<Real *[NUM_2D_TENSORS][2][2][NP][NP]>::HostMirror h_2d_tensors = Kokkos::create_mirror_view(m_2d_tensors);
  ExecViewManaged<Real *[NUM_3D_SCALARS][NUM_LEV][NP][NP]>::HostMirror h_3d_scalars = Kokkos::create_mirror_view(m_3d_scalars);
  ExecViewManaged<Real *[NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP]>::HostMirror h_4d_scalars = Kokkos::create_mirror_view(m_4d_scalars);
  ExecViewManaged<Real *[NUM_LEV_P][NP][NP]>::HostMirror h_eta_dot_dpdn = Kokkos::create_mirror_view(m_eta_dot_dpdn);
  ExecViewManaged<Real *[NUM_3D_BUFFERS][NUM_LEV][NP][NP]>::HostMirror h_3d_buffers = Kokkos::create_mirror_view(m_3d_buffers);

  for (int ie=0; ie<m_num_elems; ++ie)
  {
    // 2d scalars
    for (int jgp=0; jgp<NP; ++jgp)
    {
      for (int igp=0; igp<NP; ++igp, ++k_scalars)
      {
        for(int idx_2d = 0; idx_2d < NUM_2D_SCALARS; ++idx_2d) {
          h_2d_scalars(ie, idx_2d, igp, jgp) = random_dist(engine);
        }

        for(int ilev = 0; ilev < NUM_LEV; ++ilev) {
          for(int timelevel = 0; timelevel < NUM_TIME_LEVELS; ++timelevel) {
            for(int idx_4d = 0; idx_4d < NUM_4D_SCALARS; ++idx_4d) {
              h_4d_scalars(ie, timelevel, idx_4d, ilev, igp, jgp) = random_dist(engine);
            }
          }
          for(int idx_3d = 0; idx_3d < NUM_3D_SCALARS; ++idx_3d) {
            h_3d_scalars(ie, idx_3d, ilev, igp, jgp) = random_dist(engine);
          }
          for(int idx_3d = 0; idx_3d < NUM_3D_BUFFERS; ++idx_3d) {
            h_3d_buffers(ie, idx_3d, ilev, igp, jgp) = random_dist(engine);
          }
        }
        for(int ilev = 0; ilev < NUM_LEV_P; ++ilev) {
          h_eta_dot_dpdn(ie, ilev, igp, jgp) = random_dist(engine);
        }
      }
    }

    // 2d tensors
    for (int jdim=0; jdim<2; ++jdim)
    {
      for (int idim=0; idim<2; ++idim)
      {
        for (int jgp=0; jgp<NP; ++jgp)
        {
          for (int igp=0; igp<NP; ++igp)
          {
            h_2d_tensors(ie,static_cast<int>(IDX_D),   idim,jdim,igp,jgp) = random_dist(engine);
            h_2d_tensors(ie,static_cast<int>(IDX_DINV),idim,jdim,igp,jgp) = random_dist(engine);
          }
        }
      }
    }
  }

  Kokkos::deep_copy(m_2d_scalars, h_2d_scalars);
  Kokkos::deep_copy(m_2d_tensors, h_2d_tensors);
  Kokkos::deep_copy(m_3d_scalars, h_3d_scalars);
  Kokkos::deep_copy(m_4d_scalars, h_4d_scalars);
  Kokkos::deep_copy(m_eta_dot_dpdn, h_eta_dot_dpdn);
  Kokkos::deep_copy(m_3d_buffers, h_3d_buffers);
  return;
}


void CaarRegion::pull_from_f90_pointers(CF90Ptr& state_v, CF90Ptr& state_t, CF90Ptr& state_dp3d,
                                    CF90Ptr& derived_phi, CF90Ptr& derived_pecnd,
                                    CF90Ptr& derived_omega_p, CF90Ptr& derived_v,
                                    CF90Ptr& derived_eta_dot_dpdn, CF90Ptr& state_Qdp)
{

  ExecViewManaged<Real *[NUM_3D_SCALARS][NUM_LEV][NP][NP]>::HostMirror h_3d_scalars = Kokkos::create_mirror_view(m_3d_scalars);
  ExecViewManaged<Real *[NUM_LEV_P][NP][NP]>::HostMirror h_eta_dot_dpdn = Kokkos::create_mirror_view(m_eta_dot_dpdn);
  for (int ie=0, k_3d_scalars=0, k_3d_vectors=0, k_eta_dot_dp_dn=0; ie<m_num_elems; ++ie)
  {
    for (int ilev=0; ilev<NUM_LEV; ++ilev)
    {
      for (int jgp=0; jgp<NP; ++jgp)
      {
        for (int igp=0; igp<NP;
             ++igp, ++k_3d_scalars, ++k_eta_dot_dp_dn)
        {
          h_3d_scalars(ie, static_cast<int>(IDX_OMEGA_P), ilev, igp, jgp) = derived_omega_p[k_3d_scalars];
          h_3d_scalars(ie, static_cast<int>(IDX_PECND),   ilev, igp, jgp) = derived_pecnd[k_3d_scalars];
          h_3d_scalars(ie, static_cast<int>(IDX_PHI),     ilev, igp, jgp) = derived_phi[k_3d_scalars];
          h_eta_dot_dpdn (ie, ilev, igp, jgp) = derived_eta_dot_dpdn[k_eta_dot_dp_dn];
        }
      }

      for (int idim : {static_cast<int>(IDX_DERIVED_UN0), static_cast<int>(IDX_DERIVED_VN0)} )
      {
        for (int jgp=0; jgp<NP; ++jgp)
        {
          for (int igp=0; igp<NP; ++igp, ++k_3d_vectors)
          {
            h_3d_scalars(ie, idim, ilev, igp, jgp) = derived_v[k_3d_vectors];
          }
        }
      }
    }
    // Extra level of eta_dot_dpdn
    for (int jgp=0; jgp<NP; ++jgp)
    {
      for (int igp=0; igp<NP; ++igp, ++k_eta_dot_dp_dn)
      {
        h_eta_dot_dpdn (ie, NUM_LEV, igp, jgp) = derived_eta_dot_dpdn[k_eta_dot_dp_dn];
      }
    }
  }
  // what():  deep_copy given views that would require a temporary allocation
  Kokkos::deep_copy (m_3d_scalars, h_3d_scalars);
  Kokkos::deep_copy (m_eta_dot_dpdn, h_eta_dot_dpdn);

  // 4d scalars
  int k_4d_scalars = 0;
  int k_4d_vectors = 0;
  ExecViewManaged<Real * [NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP]>::HostMirror h_4d_scalars = Kokkos::create_mirror_view(m_4d_scalars);
  for (int ie=0; ie<m_num_elems; ++ie)
  {
    for (int tl=0; tl<NUM_TIME_LEVELS; ++tl)
    {
      for (int ilev=0; ilev<NUM_LEV; ++ilev)
      {
        for (int jgp=0; jgp<NP; ++jgp)
        {
          for (int igp=0; igp<NP; ++igp, ++k_4d_scalars)
          {
            h_4d_scalars(ie, tl, static_cast<int>(IDX_DP3D), ilev, igp, jgp) = state_dp3d[k_4d_scalars];
            h_4d_scalars(ie, tl, static_cast<int>(IDX_T),    ilev, igp, jgp) = state_t[k_4d_scalars];
          }
        }

        for (int idim : {static_cast<int>(IDX_U), static_cast<int>(IDX_V)} )
        {
          for (int jgp=0; jgp<NP; ++jgp)
          {
            for (int igp=0; igp<NP; ++igp, ++k_4d_vectors)
            {
              h_4d_scalars(ie, tl, idim, ilev, igp, jgp) = state_v[k_4d_vectors];
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy (m_4d_scalars, h_4d_scalars);

  // Qdp
  int k_qdp = 0;
  ExecViewManaged<Real *[Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP]>::HostMirror h_Qdp = Kokkos::create_mirror_view(m_Qdp);
  for (int ie=0; ie<m_num_elems; ++ie)
  {
    for (int qni=0; qni<Q_NUM_TIME_LEVELS; ++qni)
    {
      for (int iq=0; iq<QSIZE_D; ++iq)
      {
        for (int ilev=0; ilev<NUM_LEV; ++ilev)
        {
          for (int jgp=0; jgp<NP; ++jgp)
          {
            for (int igp=0; igp<NP; ++igp, ++k_qdp)
            {
              h_Qdp(ie,qni,iq,ilev,igp,jgp) = state_Qdp[k_qdp];
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy (m_Qdp, h_Qdp);
}

void CaarRegion::push_to_f90_pointers(F90Ptr& state_v, F90Ptr& state_t, F90Ptr& state_dp3d,
                                      F90Ptr& derived_phi, F90Ptr& derived_pecnd,
                                      F90Ptr& derived_omega_p, F90Ptr& derived_v,
                                      F90Ptr& derived_eta_dot_dpdn, F90Ptr& state_Qdp) const
{
  // 3d scalars
  int k_3d_scalars    = 0;
  int k_3d_vectors    = 0;
  int k_eta_dot_dp_dn = 0;
  ExecViewManaged<Real *[NUM_3D_SCALARS][NUM_LEV][NP][NP]>::HostMirror h_3d_scalars = Kokkos::create_mirror_view(m_3d_scalars);
  ExecViewManaged<Real *[NUM_LEV_P][NP][NP]>::HostMirror h_eta_dot_dpdn = Kokkos::create_mirror_view(m_eta_dot_dpdn);
  Kokkos::deep_copy (h_3d_scalars, m_3d_scalars);
  Kokkos::deep_copy (h_eta_dot_dpdn, m_eta_dot_dpdn);
  for (int ie=0; ie<m_num_elems; ++ie)
  {
    for (int ilev=0; ilev<NUM_LEV; ++ilev)
    {
      for (int jgp=0; jgp<NP; ++jgp)
      {
        for (int igp=0; igp<NP; ++igp, ++k_3d_scalars)
        {
          derived_omega_p[k_3d_scalars] = h_3d_scalars(ie, static_cast<int>(IDX_OMEGA_P), ilev, igp, jgp);
          derived_pecnd[k_3d_scalars]   = h_3d_scalars(ie, static_cast<int>(IDX_PECND),   ilev, igp, jgp);
          derived_phi[k_3d_scalars]     = h_3d_scalars(ie, static_cast<int>(IDX_PHI),     ilev, igp, jgp);
          derived_eta_dot_dpdn[k_eta_dot_dp_dn] = h_eta_dot_dpdn (ie, ilev, igp, jgp);
        }
      }

      for (int idim : {static_cast<int>(IDX_DERIVED_UN0), static_cast<int>(IDX_DERIVED_VN0)} )
      {
        for (int jgp=0; jgp<NP; ++jgp)
        {
          for (int igp=0; igp<NP; ++igp, ++k_3d_vectors)
          {
            derived_v[k_3d_vectors] = h_3d_scalars(ie, idim, ilev, igp, jgp);
          }
        }
      }
    }
    // Extra level of eta_dot_dpdn
    for (int jgp=0; jgp<NP; ++jgp)
    {
      for (int igp=0; igp<NP; ++igp, ++k_eta_dot_dp_dn)
      {
        derived_eta_dot_dpdn[k_eta_dot_dp_dn] = h_eta_dot_dpdn (ie, NUM_LEV, igp, jgp);
      }
    }
  }

  // 4d scalars
  int k_4d_scalars = 0;
  int k_4d_vectors = 0;
  ExecViewManaged<Real *[NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP]>::HostMirror h_4d_scalars = Kokkos::create_mirror_view(m_4d_scalars);
  Kokkos::deep_copy (h_4d_scalars, m_4d_scalars);
  for (int ie=0; ie<m_num_elems; ++ie)
  {
    for (int tl=0; tl<NUM_TIME_LEVELS; ++tl)
    {
      for (int ilev=0; ilev<NUM_LEV; ++ilev)
      {
        for (int jgp=0; jgp<NP; ++jgp)
        {
          for (int igp=0; igp<NP; ++igp, ++k_4d_scalars)
          {
            state_dp3d[k_4d_scalars] = h_4d_scalars(ie, tl, static_cast<int>(IDX_DP3D), ilev, igp, jgp);
            state_t[k_4d_scalars]    = h_4d_scalars(ie, tl, static_cast<int>(IDX_T),    ilev, igp, jgp);
          }
        }

        for (int idim : {static_cast<int>(IDX_U), static_cast<int>(IDX_V)} )
        {
          for (int jgp=0; jgp<NP; ++jgp)
          {
            for (int igp=0; igp<NP; ++igp, ++k_4d_vectors)
            {
              state_v[k_4d_vectors] = h_4d_scalars(ie, tl, idim, ilev, igp, jgp);
            }
          }
        }
      }
    }
  }

  // Qdp
  int k_qdp = 0;
  ExecViewManaged<Real *[Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP]>::HostMirror h_Qdp = Kokkos::create_mirror_view(m_Qdp);
  Kokkos::deep_copy (h_Qdp, m_Qdp);
  for (int ie=0; ie<m_num_elems; ++ie)
  {
    for (int qni=0; qni<Q_NUM_TIME_LEVELS; ++qni)
    {
      for (int iq=0; iq<QSIZE_D; ++iq)
      {
        for (int ilev=0; ilev<NUM_LEV; ++ilev)
        {
          for (int jgp=0; jgp<NP; ++jgp)
          {
            for (int igp=0; igp<NP; ++igp, ++k_qdp)
            {
              state_Qdp[k_qdp] = h_Qdp(ie,qni,iq,ilev,igp,jgp);
            }
          }
        }
      }
    }
  }
}

void CaarRegion::d(Real *d_ptr, int ie) {
  ExecViewManaged<Real[2][2][NP][NP]> d_device = Kokkos::subview(m_2d_tensors, ie, static_cast<int>(IDX_D), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  ExecViewManaged<Real[2][2][NP][NP]>::HostMirror d_host = Kokkos::create_mirror_view(d_device),
    d_wrapper(d_ptr);
  Kokkos::deep_copy(d_host, d_device);
  for(int m = 0; m < 2; ++m) {
    for(int n = 0; n < 2; ++n) {
      for(int igp = 0; igp < NP; ++igp) {
        for(int jgp = 0; jgp < NP; ++jgp) {
          d_wrapper(m, n, jgp, igp) = d_host(n, m, igp, jgp);
        }
      }
    }
  }
}

void CaarRegion::dinv(Real *dinv_ptr, int ie) {
  ExecViewManaged<Real[2][2][NP][NP]> dinv_device = Kokkos::subview(m_2d_tensors, ie, static_cast<int>(IDX_DINV), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  ExecViewManaged<Real[2][2][NP][NP]>::HostMirror dinv_host = Kokkos::create_mirror_view(dinv_device),
    dinv_wrapper(dinv_ptr);
  Kokkos::deep_copy(dinv_host, dinv_device);
  for(int m = 0; m < 2; ++m) {
    for(int n = 0; n < 2; ++n) {
      for(int igp = 0; igp < NP; ++igp) {
        for(int jgp = 0; jgp < NP; ++jgp) {
          dinv_wrapper(m, n, jgp, igp) = dinv_host(n, m, igp, jgp);
        }
      }
    }
  }
}

CaarRegion& get_region()
{
  static CaarRegion r;
  return r;
}

} // namespace Homme
