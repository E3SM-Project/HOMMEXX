#ifndef HOMME_REGION_HPP
#define HOMME_REGION_HPP

#include "Dimensions.hpp"

#include "Types.hpp"

#include <Kokkos_Core.hpp>
#include "CaarControl.hpp"

namespace Homme {

/* Per element data - specific velocity, temperature, pressure, etc. */
class CaarRegion {
private:

  enum : int {
    // The number of fields for each dimension
    NUM_4D_SCALARS = 4,
    NUM_3D_SCALARS = 5,
    NUM_2D_SCALARS = 4,
    NUM_2D_TENSORS = 2,

    NUM_3D_BUFFERS = 4,

    // Some constexpr for the index of different variables in the views
    // 4D Scalars
    IDX_U = 0,
    IDX_V = 1,
    IDX_T = 2,
    IDX_DP3D = 3,

    // 3D Scalars
    IDX_OMEGA_P = 0,
    IDX_PECND = 1,
    IDX_PHI = 2,
    IDX_DERIVED_UN0 = 3,
    IDX_DERIVED_VN0 = 4,

    // 2D Scalars
    IDX_FCOR = 0,
    IDX_SPHEREMP = 1,
    IDX_METDET = 2,
    IDX_PHIS = 3,

    // 2D Tensors
    IDX_D = 0,
    IDX_DINV = 1,
  };

  int m_num_elems;

  /* Contains FCOR, SPHEREMP, METDET, PHIS */
  ExecViewManaged<Real * [NUM_2D_SCALARS][NP][NP]> m_2d_scalars;
  /* Contains D, DINV */
  ExecViewManaged<Real * [NUM_2D_TENSORS][2][2][NP][NP]> m_2d_tensors;
  /* Contains OMEGA_P, PECND, PHI, DERIVED_UN0, DERIVED_VN0, QDP, ETA_DPDN */
  ExecViewManaged<Real * [NUM_3D_SCALARS][NUM_LEV][NP][NP]> m_3d_scalars;
  /* Contains U, V, T, DP3D */
  ExecViewManaged<Real * [NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP]> m_4d_scalars;

  ExecViewManaged<Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP]> m_Qdp;
  ExecViewManaged<Real * [NUM_LEV_P][NP][NP]> m_eta_dot_dpdn;

  static constexpr Kokkos::Impl::ALL_t ALL = Kokkos::Impl::ALL_t();

  ExecViewManaged<Real * [NUM_3D_BUFFERS][NUM_LEV][NP][NP]> m_3d_buffers;

public:

  CaarRegion () = default;

  void init(const int num_elems);

  // Fill the exec space views with data coming from F90 pointers
  void init_2d (CF90Ptr& D, CF90Ptr& Dinv, CF90Ptr& fcor,
                CF90Ptr& spheremp, CF90Ptr& metdet, CF90Ptr& phis);

  void random_init(int num_elems, std::mt19937_64 &engine);

  // Fill the exec space views with data coming from F90 pointers
  void pull_from_f90_pointers(CF90Ptr& state_v, CF90Ptr& state_t, CF90Ptr& state_dp3d,
                              CF90Ptr& derived_phi, CF90Ptr& derived_pecnd,
                              CF90Ptr& derived_omega_p, CF90Ptr& derived_v,
                              CF90Ptr& derived_eta_dot_dpdn, CF90Ptr& state_Qdp);

  // Push the results from the exec space views to the F90 pointers
  void push_to_f90_pointers(F90Ptr& state_v, F90Ptr& state_t, F90Ptr& state_dp,
                            F90Ptr& derived_phi, F90Ptr& derived_pecnd,
                            F90Ptr& derived_omega_p, F90Ptr& derived_v,
                            F90Ptr& derived_eta_dot_dpdn, F90Ptr& state_Qdp) const;

  void d(Real *d_ptr, int ie);
  void dinv(Real *dinv_ptr, int ie);

  // v is the tracer we're working with, 0 <= v < QSIZE_D
  // qn0 is the timelevel, 0 <= qn0 < Q_NUM_TIME_LEVELS
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> QDP(const int ie, const int qn0, const int v) const {
    return Kokkos::subview(m_Qdp, ie, qn0, v, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& QDP(const int ie, const int qn0, const int v,
            const int ilev, const int igp, const int jgp) const {
    return m_Qdp(ie, qn0, v, ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV_P][NP][NP]> ETA_DPDN(const int ie) const {
    return Kokkos::subview(m_eta_dot_dpdn, ie, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& ETA_DPDN(const int ie, const int ilev, const int igp, const int jgp) const {
    return m_eta_dot_dpdn(ie, ilev, igp, jgp);
  }

  /* 4D Scalars: pass element/time-level indices to get a subview, or all indices
     to get the value at a given gauss point */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> U(const int ie, const int tl) const {
    return Kokkos::subview(m_4d_scalars, ie, tl, static_cast<int>(IDX_U), ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> U(const int ie, const int tl, const int ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, tl, static_cast<int>(IDX_U), ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& U(const int ie, const int tl, const int ilev, const int igp, const int jgp) const {
    return m_4d_scalars(ie, tl, static_cast<int>(IDX_U), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> V(const int ie, const int tl) const {
    return Kokkos::subview(m_4d_scalars, ie, tl, static_cast<int>(IDX_V), ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> V(const int ie, const int tl, const int ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, tl, static_cast<int>(IDX_V), ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& V(const int ie, const int tl, const int ilev, const int igp, const int jgp) const {
    return m_4d_scalars(ie, tl, static_cast<int>(IDX_V), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T(const int ie, const int tl) const {
    return Kokkos::subview(m_4d_scalars, ie, tl, static_cast<int>(IDX_T), ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& T(const int ie, const int tl, const int ilev, const int igp, const int jgp) const {
    return m_4d_scalars(ie, tl, static_cast<int>(IDX_T), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> DP3D(const int ie, const int tl) const {
    return Kokkos::subview(m_4d_scalars, ie, tl, static_cast<int>(IDX_DP3D), ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& DP3D(const int ie, const int tl, const int ilev, const int igp, const int jgp) const {
    return m_4d_scalars(ie, tl, static_cast<int>(IDX_DP3D), ilev, igp, jgp);
  }

  /* 3D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> OMEGA_P(const int ie, const int ilev) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_OMEGA_P), ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& OMEGA_P(const int ie, const int ilev, const int igp, const int jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_OMEGA_P), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> PECND(const int ie, const int ilev) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_PECND), ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &PECND(const int ie, const int ilev, const int igp, const int jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_PECND), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> PHI(const int ie) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_PHI), ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &PHI(const int ie, const int ilev, const int igp, const int jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_PHI), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> DERIVED_UN0(const int ie, const int ilev) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_DERIVED_UN0), ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& DERIVED_UN0(const int ie, const int ilev, const int igp, const int jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_DERIVED_UN0), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> DERIVED_VN0(const int ie, const int ilev) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_DERIVED_VN0), ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real& DERIVED_VN0(const int ie, const int ilev, const int igp, const int jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_DERIVED_VN0), ilev, igp, jgp);
  }

  /* 2D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> FCOR(const int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, static_cast<int>(IDX_FCOR), ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real FCOR(const int ie, const int igp, const int jgp) const {
    return m_2d_scalars(ie, static_cast<int>(IDX_FCOR), igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> SPHEREMP(const int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, static_cast<int>(IDX_SPHEREMP), ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real SPHEREMP(const int ie, const int igp, const int jgp) const {
    return m_2d_scalars(ie, static_cast<int>(IDX_SPHEREMP), igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> METDET(const int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, static_cast<int>(IDX_METDET), ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> PHIS(const int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, static_cast<int>(IDX_PHIS), ALL, ALL);
  }

  /* 2D Tensors */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[2][2][NP][NP]> D(const int ie) const {
    return Kokkos::subview(m_2d_tensors, ie, static_cast<int>(IDX_D), ALL, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[2][2][NP][NP]> DINV(const int ie) const {
    return Kokkos::subview(m_2d_tensors, ie, static_cast<int>(IDX_DINV), ALL, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> get_3d_buffer (const int ie, const int ibuff) const {
    return Kokkos::subview(m_3d_buffers, ie, ibuff, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> get_3d_buffer (const int ie, const int ibuff, const int ilev) const {
    return Kokkos::subview(m_3d_buffers, ie, ibuff, ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real & get_3d_buffer (const int ie, const int ibuff, const int ilev, const int igp, const int jgp) const {
    return m_3d_buffers (ie, ibuff, ilev, igp, jgp);
  }
};

// TODO: DON'T USE SINGLETONS
CaarRegion& get_region();

} // Homme

#endif // HOMME_REGION_HPP
