#ifndef HOMMEXX_CAAR_SCRATCH_HPP
#define HOMMEXX_CAAR_SCRATCH_HPP

#include "Types.hpp"
#include "Dimensions.hpp"

namespace Homme
{

class CaarScratch {
public:

  // This constructor should only be used by the host
  // @param num_elems: the number of elements on this rank
  CaarScratch (const int num_elems)
   : m_pressure ("pressure",num_elems)
   , m_omega_p  ("omega_p",num_elems)
   , m_T_v      ("virtual temperature",num_elems)
   , m_div_vdp  ("div vdp",num_elems)
  {}

  /*
   *KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> const
   *pressure(const TeamMember &team_member) const {
   *  return Kokkos::subview(m_pressure, team_member.league_rank(), ALL, ALL, ALL);
   *}
   */

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> const
  pressure(const int ie) const {
    return Kokkos::subview(m_pressure, ie, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NP][NP]> const
  pressure(const int ie, const int ilev) const {
    return Kokkos::subview(m_pressure, ie, ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION Real &pressure(const int ie, const int ilev,
                                        const int igp, const int jgp) const {
    return m_pressure(ie, ilev, igp, jgp);
  }

  /*
   *KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> const
   *omega_p(const TeamMember& team_member) const {
   *  return Kokkos::subview(m_omega_p, team_member.league_rank(), ALL, ALL, ALL);
   *}
   */

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> const
  omega_p(const int ie) const {
    return Kokkos::subview(m_omega_p, ie, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION Real &omega_p(const int ie, const int ilev,
                                       const int igp, const int jgp) const {
    return m_omega_p(ie, ilev, igp, jgp);
  }

  /*
   *KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> const
   *T_v(const TeamMember& team_member) const {
   *  return Kokkos::subview(m_T_v, team_member.league_rank(), ALL, ALL, ALL);
   *}
   */

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> const
  T_v(const int ie) const {
    return Kokkos::subview(m_T_v, ie, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION Real &T_v(const int ie, const int ilev,
                                   const int igp, const int jgp) const {
    return m_T_v(ie, ilev, igp, jgp);
  }

  /*
   *KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> const
   *div_vdp(const TeamMember& team_member) const {
   *  return Kokkos::subview(m_div_vdp, team_member.league_rank(), ALL, ALL, ALL);
   *}
   */

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> const
  div_vdp(const int ie) const {
    return Kokkos::subview(m_div_vdp, ie, ALL, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION ExecViewUnmanaged<Real[NP][NP]> const
  div_vdp(const int ie, const int ilev) const {
    return Kokkos::subview(m_div_vdp, ie, ilev, ALL, ALL);
  }

  KOKKOS_INLINE_FUNCTION Real &div_vdp(int ie, int ilev, int igp,
                                       int jgp) const {
    return m_div_vdp(ie, ilev, igp, jgp);
  }

private:

  static constexpr Kokkos::Impl::ALL_t ALL = Kokkos::Impl::ALL_t();

  /* Device objects, to reduce the memory transfer required */
  ExecViewManaged<Real * [NUM_LEV][NP][NP]> m_pressure;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]> m_omega_p;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]> m_T_v;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]> m_div_vdp;
};

} // namespace Homme

#endif // HOMMEXX_CAAR_SCRATCH_HPP
