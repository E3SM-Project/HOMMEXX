#ifndef HOMMEXX_SPHERE_OPERATORS_HPP
#define HOMMEXX_SPHERE_OPERATORS_HPP

#include "Types.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"
#include "utilities/SubviewUtils.hpp"

#include <Kokkos_Core.hpp>

namespace Homme {

class SphereOperatorsFunctor
{
public:

  SphereOperatorsFunctor (const Elements& elems, const Derivative& derivative)
  {
    const int num_elems = elements.num_elems();

    // Create the buffers
    vector_buf_sl_1 = ExecViewManaged<Real   * [2][NP][NP]         >("single-level vector buffer 1", num_elems);
    vector_buf_sl_2 = ExecViewManaged<Real   * [2][NP][NP]         >("single-level vector buffer 2", num_elems);
    scalar_buf_1    = ExecViewManaged<Scalar *    [NP][NP][NUM_LEV]>("scalar buffer 1", num_elems);
    scalar_buf_2    = ExecViewManaged<Scalar *    [NP][NP][NUM_LEV]>("scalar buffer 2", num_elems);
    scalar_buf_3    = ExecViewManaged<Scalar *    [NP][NP][NUM_LEV]>("scalar buffer 3", num_elems);
    vector_buf_1    = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("vector buffer 1", num_elems);
    vector_buf_2    = ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>("vector buffer 2", num_elems);

    // Get dvv
    dvv = derivative.get_d

    // Get all needed 2d fields from elements
    m_d        = elements.m_d;
    m_dinv     = elements.m_dinv;
    m_metdet   = elements.m_metdet;
    m_metinv   = elements.m_metinv;
    m_spheremp = elements.m_spheremp;
    m_mp       = elements.m_mp;
  }


// ================ SINGLE-LEVEL IMPLEMENTATION =========================== //

  KOKKOS_INLINE_FUNCTION void
  gradient_sphere_sl(const TeamMember &team,
                     const ExecViewUnmanaged<const Real    [NP][NP]> scalar,
                     const ExecViewUnmanaged<      Real [2][NP][NP]> grad_s) {
    const auto& dinv = Homme::subview(m_dinv,team.league_rank());
    const auto& temp_v_buf = Homme::subview(vector_buf_sl_1,team.league_rank());
    constexpr int contra_iters = NP * NP;
    // TODO: Use scratch space for this
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int j = loop_idx / NP;
      const int l = loop_idx % NP;
      Real dsdx(0), dsdy(0);
      for (int i = 0; i < NP; ++i) {
        dsdx += dvv(l, i) * scalar(j, i);
        dsdy += dvv(l, i) * scalar(i, j);
      }
      temp_v_buf(0, j, l) = dsdx * PhysicalConstants::rrearth;
      temp_v_buf(1, l, j) = dsdy * PhysicalConstants::rrearth;
    });
    team.team_barrier();

    constexpr int grad_iters = 2 * NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, grad_iters),
                         [&](const int loop_idx) {
      const int h = (loop_idx / NP) / NP;
      const int i = (loop_idx / NP) % NP;
      const int j = loop_idx % NP;
      grad_s(h, j, i) = dinv(h, 0, j, i) * temp_v_buf(0, j, i) +
                        dinv(h, 1, j, i) * temp_v_buf(1, j, i);
    });
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION void gradient_sphere_update_sl(
      const TeamMember &team,
      const ExecViewUnmanaged<const Real    [NP][NP]> scalar,
      const ExecViewUnmanaged<      Real [2][NP][NP]> grad_s) {
    constexpr int contra_iters = NP * NP;
    const auto& temp_v_buf = Homme::subview(vector_buf_sl_1,team.league_rank());
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int j = loop_idx / NP;
      const int l = loop_idx % NP;
      Real dsdx(0), dsdy(0);
      for (int i = 0; i < NP; ++i) {
        dsdx += dvv(l, i) * scalar(j, i);
        dsdy += dvv(l, i) * scalar(i, j);
      }
      temp_v_buf(0, j, l) = dsdx * PhysicalConstants::rrearth;
      temp_v_buf(1, l, j) = dsdy * PhysicalConstants::rrearth;
    });
    team.team_barrier();

    constexpr int grad_iters = 2 * NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, grad_iters),
                         [&](const int loop_idx) {
      const int h = (loop_idx / NP) / NP;
      const int i = (loop_idx / NP) % NP;
      const int j = loop_idx % NP;
      grad_s(h, j, i) += dinv(h, 0, j, i) * temp_v_buf(0, j, i) +
                         dinv(h, 1, j, i) * temp_v_buf(1, j, i);
    });
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION void
  divergence_sphere_sl(const TeamMember &team,
                       const ExecViewUnmanaged<const Real    [2][NP][NP]> v,
                       const ExecViewUnmanaged<      Real       [NP][NP]> div_v) {
    const auto& metdet = Homme::subview(m_metdet,team.league_rank());
    const auto& dinv = Homme::subview(m_dinv,team.league_rank());
    const auto& gv_buf = Homme::subview(vector_buf_sl_1,team.league_rank());
    constexpr int contra_iters = NP * NP * 2;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int hgp = (loop_idx / NP) / NP;
      const int igp = (loop_idx / NP) % NP;
      const int jgp = loop_idx % NP;
      gv_buf(hgp, igp, jgp) = (dinv(0, hgp, igp, jgp) * v(0, igp, jgp) +
                               dinv(1, hgp, igp, jgp) * v(1, igp, jgp)) *
                              metdet(igp, jgp);
    });
    team.team_barrier();

    constexpr int div_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, div_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Real dudx = 0.0, dvdy = 0.0;
      for (int kgp = 0; kgp < NP; ++kgp) {
        dudx += dvv(igp, kgp) * gv_buf(0, jgp, kgp);
        dvdy += dvv(jgp, kgp) * gv_buf(1, kgp, igp);
      }
      div_v(jgp, igp) = (dudx + dvdy) * ((1.0 / metdet(jgp, igp)) *
                                         PhysicalConstants::rrearth);
    });
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION void divergence_sphere_wk_sl(
            const TeamMember &team,
            const ExecViewUnmanaged<const Real [2][NP][NP]> v,
            const ExecViewUnmanaged<      Real    [NP][NP]> div_v) {

    const auto& dinv = Homme::subview(m_dinv,team.league_rank());
    const auto& spheremp = Homme::subview(m_spheremp,team.league_rank());
    const auto& gv_buf = Homme::subview(vector_buf_sl_1,team.league_rank());

    // copied from strong divergence as is but without metdet
    // conversion to contravariant
    constexpr int contra_iters = NP * NP * 2;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int hgp = (loop_idx / NP) / NP;
      const int igp = (loop_idx / NP) % NP;
      const int jgp = loop_idx % NP;
      gv_buf(hgp, igp, jgp) = dinv(0, hgp, igp, jgp) * v(0, igp, jgp) +
                              dinv(1, hgp, igp, jgp) * v(1, igp, jgp);
    });
    team.team_barrier();

    // in strong div
    // kgp = i in strong code, jgp=j, igp=l
    // in weak div, n is like j in strong div,
    // n(weak)=j(strong)=jgp
    // m(weak)=l(strong)=igp
    // j(weak)=i(strong)=kgp
    constexpr int div_iters = NP * NP;
    // keeping indices' names as in F
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, div_iters),
                         [&](const int loop_idx) {
      const int mgp = loop_idx / NP;
      const int ngp = loop_idx % NP;
      Real dd = 0.0;
      for (int jgp = 0; jgp < NP; ++jgp) {
        dd -= (spheremp(ngp, jgp) * gv_buf(0, ngp, jgp) * dvv(jgp, mgp) +
               spheremp(jgp, mgp) * gv_buf(1, jgp, mgp) * dvv(jgp, ngp)) *
              PhysicalConstants::rrearth;
      }
      div_v(ngp, mgp) = dd;
    });
    team.team_barrier();

  } // end of divergence_sphere_wk_sl

  // Note that divergence_sphere requires scratch space of 3 x NP x NP Reals
  // This must be called from the device space
  KOKKOS_INLINE_FUNCTION void
  vorticity_sphere_sl(const TeamMember &team,
                      const ExecViewUnmanaged<const Real [NP][NP]> u,
                      const ExecViewUnmanaged<const Real [NP][NP]> v,
                      const ExecViewUnmanaged<      Real [NP][NP]> vort) {
    const auto& d = Homme::subview(m_d,team.league_rank());
    const auto& metdet = Homme::subview(m_metdet,team.league_rank());
    const auto& vcov_buf = Homme::subview(vector_buf_sl_1,team.league_rank());

    constexpr int covar_iters = 2 * NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, covar_iters),
                         [&](const int loop_idx) {
      const int hgp = loop_idx / NP / NP;
      const int igp = (loop_idx / NP) % NP;
      const int jgp = loop_idx % NP;
      vcov_buf(hgp, igp, jgp) = d(hgp, 0, igp, jgp) * u(igp, jgp) +
                                d(hgp, 1, igp, jgp) * v(igp, jgp);
    });
    team.team_barrier();

    constexpr int vort_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, vort_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Real dudy = 0.0;
      Real dvdx = 0.0;
      for (int kgp = 0; kgp < NP; ++kgp) {
        dvdx += dvv(igp, kgp) * vcov_buf(1, jgp, kgp);
        dudy += dvv(jgp, kgp) * vcov_buf(0, kgp, igp);
      }

      vort(jgp, igp) = (dvdx - dudy) * ((1.0 / metdet(jgp, igp)) *
                                        PhysicalConstants::rrearth);
    });
    team.team_barrier();
  }

  // analog of fortran's laplace_wk_sphere
  // Single level implementation
  KOKKOS_INLINE_FUNCTION void laplace_wk_sl(
            const TeamMember &team,
            const ExecViewUnmanaged<const Real [NP][NP]> field,
            const ExecViewUnmanaged<      Real [NP][NP]> laplace) {
    // let's ignore var coef and tensor hv
    gradient_sphere_sl(team, field, grad_s);
    divergence_sphere_wk_sl(team, grad_s, laplace);
  } // end of laplace_wk_sl

  // ================ MULTI-LEVEL IMPLEMENTATION =========================== //

  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  gradient_sphere(const TeamMember &team,
                  const ExecViewUnmanaged<const Scalar    [NP][NP][NUM_LEV]> scalar,
                  const ExecViewUnmanaged<      Scalar [2][NP][NP][NUM_LEV]> grad_s)
  {
    const auto& dinv = Homme::subview(m_dinv, team.league_rank());
    const auto& v_buf = Homme::subview(vector_buf_1,team.league_rank());

    constexpr int contra_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        Scalar dsdx, dsdy;
        for (int kgp = 0; kgp < NP; ++kgp) {
          dsdx += dvv(jgp, kgp) * scalar(igp, kgp, ilev);
          dsdy += dvv(jgp, kgp) * scalar(kgp, igp, ilev);
        }
        v_buf(0, igp, jgp, ilev) = dsdx * PhysicalConstants::rrearth;
        v_buf(1, jgp, igp, ilev) = dsdy * PhysicalConstants::rrearth;
      });
    });
    team.team_barrier();

    // TODO: merge the two parallel for's
    constexpr int grad_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, grad_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        grad_s(0, igp, jgp, ilev) =
            dinv(0, 0, igp, jgp) * v_buf(0, igp, jgp, ilev) +
            dinv(0, 1, igp, jgp) * v_buf(1, igp, jgp, ilev);
        grad_s(1, igp, jgp, ilev) =
            dinv(1, 0, igp, jgp) * v_buf(0, igp, jgp, ilev) +
            dinv(1, 1, igp, jgp) * v_buf(1, igp, jgp, ilev);
      });
    });
    team.team_barrier();
  }

  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void gradient_sphere_update(
            const TeamMember &team,
            const ExecViewUnmanaged<const Scalar    [NP][NP][NUM_LEV]> scalar,
            const ExecViewUnmanaged<      Scalar [2][NP][NP][NUM_LEV]> grad_s)
  {
    const auto& dinv = Homme::subview(m_dinv, team.league_rank());
    const auto& v_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int contra_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        Scalar dsdx, dsdy;
        for (int kgp = 0; kgp < NP; ++kgp) {
          dsdx += dvv(jgp, kgp) * scalar(igp, kgp, ilev);
          dsdy += dvv(jgp, kgp) * scalar(kgp, igp, ilev);
        }
        v_buf(0, igp, jgp, ilev) = dsdx * PhysicalConstants::rrearth;
        v_buf(1, jgp, igp, ilev) = dsdy * PhysicalConstants::rrearth;
      });
    });
    team.team_barrier();

    // TODO: merge the two parallel for's
    constexpr int grad_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, grad_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        grad_s(0, igp, jgp, ilev) +=
            dinv(0, 0, igp, jgp) * v_buf(0, igp, jgp, ilev) +
            dinv(0, 1, igp, jgp) * v_buf(1, igp, jgp, ilev);
        grad_s(1, igp, jgp, ilev) +=
            dinv(1, 0, igp, jgp) * v_buf(0, igp, jgp, ilev) +
            dinv(1, 1, igp, jgp) * v_buf(1, igp, jgp, ilev);
      });
    });
    team.team_barrier();
  }

  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  divergence_sphere(const TeamMember &team,
                    const ExecViewUnmanaged<const Scalar [2][NP][NP][NUM_LEV]> v,
                    const ExecViewUnmanaged<      Scalar    [NP][NP][NUM_LEV]> div_v)
  {
    const auto& dinv = Homme::subview(m_dinv, team.league_rank());
    const auto& metdet = Homme::subview(m_metdet, team.league_rank());
    const auto& gv_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int contra_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        gv_buf(0, igp, jgp, ilev) =
            (dinv(0, 0, igp, jgp) * v(0, igp, jgp, ilev) +
             dinv(1, 0, igp, jgp) * v(1, igp, jgp, ilev)) *
            metdet(igp, jgp);
        gv_buf(1, igp, jgp, ilev) =
            (dinv(0, 1, igp, jgp) * v(0, igp, jgp, ilev) +
             dinv(1, 1, igp, jgp) * v(1, igp, jgp, ilev)) *
            metdet(igp, jgp);
      });
    });
    team.team_barrier();

    // j, l, i -> i, j, k
    constexpr int div_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, div_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        Scalar dudx, dvdy;
        for (int kgp = 0; kgp < NP; ++kgp) {
          dudx += dvv(jgp, kgp) * gv_buf(0, igp, kgp, ilev);
          dvdy += dvv(igp, kgp) * gv_buf(1, kgp, jgp, ilev);
        }
        div_v(igp, jgp, ilev) =
            (dudx + dvdy) * (1.0 / metdet(igp, jgp) * PhysicalConstants::rrearth);
      });
    });
    team.team_barrier();
  }

  // Note: this updates the field div_v as follows:
  //     div_v = beta*div_v + alpha*div(v)
  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  divergence_sphere_update(const TeamMember &team,
                           const Real alpha, const Real beta,
                           const ExecViewUnmanaged<const Scalar [2][NP][NP][NUM_LEV]> v,
                           const ExecViewUnmanaged<      Scalar    [NP][NP][NUM_LEV]> div_v)
  {
    const auto& dinv = Homme::subview(m_dinv, team.league_rank());
    const auto& metdet = Homme::subview(m_metdet, team.league_rank());
    const auto& gv = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int contra_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        gv(0, igp, jgp, ilev) = (dinv(0,0,igp,jgp)*v(0, igp, jgp, ilev) +
                                 dinv(1,0,igp,jgp)*v(1,igp,jgp,ilev)) * metdet(igp,jgp);
        gv(1, igp, jgp, ilev) = (dinv(0,1,igp,jgp)*v(0, igp, jgp, ilev) +
                                 dinv(1,1,igp,jgp)*v(1,igp,jgp,ilev)) * metdet(igp,jgp);
      });
    });
    team.team_barrier();

    constexpr int div_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, div_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        Scalar dudx, dvdy;
        for (int kgp = 0; kgp < NP; ++kgp) {
          dudx += dvv(jgp, kgp) * gv(0, igp, kgp, ilev);
          dvdy += dvv(igp, kgp) * gv(1, kgp, jgp, ilev);
        }

        div_v(igp,jgp,ilev) *= beta;
        div_v(igp,jgp,ilev) += alpha*((dudx + dvdy) * (1.0 / metdet(igp,jgp) * PhysicalConstants::rrearth));
      });
    });
    team.team_barrier();
  }

  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  vorticity_sphere(const TeamMember &team,
                   const ExecViewUnmanaged<const Scalar [NP][NP][NUM_LEV]> u,
                   const ExecViewUnmanaged<const Scalar [NP][NP][NUM_LEV]> v,
                   const ExecViewUnmanaged<      Scalar [NP][NP][NUM_LEV]> vort)
  {
    const auto& d = Homme::subview(m_d, team.league_rank());
    const auto& metdet = Homme::subview(m_metdet, team.league_rank());
    const auto& vcov_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int covar_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, covar_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        vcov_buf(0, jgp, igp, ilev) =
            d(0, 0, jgp, igp) * u(jgp, igp, ilev) +
            d(0, 1, jgp, igp) * v(jgp, igp, ilev);
        vcov_buf(1, jgp, igp, ilev) =
            d(1, 0, jgp, igp) * u(jgp, igp, ilev) +
            d(1, 1, jgp, igp) * v(jgp, igp, ilev);
      });
    });
    team.team_barrier();

    constexpr int vort_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, vort_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        Scalar dudy, dvdx;
        for (int kgp = 0; kgp < NP; ++kgp) {
          dvdx += dvv(jgp, kgp) * vcov_buf(1, igp, kgp, ilev);
          dudy += dvv(igp, kgp) * vcov_buf(0, kgp, jgp, ilev);
        }
        vort(igp, jgp, ilev) = (dvdx - dudy) * (1.0 / metdet(igp, jgp) *
                                                PhysicalConstants::rrearth);
      });
    });
    team.team_barrier();
  }

  //Why does the prev version take u and v separately?
  //rewriting this to take vector as the input.
  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  vorticity_sphere_vector(const TeamMember &team,
                          const ExecViewUnmanaged<const Scalar [2][NP][NP][NUM_LEV]> v,
                          const ExecViewUnmanaged<      Scalar    [NP][NP][NUM_LEV]> vort)
  {
    const auto& d = Homme::subview(m_d, team.league_rank());
    const auto& metdet = Homme::subview(m_metdet, team.league_rank());
    const auto& sphere_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int covar_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, covar_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        const auto& v0 = v(0,igp,jgp,ilev);
        const auto& v1 = v(1,igp,jgp,ilev);
        sphere_buf(0,igp,jgp,ilev) = d(0,0,igp,jgp) * v0 + d(0,1,igp,jgp) * v1;
        sphere_buf(1,igp,jgp,ilev) = d(1,0,igp,jgp) * v0 + d(1,1,igp,jgp) * v1;
      });
    });
    team.team_barrier();

    constexpr int vort_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, vort_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        Scalar dudy, dvdx;
        for (int kgp = 0; kgp < NP; ++kgp) {
          dvdx += dvv(jgp, kgp) * sphere_buf(1, igp, kgp, ilev);
          dudy += dvv(igp, kgp) * sphere_buf(0, kgp, jgp, ilev);
        }
        vort(igp, jgp, ilev) = (dvdx - dudy) * (1.0 / metdet(igp, jgp) *
                                                PhysicalConstants::rrearth);
      });
    });
    team.team_barrier();
  }


  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  divergence_sphere_wk(const TeamMember &team,
                       const ExecViewUnmanaged<const Scalar [2][NP][NP][NUM_LEV]> v,
                       const ExecViewUnmanaged<      Scalar    [NP][NP][NUM_LEV]> div_v)
  {
    const auto& dinv = Homme::subview(m_dinv, team.league_rank());
    const auto& spheremp = Homme::subview(m_spheremp, team.league_rank());
    const auto& sphere_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int contra_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, contra_iters),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        const auto& v0 = v(0,igp,jgp,ilev);
        const auto& v1 = v(1,igp,jgp,ilev);
        sphere_buf(0,igp,jgp,ilev) = dinv(0, 0, igp, jgp) * v0
                                   + dinv(1, 0, igp, jgp) * v1;
        sphere_buf(1,igp,jgp,ilev) = dinv(0, 1, igp, jgp) * v0
                                   + dinv(1, 1, igp, jgp) * v1;
      });
    });
    team.team_barrier();

    constexpr int div_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, div_iters),
                         [&](const int loop_idx) {
      const int mgp = loop_idx / NP;
      const int ngp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        Scalar dd;
        // TODO: move multiplication by rrearth outside the loop
        for (int jgp = 0; jgp < NP; ++jgp) {
          dd -= (spheremp(ngp, jgp) * sphere_buf(0, ngp, jgp, ilev) * dvv(jgp, mgp) +
                 spheremp(jgp, mgp) * sphere_buf(1, jgp, mgp, ilev) * dvv(jgp, ngp)) *
                PhysicalConstants::rrearth;
        }
        div_v(ngp, mgp, ilev) = dd;
      });
    });
    team.team_barrier();

  }//end of divergence_sphere_wk

  //analog of laplace_simple_c_callable
  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  laplace_simple(const TeamMember &team,
                 const ExecViewUnmanaged<const Scalar [NP][NP][NUM_LEV]> field,
                 const ExecViewUnmanaged<      Scalar [NP][NP][NUM_LEV]> laplace)
  {
      // let's ignore var coef and tensor hv
    gradient_sphere<NUM_LEV_REQUEST>(team, field, grad_s);
    divergence_sphere_wk<NUM_LEV_REQUEST>(team, grad_s, laplace);
  }//end of laplace_simple

  //analog of laplace_wk_c_callable
  //but without if-statements for hypervis_power, var_coef, and hypervis_scaling.
  //for 2d fields, there should be either laplace_simple, or laplace_tensor for the whole run.
  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  laplace_tensor(const TeamMember &team,
                 const ExecViewUnmanaged<const Real   [2][2][NP][NP]>          tensorVisc,
                 const ExecViewUnmanaged<const Scalar       [NP][NP][NUM_LEV]> field,         // input
                 const ExecViewUnmanaged<      Scalar       [NP][NP][NUM_LEV]> laplace)
  {
    gradient_sphere<NUM_LEV_REQUEST>(team, dvv, DInv, field, sphere_buf, grad_s);
  //now multiply tensorVisc(:,:,i,j)*grad_s(i,j) (matrix*vector, independent of i,j )
  //but it requires a temp var to store a result. the result is then placed to grad_s,
  //or should it be an extra temp var instead of an extra loop?
    const auto& dinv = Homme::subview(m_dinv, team.league_rank());
    const auto& spheremp = Homme::subview(m_spheremp, team.league_rank());
    const auto& sphere_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int num_iters = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_iters),
                       [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        sphere_buf(0,igp,jgp,ilev) = tensorVisc(0,0,igp,jgp) * grad_s(0,igp,jgp,ilev)
                           + tensorVisc(1,0,igp,jgp) * grad_s(1,igp,jgp,ilev);
        sphere_buf(1,igp,jgp,ilev) = tensorVisc(0,1,igp,jgp) * grad_s(0,igp,jgp,ilev)
                           + tensorVisc(1,1,igp,jgp) * grad_s(1,igp,jgp,ilev);
      });
    });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_iters),
                       [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        grad_s(0,igp,jgp,ilev) = sphere_buf(0,igp,jgp,ilev);
        grad_s(1,igp,jgp,ilev) = sphere_buf(1,igp,jgp,ilev);
      });
    });
    team.team_barrier();

    divergence_sphere_wk<NUM_LEV_REQUEST>(team, dvv, DInv, spheremp, grad_s, sphere_buf, laplace);
  }//end of laplace_tensor

  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  curl_sphere_wk_testcov(const TeamMember &team,
                         const ExecViewUnmanaged<const Scalar    [NP][NP][NUM_LEV]> scalar,
                         const ExecViewUnmanaged<      Scalar [2][NP][NP][NUM_LEV]> curls)
  {
    const auto& d = Homme::subview(m_d, team.league_rank());
    const auto& mp = Homme::subview(m_mp, team.league_rank());
    const auto& sphere_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int np_squared = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared), [&](const int loop_idx) {
      const int ngp = loop_idx / NP;
      const int mgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        sphere_buf(0, ngp, mgp, ilev) = 0;
        sphere_buf(1, ngp, mgp, ilev) = 0;
      });
      for (int jgp = 0; jgp < NP; ++jgp) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
          sphere_buf(0, ngp, mgp, ilev) -= mp(jgp,mgp)*scalar(jgp,mgp,ilev)*dvv(jgp,ngp);
          sphere_buf(1, ngp, mgp, ilev) += mp(ngp,jgp)*scalar(ngp,jgp,ilev)*dvv(jgp,mgp);
        });
      }
    });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP; //slowest
      const int jgp = loop_idx % NP; //fastest
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        curls(0,igp,jgp,ilev) = (D(0,0,igp,jgp)*sphere_buf(0, igp, jgp, ilev)
                               + D(1,0,igp,jgp)*sphere_buf(1, igp, jgp, ilev))
                              * PhysicalConstants::rrearth;
        curls(1,igp,jgp,ilev) = (D(0,1,igp,jgp)*sphere_buf(0, igp, jgp, ilev)
                               + D(1,1,igp,jgp)*sphere_buf(1, igp, jgp, ilev))
                              * PhysicalConstants::rrearth;
      });
    });
    team.team_barrier();
  }

  // This computes curls = alpha*curl(scalar) + beta*curls, where scalar is the input view,
  // and curls is the output view
  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  curl_sphere_wk_testcov_update(const TeamMember &team, const Real alpha, const Real beta,
                                const ExecViewUnmanaged<const Scalar       [NP][NP][NUM_LEV]> scalar,
                                const ExecViewUnmanaged<      Scalar    [2][NP][NP][NUM_LEV]> curls)
  {
    const auto& d = Homme::subview(m_d, team.league_rank());
    const auto& mp = Homme::subview(m_mp, team.league_rank());
    const auto& sphere_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int np_squared = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared), [&](const int loop_idx) {
      const int ngp = loop_idx / NP;
      const int mgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        sphere_buf(0, ngp, mgp, ilev) = 0;
        sphere_buf(1, ngp, mgp, ilev) = 0;
      });
      for (int jgp = 0; jgp < NP; ++jgp) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
          sphere_buf(0, ngp, mgp, ilev) -= mp(jgp,mgp)*scalar(jgp,mgp,ilev)*dvv(jgp,ngp);
          sphere_buf(1, ngp, mgp, ilev) += mp(ngp,jgp)*scalar(ngp,jgp,ilev)*dvv(jgp,mgp);
        });
      }
    });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP; //slowest
      const int jgp = loop_idx % NP; //fastest
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        const auto& sb0 = sphere_buf(0, igp, jgp, ilev);
        const auto& sb1 = sphere_buf(1, igp, jgp, ilev);
        curls(0,igp,jgp,ilev) = beta*curls(0,igp,jgp,ilev) + alpha *
                                (D(0,0,igp,jgp)*sb0
                               + D(1,0,igp,jgp)*sb1)
                                * PhysicalConstants::rrearth;
        curls(1,igp,jgp,ilev) = beta*curls(1,igp,jgp,ilev) + alpha *
                                (D(0,1,igp,jgp)*sb0
                               + D(1,1,igp,jgp)*sb1)
                              * PhysicalConstants::rrearth;
      });
    });
    team.team_barrier();
  }

  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  grad_sphere_wk_testcov(const TeamMember &team,
                         const ExecViewUnmanaged<const Scalar    [NP][NP][NUM_LEV]> scalar,
                         const ExecViewUnmanaged<      Scalar [2][NP][NP][NUM_LEV]> grads)
  {
    const auto& d = Homme::subview(m_d, team.league_rank());
    const auto& mp = Homme::subview(m_mp, team.league_rank());
    const auto& metinv = Homme::subview(m_metinv, team.league_rank());
    const auto& metdet = Homme::subview(m_metdet, team.league_rank());
    const auto& sphere_buf = Homme::subview(vector_buf_1,team.league_rank());
    constexpr int np_squared = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared), [&](const int loop_idx) {
      const int ngp = loop_idx / NP;
      const int mgp = loop_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        sphere_buf(0, ngp, mgp, ilev) = 0;
        sphere_buf(1, ngp, mgp, ilev) = 0;
      });
      for (int jgp = 0; jgp < NP; ++jgp) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
          const auto& mpnj = mp(ngp,jgp);
          const auto& mpjm = mp(jgp,mgp);
          const auto& md = metdet(ngp,mgp);
          const auto& snj = scalar(ngp,jgp,ilev);
          const auto& sjm = scalar(jgp,mgp,ilev);
          const auto& djm = dvv(jgp,mgp);
          const auto& djn = dvv(jgp,ngp);
          sphere_buf(0, ngp, mgp, ilev) -= (
            mpnj*
            metinv(0,0,ngp,mgp)*
            md*
            snj*
            djm
            +
            mpjm*
            metinv(0,1,ngp,mgp)*
            md*
            sjm*
            djn);

          sphere_buf(1, ngp, mgp, ilev) -= (
            mpnj*
            metinv(1,0,ngp,mgp)*
            md*
            snj*
            djm
            +
            mpjm*
            metinv(1,1,ngp,mgp)*
            md*
            sjm*
            djn);
        });
      }
    });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP; //slowest
      const int jgp = loop_idx % NP; //fastest
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        const auto& sb0 = sphere_buf(0, igp, jgp, ilev);
        const auto& sb1 = sphere_buf(1, igp, jgp, ilev);
        grads(0,igp,jgp,ilev) = (D(0,0,igp,jgp)*sb0
                               + D(1,0,igp,jgp)*sb1)
                              * PhysicalConstants::rrearth;
        grads(1,igp,jgp,ilev) = (D(0,1,igp,jgp)*sb0
                               + D(1,1,igp,jgp)*sb1)
                              * PhysicalConstants::rrearth;
      });
    });
    team.team_barrier();
  }

  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  vlaplace_sphere_wk_cartesian(const TeamMember &team,
                               const ExecViewUnmanaged<const Real   [2][2][NP][NP]>          tensorVisc,
                               const ExecViewUnmanaged<const Real   [2][3][NP][NP]>          vec_sph2cart,
                               const ExecViewUnmanaged<const Scalar    [2][NP][NP][NUM_LEV]> vector,
                               const ExecViewUnmanaged<      Scalar    [2][NP][NP][NUM_LEV]> laplace)
  {
    const auto& dinv = Homme::subview(m_dinv, team.league_rank());
    const auto& spheremp = Homme::subview(m_spheremp, team.league_rank());
    const auto& grads = Homme::subview(vector_buf_1,team.league_rank());
    const auto& laplace0 = Homme::subview(scalar_buf_1,team.league_rank());
    const auto& laplace1 = Homme::subview(scalar_buf_2,team.league_rank());
    const auto& laplace2 = Homme::subview(scalar_buf_3,team.league_rank());
    const auto& sphere_buf = Homme::subview(vector_buf_2,team.league_rank());
    constexpr int np_squared = NP * NP;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP; //slowest
      const int jgp = loop_idx % NP; //fastest
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
        laplace0(igp,jgp,ilev) = vec_sph2cart(0,0,igp,jgp)*vector(0,igp,jgp,ilev)
                               + vec_sph2cart(1,0,igp,jgp)*vector(1,igp,jgp,ilev) ;
        laplace1(igp,jgp,ilev) = vec_sph2cart(0,1,igp,jgp)*vector(0,igp,jgp,ilev)
                               + vec_sph2cart(1,1,igp,jgp)*vector(1,igp,jgp,ilev) ;
        laplace2(igp,jgp,ilev) = vec_sph2cart(0,2,igp,jgp)*vector(0,igp,jgp,ilev)
                               + vec_sph2cart(1,2,igp,jgp)*vector(1,igp,jgp,ilev) ;
      });
    });
    team.team_barrier();

    // Use laplace* as input, and then overwrite it with the output (saves temporaries)
    laplace_tensor<NUM_LEV_REQUEST>(team,dvv,Dinv,spheremp,tensorVisc,grads,laplace0,sphere_buf,laplace0);
    laplace_tensor<NUM_LEV_REQUEST>(team,dvv,Dinv,spheremp,tensorVisc,grads,laplace1,sphere_buf,laplace1);
    laplace_tensor<NUM_LEV_REQUEST>(team,dvv,Dinv,spheremp,tensorVisc,grads,laplace2,sphere_buf,laplace2);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP; //slowest
      const int jgp = loop_idx % NP; //fastest
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
  #define UNDAMPRRCART
  #ifdef UNDAMPRRCART
        laplace(0,igp,jgp,ilev) = vec_sph2cart(0,0,igp,jgp)*laplace0(igp,jgp,ilev)
                                + vec_sph2cart(0,1,igp,jgp)*laplace1(igp,jgp,ilev)
                                + vec_sph2cart(0,2,igp,jgp)*laplace2(igp,jgp,ilev)
                                + 2.0*spheremp(igp,jgp)*vector(0,igp,jgp,ilev)
                                        *(PhysicalConstants::rrearth)*(PhysicalConstants::rrearth);

        laplace(1,igp,jgp,ilev) = vec_sph2cart(1,0,igp,jgp)*laplace0(igp,jgp,ilev)
                                + vec_sph2cart(1,1,igp,jgp)*laplace1(igp,jgp,ilev)
                                + vec_sph2cart(1,2,igp,jgp)*laplace2(igp,jgp,ilev)
                                + 2.0*spheremp(igp,jgp)*vector(1,igp,jgp,ilev)
                                        *(PhysicalConstants::rrearth)*(PhysicalConstants::rrearth);
  #else
        laplace(0,igp,jgp,ilev) = vec_sph2cart(0,0,igp,jgp)*laplace0(igp,jgp,ilev)
                                + vec_sph2cart(0,1,igp,jgp)*laplace1(igp,jgp,ilev)
                                + vec_sph2cart(0,2,igp,jgp)*laplace2(igp,jgp,ilev);
        laplace(1,igp,jgp,ilev) = vec_sph2cart(1,0,igp,jgp)*laplace0(igp,jgp,ilev)
                                + vec_sph2cart(1,1,igp,jgp)*laplace1(igp,jgp,ilev)
                                + vec_sph2cart(1,2,igp,jgp)*laplace2(igp,jgp,ilev);
  #endif
      });
    });
    team.team_barrier();
  } // end of vlaplace_sphere_wk_cartesian

  template<int NUM_LEV_REQUEST = NUM_LEV>
  KOKKOS_INLINE_FUNCTION void
  vlaplace_sphere_wk_contra(const TeamMember &team, const Real nu_ratio,
                            const ExecViewUnmanaged<const Scalar    [2][NP][NP][NUM_LEV]> vector,
                            const ExecViewUnmanaged<      Scalar    [2][NP][NP][NUM_LEV]> laplace) {

    // grad(div(v))
    divergence_sphere<NUM_LEV_REQUEST>(team,dvv,dinv,metdet,vector,sphere_buf,div_vort_temp);

    const auto& d = Homme::subview(m_d, team.league_rank());
    const auto& dinv = Homme::subview(m_dinv, team.league_rank());
    const auto& mp = Homme::subview(m_mp, team.league_rank());
    const auto& spheremp = Homme::subview(m_spheremp, team.league_rank());
    const auto& metinv = Homme::subview(m_metinv, team.league_rank());
    const auto& metdet = Homme::subview(m_metdet, team.league_rank());
    const auto& div_vort_temp = Homme::subview(scalar_buf_1,team.league_rank());
    const auto& grad_curl_cov = Homme::subview(vector_buf_1,team.league_rank());
    const auto& sphere_buf = Homme::subview(vector_buf_2,team.league_rank());
    constexpr int np_squared = NP * NP;
    if (nu_ratio>0 && nu_ratio!=1.0) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared),
                          [&](const int loop_idx) {
        const int igp = loop_idx / NP; //slow
        const int jgp = loop_idx % NP; //fast
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
          div_vort_temp(igp,jgp,ilev) *= nu_ratio;
        });
      });
      team.team_barrier();
    }
    grad_sphere_wk_testcov<NUM_LEV_REQUEST>(team,dvv,d,mp,metinv,metdet,div_vort_temp,sphere_buf,grad_curl_cov);

    // curl(curl(v))
    vorticity_sphere_vector<NUM_LEV_REQUEST>(team,dvv,d,metdet,vector,sphere_buf,div_vort_temp);
    curl_sphere_wk_testcov_update<NUM_LEV_REQUEST>(team,-1.0,1.0,dvv,d,mp,div_vort_temp,sphere_buf,grad_curl_cov);

    const auto re2 = PhysicalConstants::rrearth*PhysicalConstants::rrearth;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, np_squared),
                        [&](const int loop_idx) {
      const int igp = loop_idx / NP; //slow
      const int jgp = loop_idx % NP; //fast
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NUM_LEV_REQUEST), [&] (const int& ilev) {
  #define UNDAMPRRCART
  #ifdef UNDAMPRRCART
        const auto f = 2.0*spheremp(igp,jgp);
        laplace(0,igp,jgp,ilev) = f*vector(0,igp,jgp,ilev)*re2;
        laplace(1,igp,jgp,ilev) = f*vector(1,igp,jgp,ilev)*re2;
  #endif
        laplace(0,igp,jgp,ilev) += grad_curl_cov(0,igp,jgp,ilev);
        laplace(1,igp,jgp,ilev) += grad_curl_cov(1,igp,jgp,ilev);
      });
     });
     team.team_barrier();
  }//end of vlaplace_sphere_wk_contra

private:

  // These buffers should be enough to handle any single call to any single sphere operator
  ExecViewManaged<Real   * [2][NP][NP]>            vector_buf_sl_1;
  ExecViewManaged<Real   * [2][NP][NP]>            vector_buf_sl_2;
  ExecViewManaged<Scalar *    [NP][NP][NUM_LEV]>   scalar_buf_1;
  ExecViewManaged<Scalar *    [NP][NP][NUM_LEV]>   scalar_buf_2;
  ExecViewManaged<Scalar *    [NP][NP][NUM_LEV]>   scalar_buf_3;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>   vector_buf_1;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>   vector_buf_2;

  ExecViewUnmanaged<Real [NP][NP]>          dvv;

  ExecViewUnmanaged<Real * [NP][NP]>        m_mp;
  ExecViewUnmanaged<Real * [NP][NP]>        m_spheremp;
  ExecViewUnmanaged<Real * [NP][NP]>        m_rspheremp;
  ExecViewUnmanaged<Real * [2][2][NP][NP]>  m_metinv;
  ExecViewUnmanaged<Real * [NP][NP]>        m_metdet;
  ExecViewUnmanaged<Real * [2][2][NP][NP]>  m_d;
  ExecViewUnmanaged<Real * [2][2][NP][NP]>  m_dinv;

};

} // namespace Homme

#endif // HOMMEXX_SPHERE_OPERATORS_HPP
