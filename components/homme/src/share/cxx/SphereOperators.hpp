#ifndef HOMMEXX_SPHERE_OPERATORS_HPP
#define HOMMEXX_SPHERE_OPERATORS_HPP

#include "Types.hpp"
#include "Elements.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"

#include <Kokkos_Core.hpp>

namespace Homme {

// ================ SINGLE-LEVEL IMPLEMENTATION =========================== //

KOKKOS_INLINE_FUNCTION void
gradient_sphere_sl(const KernelVariables &kv,
                   const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                   const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                   const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                   ExecViewUnmanaged<Real[2][NP][NP]> temp_v_buf,
                   ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  constexpr int contra_iters = NP * NP;
  // TODO: Use scratch space for this
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
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

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h, j, i) = dinv(kv.ie, h, 0, j, i) * temp_v_buf(0, j, i) +
                      dinv(kv.ie, h, 1, j, i) * temp_v_buf(1, j, i);
  });
}

KOKKOS_INLINE_FUNCTION void gradient_sphere_update_sl(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real[NP][NP]> scalar,
    ExecViewUnmanaged<Real[2][NP][NP]> temp_v_buf,
    ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  constexpr int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
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

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h, j, i) += dinv(kv.ie, h, 0, j, i) * temp_v_buf(0, j, i) +
                       dinv(kv.ie, h, 1, j, i) * temp_v_buf(1, j, i);
  });
}

KOKKOS_INLINE_FUNCTION void
divergence_sphere_sl(const KernelVariables &kv,
                     const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                     const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                     const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                     const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                     ExecViewUnmanaged<Real[2][NP][NP]> gv_buf,
                     ExecViewUnmanaged<Real[NP][NP]> div_v) {
  constexpr int contra_iters = NP * NP * 2;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int hgp = (loop_idx / NP) / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    gv_buf(hgp, igp, jgp) = (dinv(kv.ie, 0, hgp, igp, jgp) * v(0, igp, jgp) +
                             dinv(kv.ie, 1, hgp, igp, jgp) * v(1, igp, jgp)) *
                            metdet(kv.ie, igp, jgp);
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudx = 0.0, dvdy = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dudx += dvv(igp, kgp) * gv_buf(0, jgp, kgp);
      dvdy += dvv(jgp, kgp) * gv_buf(1, kgp, igp);
    }
    div_v(jgp, igp) = (dudx + dvdy) * ((1.0 / metdet(kv.ie, jgp, igp)) *
                                       PhysicalConstants::rrearth);
  });
}

KOKKOS_INLINE_FUNCTION void divergence_sphere_wk_sl(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real[2][NP][NP]> v,
    ExecViewUnmanaged<Real[2][NP][NP]> gv_buf,
    ExecViewUnmanaged<Real[NP][NP]> div_v) {

  // copied from strong divergence as is but without metdet
  // conversion to contravariant
  constexpr int contra_iters = NP * NP * 2;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int hgp = (loop_idx / NP) / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    gv_buf(hgp, igp, jgp) = dinv(kv.ie, 0, hgp, igp, jgp) * v(0, igp, jgp) +
                            dinv(kv.ie, 1, hgp, igp, jgp) * v(1, igp, jgp);
  });

  // in strong div
  // kgp = i in strong code, jgp=j, igp=l
  // in weak div, n is like j in strong div,
  // n(weak)=j(strong)=jgp
  // m(weak)=l(strong)=igp
  // j(weak)=i(strong)=kgp
  constexpr int div_iters = NP * NP;
  // keeping indices' names as in F
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int mgp = loop_idx / NP;
    const int ngp = loop_idx % NP;
    Real dd = 0.0;
    for (int jgp = 0; jgp < NP; ++jgp) {
      dd -= (spheremp(kv.ie, ngp, jgp) * gv_buf(0, ngp, jgp) * dvv(jgp, mgp) +
             spheremp(kv.ie, jgp, mgp) * gv_buf(1, jgp, mgp) * dvv(jgp, ngp)) *
            PhysicalConstants::rrearth;
    }
    div_v(ngp, mgp) = dd;
  });

} // end of divergence_sphere_wk_sl

// Note that divergence_sphere requires scratch space of 3 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
vorticity_sphere_sl(const KernelVariables &kv,
                    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                    const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                    const ExecViewUnmanaged<const Real[NP][NP]> u,
                    const ExecViewUnmanaged<const Real[NP][NP]> v,
                    ExecViewUnmanaged<Real[2][NP][NP]> vcov_buf,
                    ExecViewUnmanaged<Real[NP][NP]> vort) {
  constexpr int covar_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, covar_iters),
                       [&](const int loop_idx) {
    const int hgp = loop_idx / NP / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    vcov_buf(hgp, igp, jgp) = d(kv.ie, hgp, 0, igp, jgp) * u(igp, jgp) +
                              d(kv.ie, hgp, 1, igp, jgp) * v(igp, jgp);
  });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, vort_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudy = 0.0;
    Real dvdx = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dvdx += dvv(igp, kgp) * vcov_buf(1, jgp, kgp);
      dudy += dvv(jgp, kgp) * vcov_buf(0, kgp, igp);
    }

    vort(jgp, igp) = (dvdx - dudy) * ((1.0 / metdet(kv.ie, jgp, igp)) *
                                      PhysicalConstants::rrearth);
  });
}

// analog of fortran's laplace_wk_sphere
// Single level implementation
KOKKOS_INLINE_FUNCTION void laplace_wk_sl(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> DInv, // for grad, div
    const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,   // for div
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,           // for grad, div
    // how to get rid of this temp var? passing real* instead of kokkos view
    ////does not work. is creating kokkos temorary in a kernel the correct way?
    ExecViewUnmanaged<Real[2][NP][NP]> grad_s,         // temp to store grad
    const ExecViewUnmanaged<const Real[NP][NP]> field, // input
    // output
    ExecViewUnmanaged<Real[NP][NP]> laplace) {
  // Real grad_s[2][NP][NP];
  // let's ignore var coef and tensor hv
  // gradient_sphere_sl(kv, DInv, dvv, field, grad_s);
  // divergence_sphere_wk_sl(kv, DInv, spheremp, dvv, grad_s, laplace);
} // end of laplace_wk_sl

// ================ MULTI-LEVEL IMPLEMENTATION =========================== //

KOKKOS_INLINE_FUNCTION void
gradient_sphere(const KernelVariables &kv,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
                ExecViewUnmanaged<Scalar * [NUM_LEV][2][NP][NP]> v_buf,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s) {
  const int buf_lev = (kv.ilev < 0 || kv.ilev >= NUM_LEV) ? 0 : kv.ilev;
  constexpr int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dsdx(0.0), dsdy(0.0);
    for (int kgp = 0; kgp < NP; ++kgp) {
      dsdx += dvv(jgp, kgp) * scalar(igp, kgp, kv.ilev);
      dsdy += dvv(jgp, kgp) * scalar(kgp, igp, kv.ilev);
    }
    v_buf(kv.ie, buf_lev, 0, igp, jgp) = dsdx * PhysicalConstants::rrearth;
    v_buf(kv.ie, buf_lev, 1, jgp, igp) = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    grad_s(0, igp, jgp, kv.ilev) =
        dinv(kv.ie, 0, 0, igp, jgp) * v_buf(kv.ie, buf_lev, 0, igp, jgp) +
        dinv(kv.ie, 0, 1, igp, jgp) * v_buf(kv.ie, buf_lev, 1, igp, jgp);
    grad_s(1, igp, jgp, kv.ilev) =
        dinv(kv.ie, 1, 0, igp, jgp) * v_buf(kv.ie, buf_lev, 0, igp, jgp) +
        dinv(kv.ie, 1, 1, igp, jgp) * v_buf(kv.ie, buf_lev, 1, igp, jgp);
  });
}

KOKKOS_INLINE_FUNCTION void gradient_sphere_update(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
    ExecViewUnmanaged<Scalar * [NUM_LEV][2][NP][NP]> v_buf,
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s) {
  const int buf_lev = (kv.ilev < 0 || kv.ilev >= NUM_LEV) ? 0 : kv.ilev;
  constexpr int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dsdx(0.0), dsdy(0.0);
    for (int kgp = 0; kgp < NP; ++kgp) {
      dsdx += dvv(jgp, kgp) * scalar(igp, kgp, kv.ilev);
      dsdy += dvv(jgp, kgp) * scalar(kgp, igp, kv.ilev);
    }
    v_buf(kv.ie, buf_lev, 0, igp, jgp) = dsdx * PhysicalConstants::rrearth;
    v_buf(kv.ie, buf_lev, 1, jgp, igp) = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    grad_s(0, igp, jgp, kv.ilev) =
        dinv(kv.ie, 0, 0, igp, jgp) * v_buf(kv.ie, buf_lev, 0, igp, jgp) +
        dinv(kv.ie, 0, 1, igp, jgp) * v_buf(kv.ie, buf_lev, 1, igp, jgp);
    grad_s(1, igp, jgp, kv.ilev) =
        dinv(kv.ie, 1, 0, igp, jgp) * v_buf(kv.ie, buf_lev, 0, igp, jgp) +
        dinv(kv.ie, 1, 1, igp, jgp) * v_buf(kv.ie, buf_lev, 1, igp, jgp);
  });
}

KOKKOS_INLINE_FUNCTION void
divergence_sphere(const KernelVariables &kv,
                  const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                  const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> v,
                  ExecViewUnmanaged<Scalar * [NUM_LEV][2][NP][NP]> gv_buf,
                  ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> div_v) {
  const int buf_lev = (kv.ilev < 0 || kv.ilev >= NUM_LEV) ? 0 : kv.ilev;
  constexpr int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    gv_buf(kv.ie, buf_lev, 0, igp, jgp) =
        (dinv(kv.ie, 0, 0, igp, jgp) * v(0, igp, jgp, kv.ilev) +
         dinv(kv.ie, 1, 0, igp, jgp) * v(1, igp, jgp, kv.ilev)) *
        metdet(kv.ie, igp, jgp);
    gv_buf(kv.ie, buf_lev, 1, igp, jgp) =
        (dinv(kv.ie, 0, 1, igp, jgp) * v(0, igp, jgp, kv.ilev) +
         dinv(kv.ie, 1, 1, igp, jgp) * v(1, igp, jgp, kv.ilev)) *
        metdet(kv.ie, igp, jgp);
  });

  // j, l, i -> i, j, k
  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dudx = 0.0, dvdy = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dudx += dvv(jgp, kgp) * gv_buf(kv.ie, buf_lev, 0, igp, kgp);
      dvdy += dvv(igp, kgp) * gv_buf(kv.ie, buf_lev, 1, kgp, jgp);
    }
    div_v(igp, jgp, kv.ilev) =
        (dudx + dvdy) *
        ((1.0 / metdet(kv.ie, igp, jgp)) * PhysicalConstants::rrearth);
  });
}

// Note: this updates the field div_v as follows:
//     div_v = beta*div_v + alpha*div(v)
KOKKOS_INLINE_FUNCTION void
divergence_sphere_update(const KernelVariables &kv,
                         const Real alpha, const Real beta,
                         const ExecViewUnmanaged<const Real [2][2][NP][NP]> dinv,
                         const ExecViewUnmanaged<const Real [NP][NP]> metdet,
                         const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                         const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> v,
                         const ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> div_v) {
  constexpr int contra_iters = NP * NP;
  Scalar gv[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    gv[0][igp][jgp] = (dinv(0, 0, igp, jgp) * v(0, igp, jgp, kv.ilev) + dinv(1, 0, igp, jgp) * v(1, igp, jgp, kv.ilev)) * metdet(igp, jgp);
    gv[1][igp][jgp] = (dinv(0, 1, igp, jgp) * v(0, igp, jgp, kv.ilev) + dinv(1, 1, igp, jgp) * v(1, igp, jgp, kv.ilev)) * metdet(igp, jgp);
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dudx = 0.0, dvdy = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dudx += dvv(jgp, kgp) * gv[0][igp][kgp];
      dvdy += dvv(igp, kgp) * gv[1][kgp][jgp];
    }

    div_v(igp,jgp,kv.ilev) *= beta;
    div_v(igp,jgp,kv.ilev) += alpha*((dudx + dvdy) * ((1.0 / metdet(igp, jgp)) * PhysicalConstants::rrearth));
  });
}

KOKKOS_INLINE_FUNCTION void
vorticity_sphere(const KernelVariables &kv,
                 const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                 const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> u,
                 const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> v,
                 ExecViewUnmanaged<Scalar * [NUM_LEV][2][NP][NP]> vcov_buf,
                 ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> vort) {
  const int buf_lev = (kv.ilev < 0 || kv.ilev >= NUM_LEV) ? 0 : kv.ilev;
  constexpr int covar_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, covar_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    vcov_buf(kv.ie, buf_lev, 0, jgp, igp) =
        d(kv.ie, 0, 0, jgp, igp) * u(jgp, igp, kv.ilev) +
        d(kv.ie, 0, 1, jgp, igp) * v(jgp, igp, kv.ilev);
    vcov_buf(kv.ie, buf_lev, 1, jgp, igp) =
        d(kv.ie, 1, 0, jgp, igp) * u(jgp, igp, kv.ilev) +
        d(kv.ie, 1, 1, jgp, igp) * v(jgp, igp, kv.ilev);
  });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, vort_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dudy = 0.0;
    Scalar dvdx = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dvdx += dvv(jgp, kgp) * vcov_buf(kv.ie, buf_lev, 1, igp, kgp);
      dudy += dvv(igp, kgp) * vcov_buf(kv.ie, buf_lev, 0, kgp, jgp);
    }
    vort(igp, jgp, kv.ilev) = (dvdx - dudy) * ((1.0 / metdet(kv.ie, igp, jgp)) *
                                               PhysicalConstants::rrearth);
  });
}

//Why does the prev version take u and v separately? 
//rewriting this to take vector as the input.
KOKKOS_INLINE_FUNCTION void
vorticity_sphere_vector(const KernelVariables &kv,
                 const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                 const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> v,
                 ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> vort) {
  constexpr int covar_iters = NP * NP;
  Scalar vcov[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, covar_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    vcov[0][jgp][igp] = d(kv.ie, 0, 0, jgp, igp) * v(0,jgp, igp, kv.ilev) 
                      + d(kv.ie, 0, 1, jgp, igp) * v(1,jgp, igp, kv.ilev);
    vcov[1][jgp][igp] = d(kv.ie, 1, 0, jgp, igp) * v(0,jgp, igp, kv.ilev) 
                      + d(kv.ie, 1, 1, jgp, igp) * v(1,jgp, igp, kv.ilev);
  });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, vort_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dudy = 0.0;
    Scalar dvdx = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dvdx += dvv(jgp, kgp) * vcov[1][igp][kgp];
      dudy += dvv(igp, kgp) * vcov[0][kgp][jgp];
    }
    vort(igp, jgp, kv.ilev) = (dvdx - dudy) * ((1.0 / metdet(kv.ie, igp, jgp)) *
                                               PhysicalConstants::rrearth);
  });
}


KOKKOS_INLINE_FUNCTION void divergence_sphere_wk(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> v,
    ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> div_v) {

  constexpr int contra_iters = NP * NP;
  Scalar gv[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    gv[0][igp][jgp] = dinv(kv.ie, 0, 0, igp, jgp) * v(0, igp, jgp, kv.ilev) +
                      dinv(kv.ie, 1, 0, igp, jgp) * v(1, igp, jgp, kv.ilev);
    gv[1][igp][jgp] = dinv(kv.ie, 0, 1, igp, jgp) * v(0, igp, jgp, kv.ilev) +
                      dinv(kv.ie, 1, 1, igp, jgp) * v(1, igp, jgp, kv.ilev);
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int mgp = loop_idx / NP;
    const int ngp = loop_idx % NP;
    Scalar dd = 0.0;
    for (int jgp = 0; jgp < NP; ++jgp) {
      dd -= (spheremp(kv.ie, ngp, jgp) * gv[0][ngp][jgp] * dvv(jgp, mgp) +
             spheremp(kv.ie, jgp, mgp) * gv[1][jgp][mgp] * dvv(jgp, ngp)) *
            PhysicalConstants::rrearth;
    }
    div_v(ngp, mgp, kv.ilev) = dd;
  });

}//end of divergence_sphere_wk

//analog of laplace_simple_c_callable
KOKKOS_INLINE_FUNCTION void laplace_simple(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> DInv, // for grad, div
    const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,     // for div
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s, // temp to store grad
    const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> field,         // input
    ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace) {
    // let's ignore var coef and tensor hv
       gradient_sphere(kv, DInv, dvv, field, grad_s);
       divergence_sphere_wk(kv, DInv, spheremp, dvv, grad_s, laplace);
}//end of laplace_simple

//analog of laplace_wk_c_callable
//but without if-statements for hypervis_power, var_coef, and hypervis_scaling.
//for 2d fields, there should be either laplace_simple, or laplace_tensor for the whole run.
KOKKOS_INLINE_FUNCTION void laplace_tensor(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> DInv, // for grad, div
    const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,     // for div
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> tensorVisc,
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s, // temp to store grad
    const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> field,         // input
    ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace) {

       gradient_sphere(kv, DInv, dvv, field, grad_s);
//now multiply tensorVisc(:,:,i,j)*grad_s(i,j) (matrix*vector, independent of i,j )
//but it requires a temp var to store a result. the result is then placed to grad_s,
//or should it be an extra temp var instead of an extra loop?
       constexpr int num_iters = NP * NP;
       Scalar gv[2][NP][NP];
       Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, num_iters),
                          [&](const int loop_idx) {
          const int igp = loop_idx / NP;
          const int jgp = loop_idx % NP;
          gv[0][igp][jgp] = tensorVisc(kv.ie,0,0,igp,jgp) * grad_s(0,igp,jgp, kv.ilev) +
                            tensorVisc(kv.ie,1,0,igp,jgp) * grad_s(1,igp,jgp, kv.ilev);
          gv[1][igp][jgp] = tensorVisc(kv.ie,0,1,igp,jgp) * grad_s(0,igp,jgp, kv.ilev) +
                            tensorVisc(kv.ie,1,1,igp,jgp) * grad_s(1,igp,jgp, kv.ilev);
       });

       Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, num_iters),
                          [&](const int loop_idx) {
          const int igp = loop_idx / NP;
          const int jgp = loop_idx % NP;
          grad_s(0,igp,jgp, kv.ilev) = gv[0][igp][jgp];
          grad_s(1,igp,jgp, kv.ilev) = gv[1][igp][jgp];
       });

       divergence_sphere_wk(kv, DInv, spheremp, dvv, grad_s, laplace);
}//end of laplace_tensor


//a version of laplace_tensor where input is replaced by output
KOKKOS_INLINE_FUNCTION void laplace_tensor_replace(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> DInv, // for grad, div
    const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,     // for div
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> tensorVisc,
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s, // temp to store grad
    ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace) { //input/output

       gradient_sphere(kv, DInv, dvv, laplace, grad_s);
       constexpr int num_iters = NP * NP;
       Scalar gv[2][NP][NP];
       Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, num_iters),
                          [&](const int loop_idx) {
          const int igp = loop_idx / NP;
          const int jgp = loop_idx % NP;
          gv[0][igp][jgp] = tensorVisc(kv.ie,0,0,igp,jgp) * grad_s(0,igp,jgp, kv.ilev) +
                            tensorVisc(kv.ie,1,0,igp,jgp) * grad_s(1,igp,jgp, kv.ilev);
          gv[1][igp][jgp] = tensorVisc(kv.ie,0,1,igp,jgp) * grad_s(0,igp,jgp, kv.ilev) +
                            tensorVisc(kv.ie,1,1,igp,jgp) * grad_s(1,igp,jgp, kv.ilev);
       });

       Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, num_iters),
                          [&](const int loop_idx) {
          const int igp = loop_idx / NP;
          const int jgp = loop_idx % NP;
          grad_s(0,igp,jgp, kv.ilev) = gv[0][igp][jgp];
          grad_s(1,igp,jgp, kv.ilev) = gv[1][igp][jgp];
       });

       divergence_sphere_wk(kv, DInv, spheremp, dvv, grad_s, laplace);
}//end of laplace_tensor_replace

//check mp, why is it an ie quantity?
KOKKOS_INLINE_FUNCTION void
curl_sphere_wk_testcov(const KernelVariables &kv,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> D,
                const ExecViewUnmanaged<const Real * [NP][NP]> mp,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> curls) {

  Scalar dscontra[2][NP][NP];
  constexpr int np_squared = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
    dscontra[0][igp][jgp] = 0.0; 
    dscontra[1][igp][jgp] = 0.0; 

  });
//in here, which array should be addressed fastest?
  constexpr int np_cubed = NP * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_cubed),
                       [&](const int loop_idx) {
    const int ngp = loop_idx / NP / NP; //slowest
    const int mgp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP; //fastest
//One can move multiplication by rrearth to the last loop, but it breaks BFB
//property for curl. 
    dscontra[0][ngp][mgp] -= 
       mp(kv.ie,jgp,mgp)*scalar(jgp,mgp,kv.ilev)*dvv(jgp,ngp); 
    dscontra[1][ngp][mgp] += 
       mp(kv.ie,ngp,jgp)*scalar(ngp,jgp,kv.ilev)*dvv(jgp,mgp); 
  });

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
    curls(0,igp,jgp,kv.ilev) = (D(kv.ie,0,0,igp,jgp)*dscontra[0][igp][jgp]
                             + D(kv.ie,1,0,igp,jgp)*dscontra[1][igp][jgp])
                             *PhysicalConstants::rrearth; 
    curls(1,igp,jgp,kv.ilev) = (D(kv.ie,0,1,igp,jgp)*dscontra[0][igp][jgp]
                             + D(kv.ie,1,1,igp,jgp)*dscontra[1][igp][jgp])
                             *PhysicalConstants::rrearth; 
  });
}


KOKKOS_INLINE_FUNCTION void
grad_sphere_wk_testcov(const KernelVariables &kv,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> D,
                const ExecViewUnmanaged<const Real * [NP][NP]> mp,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> metinv,
                const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grads) {

  Scalar dscontra[2][NP][NP];
  constexpr int np_squared = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
    dscontra[0][igp][jgp] = 0.0;
    dscontra[1][igp][jgp] = 0.0;

  });

  constexpr int np_cubed = NP * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_cubed),
                       [&](const int loop_idx) {
    const int ngp = loop_idx / NP / NP; //slowest
    const int mgp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP; //fastest
    dscontra[0][ngp][mgp] -=(
       mp(kv.ie,ngp,jgp)*
       metinv(kv.ie,0,0,ngp,mgp)*
       metdet(kv.ie,ngp,mgp)*
       scalar(ngp,jgp,kv.ilev)*
       dvv(jgp,mgp)
       +
       mp(kv.ie,jgp,mgp)*
       metinv(kv.ie,0,1,ngp,mgp)*
       metdet(kv.ie,ngp,mgp)*
       scalar(jgp,mgp,kv.ilev)*
       dvv(jgp,ngp));
//                            )*PhysicalConstants::rrearth;

    dscontra[1][ngp][mgp] -=(
       mp(kv.ie,ngp,jgp)*
       metinv(kv.ie,1,0,ngp,mgp)*
       metdet(kv.ie,ngp,mgp)*
       scalar(ngp,jgp,kv.ilev)*
       dvv(jgp,mgp)
       +
       mp(kv.ie,jgp,mgp)*
       metinv(kv.ie,1,1,ngp,mgp)*
       metdet(kv.ie,ngp,mgp)*
       scalar(jgp,mgp,kv.ilev)*
       dvv(jgp,ngp));
//                            )*PhysicalConstants::rrearth;

  });

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//don't forget to move rrearth here and in curl and in F code.
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
    grads(0,igp,jgp,kv.ilev) = (D(kv.ie,0,0,igp,jgp)*dscontra[0][igp][jgp]
                             + D(kv.ie,1,0,igp,jgp)*dscontra[1][igp][jgp])
                             *PhysicalConstants::rrearth;
    grads(1,igp,jgp,kv.ilev) = (D(kv.ie,0,1,igp,jgp)*dscontra[0][igp][jgp]
                             + D(kv.ie,1,1,igp,jgp)*dscontra[1][igp][jgp])
                             *PhysicalConstants::rrearth;
  });
}


//this version needs too many temp vars
//needs dvv, dinv, spheremp, tensor, vec_sph2cart, 
//NOT TESTED, not verified, MISSING LINES FOR RIGID ROTATION
KOKKOS_INLINE_FUNCTION void
vlaplace_sphere_wk_cartesian(const KernelVariables &kv,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> Dinv,
                const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> tensorVisc,
                const ExecViewUnmanaged<const Real * [2][3][NP][NP]> vec_sph2cart,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
//temps to store results
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grads,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> component0,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> component1,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> component2,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace0,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace1,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace2,
                const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> vector,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> laplace) {

//  Scalar dum_cart[2][NP][NP];
  constexpr int np_squared = NP * NP;
//  constexpr int np_squared_3;
/* // insert after debugging? still won't work because dum_comp cannot be input for laplace
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared_3),
                       [&](const int loop_idx) {
    const int comp = loop_idx % 3 ;        //fastest
    const int igp = (loop_idx / 3 ) / NP ; //slowest
    const int jgp = (loop_idx / 3 ) % NP; 
    dum_cart[comp][igp][jgp] = vec_sph2cart(kv.ie,0,comp,igp,jgp)*vector(0,igp,jgp)
                        + vec_sph2cart(kv.ie,1,comp,igp,jgp)*vector(1,igp,jgp) ;
}
*/
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
//this is for debug
    component0(igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,0,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                        + vec_sph2cart(kv.ie,1,0,igp,jgp)*vector(1,igp,jgp,kv.ilev) ;
    component1(igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,1,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                        + vec_sph2cart(kv.ie,1,1,igp,jgp)*vector(1,igp,jgp,kv.ilev) ;
    component2(igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,2,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                        + vec_sph2cart(kv.ie,1,2,igp,jgp)*vector(1,igp,jgp,kv.ilev) ;
  });
//apply laplace to each component
//WE NEED LAPLACE_UPDATE(or replace?), way too many temp vars
  laplace_tensor(kv,Dinv,spheremp,dvv,tensorVisc,grads,component0,laplace0);
  laplace_tensor(kv,Dinv,spheremp,dvv,tensorVisc,grads,component1,laplace1);
  laplace_tensor(kv,Dinv,spheremp,dvv,tensorVisc,grads,component2,laplace2);

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
    laplace(0,igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,0,igp,jgp)*laplace0(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,0,1,igp,jgp)*laplace1(igp,jgp,kv.ilev) 
                       + vec_sph2cart(kv.ie,0,2,igp,jgp)*laplace2(igp,jgp,kv.ilev);

    laplace(1,igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,1,0,igp,jgp)*laplace0(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,1,1,igp,jgp)*laplace1(igp,jgp,kv.ilev) 
                       + vec_sph2cart(kv.ie,1,2,igp,jgp)*laplace2(igp,jgp,kv.ilev);
                       
  });


}//end of vlaplace_cartesian



KOKKOS_INLINE_FUNCTION void
vlaplace_sphere_wk_cartesian_reduced(const KernelVariables &kv,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> Dinv,
                const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> tensorVisc,
                const ExecViewUnmanaged<const Real * [2][3][NP][NP]> vec_sph2cart,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
//temp vars
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grads,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace0,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace1,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> laplace2,
//input
                const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> vector,
//output
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> laplace) {
  constexpr int np_squared = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
    laplace0(igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,0,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                        + vec_sph2cart(kv.ie,1,0,igp,jgp)*vector(1,igp,jgp,kv.ilev) ;
    laplace1(igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,1,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                        + vec_sph2cart(kv.ie,1,1,igp,jgp)*vector(1,igp,jgp,kv.ilev) ;
    laplace2(igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,2,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                        + vec_sph2cart(kv.ie,1,2,igp,jgp)*vector(1,igp,jgp,kv.ilev) ;
  });

  laplace_tensor_replace(kv,Dinv,spheremp,dvv,tensorVisc,grads,laplace0);
  laplace_tensor_replace(kv,Dinv,spheremp,dvv,tensorVisc,grads,laplace1);
  laplace_tensor_replace(kv,Dinv,spheremp,dvv,tensorVisc,grads,laplace2);

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
#define UNDAMPRRCART
#ifdef UNDAMPRRCART
    laplace(0,igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,0,igp,jgp)*laplace0(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,0,1,igp,jgp)*laplace1(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,0,2,igp,jgp)*laplace2(igp,jgp,kv.ilev)
                       + 2.0*spheremp(kv.ie,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                               *(PhysicalConstants::rrearth)*(PhysicalConstants::rrearth);

    laplace(1,igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,1,0,igp,jgp)*laplace0(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,1,1,igp,jgp)*laplace1(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,1,2,igp,jgp)*laplace2(igp,jgp,kv.ilev)
                       + 2.0*spheremp(kv.ie,igp,jgp)*vector(1,igp,jgp,kv.ilev)
                               *(PhysicalConstants::rrearth)*(PhysicalConstants::rrearth);
#else
    laplace(0,igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,0,0,igp,jgp)*laplace0(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,0,1,igp,jgp)*laplace1(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,0,2,igp,jgp)*laplace2(igp,jgp,kv.ilev);
    laplace(1,igp,jgp,kv.ilev) = vec_sph2cart(kv.ie,1,0,igp,jgp)*laplace0(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,1,1,igp,jgp)*laplace1(igp,jgp,kv.ilev)
                       + vec_sph2cart(kv.ie,1,2,igp,jgp)*laplace2(igp,jgp,kv.ilev);
#endif
  });
} // end of divergence_sphere_wk

/*
#define UNDAMPRRCART
#ifdef UNDAMPRRCART
//rigid rotation is not damped
//this code can be brought to the loop above

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
    laplace(0,igp,jgp,kv.ilev) += 2.0*spheremp(kv.ie,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                               *(PhysicalConstants::rrearth)*(PhysicalConstants::rrearth);

    laplace(1,igp,jgp,kv.ilev) += 2.0*spheremp(kv.ie,igp,jgp)*vector(1,igp,jgp,kv.ilev)
                               *(PhysicalConstants::rrearth)*(PhysicalConstants::rrearth);
  });

#endif
*/


}//end of vlaplace_cartesian_reduced


KOKKOS_INLINE_FUNCTION void
vlaplace_sphere_wk_contra(const KernelVariables &kv,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                const ExecViewUnmanaged<const Real * [NP][NP]> mp,
                const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> metinv,
                const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const Real nu_ratio,
//temps
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> div,
                ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> vort,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> gradcov,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> curlcov,
//input, later write a version to replace input with output
                const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> vector,
//output
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> laplace) {

   divergence_sphere(kv,dinv,metdet,dvv,vector,div);
   vorticity_sphere_vector(kv,d,metdet,dvv,vector,vort);

   constexpr int np_squared = NP * NP;
   Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
     const int igp = loop_idx / NP; //slow
     const int jgp = loop_idx % NP; //fast
     div(igp,jgp,kv.ilev) *= nu_ratio;
   });

   grad_sphere_wk_testcov(kv,d,mp,metinv,metdet,dvv,div,gradcov);
   curl_sphere_wk_testcov(kv,d,mp,dvv,vort,curlcov); 

   Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
     const int igp = loop_idx / NP; //slow
     const int jgp = loop_idx % NP; //fast

#define UNDAMPRRCART
#ifdef UNDAMPRRCART
     laplace(0,igp,jgp,kv.ilev) = 2.0*spheremp(kv.ie,igp,jgp)*vector(0,igp,jgp,kv.ilev)
                                *(PhysicalConstants::rrearth)*(PhysicalConstants::rrearth);

     laplace(1,igp,jgp,kv.ilev) = 2.0*spheremp(kv.ie,igp,jgp)*vector(1,igp,jgp,kv.ilev)
                                *(PhysicalConstants::rrearth)*(PhysicalConstants::rrearth);
#endif

     laplace(0,igp,jgp,kv.ilev) += (gradcov(0,igp,jgp,kv.ilev)- curlcov(0,igp,jgp,kv.ilev));
     laplace(1,igp,jgp,kv.ilev) += (gradcov(1,igp,jgp,kv.ilev)- curlcov(1,igp,jgp,kv.ilev));
   });

}//end of vlaplace_sphere_wk_contra

} // namespace Homme

#endif // HOMMEXX_SPHERE_OPERATORS_HPP
