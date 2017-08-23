#ifndef HOMMEXX_SPHERE_OPERATORS_HPP
#define HOMMEXX_SPHERE_OPERATORS_HPP

#include "Types.hpp"
#include "CaarRegion.hpp"
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
                   ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  constexpr int contra_iters = NP * NP;
  // TODO: Use scratch space for this
  Real temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx(0), dsdy(0);
    for (int i = 0; i < NP; ++i) {
      dsdx += dvv(l, i) * scalar(j, i);
      dsdy += dvv(l, i) * scalar(i, j);
    }
    temp_v[0][j][l] = dsdx * PhysicalConstants::rrearth;
    temp_v[1][l][j] = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h, j, i) = dinv(kv.ie, h, 0, j, i) * temp_v[0][j][i] +
                      dinv(kv.ie, h, 1, j, i) * temp_v[1][j][i];
  });
}

KOKKOS_INLINE_FUNCTION void gradient_sphere_update_sl(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Real[NP][NP]> scalar,
    ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  constexpr int contra_iters = NP * NP;
  Real temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx(0), dsdy(0);
    for (int i = 0; i < NP; ++i) {
      dsdx += dvv(l, i) * scalar(j, i);
      dsdy += dvv(l, i) * scalar(i, j);
    }
    temp_v[0][j][l] = dsdx * PhysicalConstants::rrearth;
    temp_v[1][l][j] = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = 2 * NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int h = (loop_idx / NP) / NP;
    const int i = (loop_idx / NP) % NP;
    const int j = loop_idx % NP;
    grad_s(h, j, i) += dinv(kv.ie, h, 0, j, i) * temp_v[0][j][i] +
                       dinv(kv.ie, h, 1, j, i) * temp_v[1][j][i];
  });
}

KOKKOS_INLINE_FUNCTION void
divergence_sphere_sl(const KernelVariables &kv,
                     const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                     const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                     const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                     const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                     ExecViewUnmanaged<Real[NP][NP]> div_v) {
  constexpr int contra_iters = NP * NP * 2;
  Real gv[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int hgp = (loop_idx / NP) / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    gv[hgp][igp][jgp] = (dinv(kv.ie, 0, hgp, igp, jgp) * v(0, igp, jgp) +
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
      dudx += dvv(igp, kgp) * gv[0][jgp][kgp];
      dvdy += dvv(jgp, kgp) * gv[1][kgp][igp];
    }
    div_v(jgp, igp) = (dudx + dvdy) * ((1.0 / metdet(kv.ie, jgp, igp)) *
                                       PhysicalConstants::rrearth);
  });
}

KOKKOS_INLINE_FUNCTION void
divergence_sphere_wk_sl(const KernelVariables &kv,
                  const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                  const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                  ExecViewUnmanaged<Real[NP][NP]> div_v) {

//copied from strong divergence as is but without metdet
//conversion to contravariant
  constexpr int contra_iters = NP * NP * 2;
  Real gv[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int hgp = (loop_idx / NP) / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    gv[hgp][igp][jgp] = dinv(kv.ie,0,hgp,igp,jgp) * v(0,igp,jgp) +
                        dinv(kv.ie,1,hgp,igp,jgp) * v(1,igp,jgp);
  });

//in strong div
//kgp = i in strong code, jgp=j, igp=l
//in weak div, n is like j in strong div, 
//n(weak)=j(strong)=jgp
//m(weak)=l(strong)=igp
//j(weak)=i(strong)=kgp
  constexpr int div_iters = NP * NP;
//keeping indices' names as in F
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int mgp = loop_idx / NP;
    const int ngp = loop_idx % NP;
    Real dd = 0.0;
    for (int jgp = 0; jgp < NP; ++jgp) {
      dd -= (  spheremp(kv.ie,ngp,jgp)*gv[0][ngp][jgp]*dvv(jgp,mgp)
             + spheremp(kv.ie,jgp,mgp)*gv[1][jgp][mgp]*dvv(jgp,ngp) )
                         *PhysicalConstants::rrearth;
    }
    div_v(ngp,mgp) = dd;
  });

}//end of divergence_sphere_wk_sl



// Note that divergence_sphere requires scratch space of 3 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
vorticity_sphere_sl(const KernelVariables &kv,
                    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                    const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                    const ExecViewUnmanaged<const Real[NP][NP]> u,
                    const ExecViewUnmanaged<const Real[NP][NP]> v,
                    ExecViewUnmanaged<Real[NP][NP]> vort) {
  constexpr int covar_iters = 2 * NP * NP;
  Real vcov[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, covar_iters),
                       [&](const int loop_idx) {
    const int hgp = loop_idx / NP / NP;
    const int igp = (loop_idx / NP) % NP;
    const int jgp = loop_idx % NP;
    vcov[hgp][igp][jgp] = d(kv.ie, hgp, 0, igp, jgp) * u(igp, jgp) +
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
      dvdx += dvv(igp, kgp) * vcov[1][jgp][kgp];
      dudy += dvv(jgp, kgp) * vcov[0][kgp][igp];
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
    const ExecViewUnmanaged<const Real * [NP][NP]> spheremp,     // for div
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,           // for grad, div
//how to get rid of this temp var? passing real* instead of kokkos view
////does not work. is creating kokkos temorary in a kernel the correct way?
    ExecViewUnmanaged<Real[2][NP][NP]> grad_s, // temp to store grad
    const ExecViewUnmanaged<const Real[NP][NP]> field,         // input
    // output
    ExecViewUnmanaged<Real[NP][NP]> laplace) {
    // Real grad_s[2][NP][NP];
    // let's ignore var coef and tensor hv
       gradient_sphere_sl(kv, DInv, dvv, field, grad_s);
       divergence_sphere_wk_sl(kv, DInv, spheremp, dvv, grad_s, laplace);
}//end of laplace_wk_sl


// ================ MULTI-LEVEL IMPLEMENTATION =========================== //


KOKKOS_INLINE_FUNCTION void
gradient_sphere(const KernelVariables &kv,
                const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
                ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s) {
  constexpr int contra_iters = NP * NP;
  // TODO: Use scratch space for this
  Scalar temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dsdx(0.0), dsdy(0.0);
    for (int kgp = 0; kgp < NP; ++kgp) {
      dsdx += dvv(jgp, kgp) * scalar(igp, kgp, kv.ilev);
      dsdy += dvv(jgp, kgp) * scalar(kgp, igp, kv.ilev);
    }
    temp_v[0][igp][jgp] = dsdx * PhysicalConstants::rrearth;
    temp_v[1][jgp][igp] = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    grad_s(0, igp, jgp, kv.ilev) = dinv(kv.ie, 0, 0, igp, jgp) * temp_v[0][igp][jgp] + dinv(kv.ie, 0, 1, igp, jgp) * temp_v[1][igp][jgp];
    grad_s(1, igp, jgp, kv.ilev) = dinv(kv.ie, 1, 0, igp, jgp) * temp_v[0][igp][jgp] + dinv(kv.ie, 1, 1, igp, jgp) * temp_v[1][igp][jgp];
  });
}

KOKKOS_INLINE_FUNCTION void gradient_sphere_update(
    const KernelVariables &kv,
    const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
    const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> scalar,
    ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> grad_s) {
  constexpr int contra_iters = NP * NP;
  Scalar temp_v[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Scalar dsdx(0.0), dsdy(0.0);
    for (int kgp = 0; kgp < NP; ++kgp) {
      dsdx += dvv(jgp, kgp) * scalar(igp, kgp, kv.ilev);
      dsdy += dvv(igp, kgp) * scalar(kgp, jgp, kv.ilev);
    }
    temp_v[0][igp][jgp] = dsdx * PhysicalConstants::rrearth;
    temp_v[1][igp][jgp] = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, grad_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    grad_s(0, igp, jgp, kv.ilev) += dinv(kv.ie, 0, 0, igp, jgp) * temp_v[0][igp][jgp] + dinv(kv.ie, 0, 1, igp, jgp) * temp_v[1][igp][jgp];
    grad_s(1, igp, jgp, kv.ilev) += dinv(kv.ie, 1, 0, igp, jgp) * temp_v[0][igp][jgp] + dinv(kv.ie, 1, 1, igp, jgp) * temp_v[1][igp][jgp];
  });
}

KOKKOS_INLINE_FUNCTION void
divergence_sphere(const KernelVariables &kv,
                  const ExecViewUnmanaged<const Real * [2][2][NP][NP]> dinv,
                  const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                  const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                  const ExecViewUnmanaged<const Scalar[2][NP][NP][NUM_LEV]> v,
                  ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> div_v) {
  constexpr int contra_iters = NP * NP;
  Scalar gv[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    gv[0][igp][jgp] = (dinv(kv.ie, 0, 0, igp, jgp) * v(0, igp, jgp, kv.ilev) + dinv(kv.ie, 1, 0, igp, jgp) * v(1, igp, jgp, kv.ilev)) * metdet(kv.ie, igp, jgp);
    gv[1][igp][jgp] = (dinv(kv.ie, 0, 1, igp, jgp) * v(0, igp, jgp, kv.ilev) + dinv(kv.ie, 1, 1, igp, jgp) * v(1, igp, jgp, kv.ilev)) * metdet(kv.ie, igp, jgp);
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
    div_v(igp, jgp, kv.ilev) = (dudx + dvdy) * ((1.0 / metdet(kv.ie, igp, jgp)) * PhysicalConstants::rrearth);
  });
}

KOKKOS_INLINE_FUNCTION void
vorticity_sphere(const KernelVariables &kv,
                 const ExecViewUnmanaged<const Real * [2][2][NP][NP]> d,
                 const ExecViewUnmanaged<const Real * [NP][NP]> metdet,
                 const ExecViewUnmanaged<const Real[NP][NP]> dvv,
                 const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> u,
                 const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]> v,
                 ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]> vort) {
  constexpr int covar_iters = NP * NP;
  Scalar vcov[2][NP][NP];
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, covar_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    vcov[0][jgp][igp] = d(kv.ie, 0, 0, jgp, igp) * u(jgp, igp, kv.ilev) + d(kv.ie, 0, 1, jgp, igp) * v(jgp, igp, kv.ilev);
    vcov[1][jgp][igp] = d(kv.ie, 1, 0, jgp, igp) * u(jgp, igp, kv.ilev) + d(kv.ie, 1, 1, jgp, igp) * v(jgp, igp, kv.ilev);
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


KOKKOS_INLINE_FUNCTION void
divergence_sphere_wk(const KernelVariables &kv,
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
    gv[0][igp][jgp] = dinv(kv.ie,0,0,igp,jgp) * v(0,igp,jgp, kv.ilev) +
                      dinv(kv.ie,1,0,igp,jgp) * v(1,igp,jgp, kv.ilev);
    gv[1][igp][jgp] = dinv(kv.ie,0,1,igp,jgp) * v(0,igp,jgp, kv.ilev) +
                      dinv(kv.ie,1,1,igp,jgp) * v(1,igp,jgp, kv.ilev);
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, div_iters),
                       [&](const int loop_idx) {
    const int mgp = loop_idx / NP;
    const int ngp = loop_idx % NP;
    Scalar dd = 0.0;
    for (int jgp = 0; jgp < NP; ++jgp) {
      dd -= (  spheremp(kv.ie,ngp,jgp)*gv[0][ngp][jgp]*dvv(jgp,mgp)
             + spheremp(kv.ie,jgp,mgp)*gv[1][jgp][mgp]*dvv(jgp,ngp) )
                         *PhysicalConstants::rrearth;
    }
    div_v(ngp,mgp,kv.ilev) = dd;
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
//move this multiplication to the last loop after debug
    dscontra[0][ngp][mgp] -= 
       mp(kv.ie,jgp,mgp)*scalar(jgp,mgp,kv.ilev)*dvv(jgp,ngp)*PhysicalConstants::rrearth; 
    dscontra[1][ngp][mgp] += 
       mp(kv.ie,ngp,jgp)*scalar(ngp,jgp,kv.ilev)*dvv(jgp,mgp)*PhysicalConstants::rrearth; 
  });

for(int i=0; i< NP; i++)
for(int j=0; j< NP; j++){

std::cout << "i=" << i << ", j=" << j << ", ds = " 
<< dscontra[0][i][j][0] << ", " << dscontra[1][i][j][0] <<"\n";
}

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, np_squared),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP; //slowest
    const int jgp = loop_idx % NP; //fastest
    curls(0,igp,jgp,kv.ilev) = D(kv.ie,0,0,igp,jgp)*dscontra[0][igp][jgp]
                             + D(kv.ie,1,0,igp,jgp)*dscontra[1][igp][jgp]; 
    curls(1,igp,jgp,kv.ilev) = D(kv.ie,0,1,igp,jgp)*dscontra[0][igp][jgp]
                             + D(kv.ie,1,1,igp,jgp)*dscontra[1][igp][jgp]; 
  });
}





} // namespace Homme

#endif // HOMMEXX_SPHERE_OPERATORS_HPP
