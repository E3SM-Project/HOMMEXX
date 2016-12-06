
#include <Types.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

#include <cmath>

namespace Homme {

constexpr const real rearth = 6.376E6;
constexpr const real rrearth = 1.0 / rearth;

template <typename ScalarQP>
void gradient_sphere_c(int ie, const ScalarQP &s,
                       const Dvv &dvv, const D &dinv,
                       Vector_Field &grad) {
  Homme_Local<real *> dsd("Velocity Spatial Derivatives",
                          dim);
  Homme_Local<real ***> v("Velocity", np, np, dim);
  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int k = 0; k < dim; k++) {
        dsd(k) = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dsd(0) += dvv(i, l) * s(i, j);
        dsd(1) += dvv(i, l) * s(j, i);
      }
      v(l, j, 0) = dsd(0) * rrearth;
      v(j, l, 1) = dsd(1) * rrearth;
    }
  }
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < dim; h++) {
        grad(i, j, h) = dinv(i, j, 0, h, ie) * v(i, j, 0) +
                        dinv(i, j, 1, h, ie) * v(i, j, 1);
      }
    }
  }
}

void vorticity_sphere_c(int ie, const Vector_Field &v,
                        const Dvv &dvv, const D &d,
                        const MetDet &rmetdet,
                        Scalar_Field &vorticity) {
  Homme_Local<real *> dvd("Velocity Spatial Derivatives",
                          dim);
  Homme_Local<real ***> vco("buffer for metric adjustments",
                            np, np, dim);
  Homme_Local<real **> vtemp("buffer for performance", np,
                             np);

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < dim; h++) {
        vco(i, j, h) = d(i, j, 0, h, ie) * v(i, j, 0) +
                       d(i, j, 1, h, ie) * v(i, j, 1);
      }
    }
  }

  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int h = 0; h < dim; h++) {
        dvd(h) = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dvd(0) += dvv(i, l) * vco(i, j, 1);
        dvd(1) += dvv(i, l) * vco(j, i, 0);
      }
      vorticity(l, j) = dvd(0);
      vtemp(j, l) = dvd(1);
    }
  }

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      vorticity(i, j) = (vorticity(i, j) - vtemp(i, j)) *
                        (rmetdet(i, j, ie) * rrearth);
    }
  }
}

void divergence_sphere_c(int ie, const Vector_Field &v,
                         const Dvv &dvv,
                         const MetDet &metdet,
                         const MetDet &rmetdet,
                         const D &dinv,
                         Scalar_Field &divergence) {
  Homme_Local<real *> dvd("Positional Velocity Derivatives",
                          dim);
  Homme_Local<real ***> gv("Contravariant Form", np, np,
                           dim);
  Homme_Local<real **> vvtemp(
      "Divergence Performance Buffer", np, np);

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < dim; h++) {
        gv(i, j, h) = metdet(i, j, ie) *
                      (dinv(i, j, h, 0, ie) * v(i, j, 0) +
                       dinv(i, j, h, 1, ie) * v(i, j, 1));
      }
    }
  }

  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int h = 0; h < dim; h++) {
        dvd(h) = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dvd(0) += dvv(i, l) * gv(i, j, 0);
        dvd(1) += dvv(i, l) * gv(j, i, 1);
      }
      divergence(l, j) = dvd(0);
      vvtemp(j, l) = dvd(1);
    }
  }
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      divergence(i, j) = (divergence(i, j) + vvtemp(i, j)) *
                         (rmetdet(i, j, ie) * rrearth);
    }
  }
}

extern "C" {

/* TODO: Give this a better name */
void loop7_c(const int &nets, const int &nete,
             const int &n0, const int &nelemd,
             const int &tracer_advection_formulation,
             const real &pmean, const real &dtstage,
             real *&dvv_ptr, real *&d_ptr, real *&dinv_ptr,
             real *&metdet_ptr, real *&rmetdet_ptr,
             real *&fcor_ptr, real *&p_ptr, real *&ps_ptr,
             real *&v_ptr, real *&ptens_ptr,
             real *&vtens_ptr) {
  Dvv dvv(dvv_ptr, np, np);
  D d(d_ptr, np, np, dim, dim, nelemd);
  D dinv(dinv_ptr, np, np, dim, dim, nelemd);
  MetDet metdet(metdet_ptr, np, np, nelemd);
  MetDet rmetdet(rmetdet_ptr, np, np, nelemd);
  FCor fcor(fcor_ptr, np, np, nelemd);
  P p(p_ptr, np, np, nlev, timelevels, nelemd);
  PS ps(ps_ptr, np, np, nelemd);
  V v(v_ptr, np, np, dim, nlev, timelevels, nelemd);
  PTens ptens(ptens_ptr, np, np, nlev, nete - nets + 1);
  VTens vtens(vtens_ptr, np, np, dim, nlev,
              nete - nets + 1);

  Scalar_Field zeta("Vorticity", np, np);

  enum {
    TRACERADV_UGRADQ = 0,
    TRACERADV_TOTAL_DIVERGENCE = 1
  };

  for(int ie = nets - 1; ie < nete; ++ie) {
    for(int k = 0; k < nlev; ++k) {
      Vector_Field ulatlon("ulatlon", np, np, dim);
      Scalar_Field e("Energy", np, np);
      Vector_Field pv("PV", np, np, dim);

      for(int j = 0; j < np; ++j) {
        for(int i = 0; i < np; ++i) {
          real v1 = v(i, j, 0, k, n0 - 1, ie);
          real v2 = v(i, j, 1, k, n0 - 1, ie);
          e(i, j) = 0.0;
          for(int h = 0; h < dim; h++) {
            ulatlon(i, j, h) = d(i, j, h, 0, ie) * v1 +
                               d(i, j, h, 1, ie) * v2;
            pv(i, j, h) = ulatlon(i, j, h) *
                          (pmean + p(i, j, k, n0 - 1, ie));
            e(i, j) += ulatlon(i, j, h) * ulatlon(i, j, h);
          }
          e(i, j) /= 2.0;
          e(i, j) += p(i, j, k, n0 - 1, ie) + ps(i, j, ie);
        }
      }
      // Verified ulatlon, pv, and e up to this point

      // grade(np, np, dim)
      Vector_Field grade("Energy Gradient", np, np, dim);
      gradient_sphere_c(ie, e, dvv, dinv, grade);
      vorticity_sphere_c(ie, ulatlon, dvv, d, rmetdet,
                         zeta);
      // Verified grade and zeta up to this point

      Vector_Field gradh("Pressure Gradient", np, np, dim);
      Scalar_Field div("PV Divergence", np, np);
      if(tracer_advection_formulation == TRACERADV_UGRADQ) {
        auto p_slice = Kokkos::subview(
            p, std::make_pair(0, np), std::make_pair(0, np),
            k, n0 - 1, ie);
        gradient_sphere_c(ie, p_slice, dvv, dinv, gradh);
        for(int j = 0; j < np; j++) {
          for(int i = 0; i < np; i++) {
            div(i, j) = ulatlon(i, j, 0) * gradh(i, j, 0) +
                        ulatlon(i, j, 1) * gradh(i, j, 1);
          }
        }
      } else {
        divergence_sphere_c(ie, pv, dvv, metdet, rmetdet,
                            dinv, div);
      }
      // Verified gradh and div up to this point
      // ==============================================
      // Compute velocity
      // tendency terms
      // ==============================================
      // accumulate all RHS terms
      for(int j = 0; j < np; ++j) {
        for(int i = 0; i < np; ++i) {
          vtens(i, j, 0, k, ie - nets + 1) +=
              (ulatlon(i, j, 1) *
                   (fcor(i, j, ie) + zeta(i, j)) -
               grade(i, j, 0));

          vtens(i, j, 1, k, ie - nets + 1) +=
              (-ulatlon(i, j, 0) *
                   (fcor(i, j, ie) + zeta(i, j)) -
               grade(i, j, 1));

          ptens(i, j, k, ie - nets + 1) -= div(i, j);
        }
      }

      // ulatlon, pv, e, grade, zeta, gradh, div, ptens,
      // vtens are verified
      // fcor, dtstage are untouched
      // vtens, ulatlon, fcor, zeta, grade, ptens, div are
      // needed
      for(int j = 0; j < np; j++) {
        for(int i = 0; i < np; i++) {
          // take the local element timestep
          for(int h = 0; h < dim; h++) {
            vtens(i, j, h, k, ie - nets + 1) =
                ulatlon(i, j, h) +
                dtstage * vtens(i, j, h, k, ie - nets + 1);
          }
          ptens(i, j, k, ie - nets + 1) =
              p(i, j, k, n0 - 1, ie - nets + 1) +
              dtstage * ptens(i, j, k, ie - nets + 1);
        }
      }
    }
  }
}

}  // extern "C"

}  // Homme
