
#include <Types.hpp>

#include <dimensions.hpp>
#include <kinds.hpp>

#include <cmath>

namespace Homme {

constexpr const real rearth = 6.376E6;
constexpr const real rrearth = 1.0 / rearth;

template <typename T, int size>
using Array = Kokkos::Array<T, size>;

template <typename ScalarQP>
void gradient_sphere(int ie, const ScalarQP &s,
                     const derivative &deriv, const D &dinv,
                     VectorField &grad) {
  HommeLocal<real *> dsd("Velocity Spatial Derivatives",
                         dim);
  HommeLocal<real ***> v("Velocity", np, np, dim);
  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int k = 0; k < dim; k++) {
        dsd(k) = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dsd(0) += deriv.Dvv[i][l] * s(i, j);
        dsd(1) += deriv.Dvv[i][l] * s(j, i);
      }
      v(l, j, 0) = dsd(0) * rrearth;
      v(j, l, 1) = dsd(1) * rrearth;
    }
  }
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < np; h++) {
        grad(i, j, h) = dinv(i, j, 0, h, ie) * v(i, j, 0) +
                        dinv(i, j, 1, h, ie) * v(i, j, 1);
      }
    }
  }
}

void vorticity_sphere(int ie, const VectorField &v,
                      const derivative &deriv, const D &d,
                      const MetDet &rmetdet,
                      ScalarField &vorticity) {
  Array<real, dim> dvd;
  Array<Array<Array<real, np>, np>, dim> vco;
  Array<Array<real, np>, np> vtemp;

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < dim; h++) {
        vco[h][j][i] = d(i, j, 0, h, ie) * v(i, j, 0) +
                       d(i, j, 1, h, ie) * v(i, j, 1);
      }
    }
  }

  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int h = 0; h < dim; h++) {
        dvd[h] = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dvd[0] += deriv.Dvv[l][i] * vco[1][j][i];
        dvd[1] += deriv.Dvv[l][i] * vco[0][i][j];
      }
      vorticity(l, j) = dvd[0];
      vorticity(j, l) = dvd[0];
    }
  }

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      vorticity(i, j) = (vorticity(i, j) - vtemp[j][i]) *
                        (rmetdet(i, j, ie) * rrearth);
    }
  }
}

void divergence_sphere(int ie, const VectorField &v,
                       const derivative &deriv,
                       const MetDet &metdet,
                       const MetDet &rmetdet, const D &dinv,
                       ScalarField &divergence) {
  HommeLocal<real *> dvd("Positional Velocity Derivatives",
                         dim);
  HommeLocal<real ***> gv("Contravariant Form", np, np,
                          dim);
  HommeLocal<real **> vvtemp(
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
        dvd[h] = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dvd[0] += deriv.Dvv[l][i] * gv(i, j, 0);
        dvd[1] += deriv.Dvv[l][i] * gv(j, i, 1);
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

void team_parallel_ex(
    const int &nets, const int &nete, const int &n0,
    const int &nelemd,
    const int &tracer_advection_formulation,
    const real &pmean, const derivative &deriv,
    const real &dtstage, real *&d_ptr, real *&dinv_ptr,
    real *&metdet_ptr, real *&rmetdet_ptr, real *&fcor_ptr,
    real *&p_ptr, real *&ps_ptr, real *&v_ptr,
    real *&ptens_ptr, real *&vtens_ptr) {
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

  ScalarField zeta("Vorticity", np, np);

  enum {
    TRACERADV_UGRADQ = 0,
    TRACERADV_TOTAL_DIVERGENCE = 1
  };

  for(int ie = nets - 1; ie < nete; ie++) {
    for(int k = 0; k < nlev; k++) {
      // ulatlon(np, np, dim), local variable
      VectorField ulatlon("ulatlon", np, np, dim);
      // E(np, np), local variable
      ScalarField e("Energy", np, np);
      // pv(np, np, dim)
      VectorField pv("PV", np, np, dim);

      for(int j = 0; j < np; j++) {
        for(int i = 0; i < np; i++) {
          real v1 = v(i, j, 0, k, n0, ie);
          real v2 = v(i, j, 1, k, n0, ie);
          e(i, j) = 0.0;
          for(int h = 0; h < dim; h++) {
            ulatlon(i, j, h) = d(i, j, h, 0, ie) * v1 +
                               d(i, j, h, 1, ie) * v2;
            e(i, j) += ulatlon(i, j, h) * ulatlon(i, j, h);
          }
          e(i, j) /= 2.0;
          e(i, j) += p(i, j, k, n0, ie) + ps(i, j, ie);
          for(int h = 0; h < dim; h++) {
            pv(i, j, h) = ulatlon(i, j, h) *
                          (pmean + p(i, j, k, n0, ie));
          }
        }
      }

      // grade(np, np, dim)
      VectorField grade("Energy Gradient", np, np, dim);
      // TODO: Implement gradient_sphere
      gradient_sphere(ie, e, deriv, dinv, grade);
      vorticity_sphere(ie, ulatlon, deriv, d, rmetdet,
                       zeta);
      // latlon vector -> scalar
      // gradh(np, np, dim)
      VectorField gradh("Pressure Gradient", np, np, dim);
      // div(np, np)
      ScalarField div("PV Divergence", np, np);
      if(tracer_advection_formulation == TRACERADV_UGRADQ) {
        auto p_slice = Kokkos::subview(
            p, std::make_pair(0, np), std::make_pair(0, np),
            k, n0, ie);
        gradient_sphere(ie, p_slice, deriv, dinv, gradh);
        for(int j = 0; j < np; j++) {
          for(int i = 0; i < np; i++) {
            div(i, j) = ulatlon(i, j, 0) * gradh(i, j, 0) +
                        ulatlon(i, j, 1) * gradh(i, j, 1);
          }
        }
      } else {
        divergence_sphere(ie, pv, deriv, metdet, rmetdet,
                          dinv, div);
      }

      // ==============================================
      // Compute velocity
      // tendency terms
      // ==============================================
      // accumulate all RHS terms
      // vtens(np, np, dim, nlev, nelemd)
      // ptens(np, np, nlev, nelemd)
      for(int j = 0; j < np; j++) {
        for(int i = 0; i < np; i++) {
          vtens(i, j, 0, k, ie - nets + 1) =
              vtens(i, j, 0, k, ie - nets + 1) +
              (ulatlon(i, j, 1) *
                   (fcor(i, j, ie) + zeta(i, j)) -
               grade(i, j, 0));
          vtens(i, j, 1, k, ie - nets + 1) =
              vtens(i, j, 1, k, ie - nets + 1) +
              (-ulatlon(i, j, 0) *
                   (fcor(i, j, ie) + zeta(i, j)) -
               grade(i, j, 1));
          // take the local element timestep
          for(int h = 0; h < dim; h++) {
            vtens(i, j, h, k, ie - nets + 1) =
                ulatlon(i, j, h) +
                dtstage * vtens(i, j, h, k, ie - nets + 1);
          }
          ptens(i, j, k, ie) +=
              dtstage * ptens(i, j, k, ie);
        }
      }
    }
  }
}

}  // Homme
