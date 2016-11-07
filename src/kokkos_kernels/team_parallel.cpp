
namespace Homme {

#include <Types.hpp>
#include <dimensions.hpp>
#include <kinds.hpp>

#include <cmath>

#include <string.h>

using Array = Kokkos::Array;

void tc1_velocity(int ie, int n0, int k, real u0,
                  SphereP &spherep, D &d, V &v) {
  constexpr const real alpha = 0.0;
  const real csalpha = std::cos(alpha);
  const real snalpha = std::sin(alpha);
  Array<Array<real, np>, np> snlon, cslon, snlat, cslat;
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      snlat[i][j] = std::sin(
          spherep(Spherical_Polar_e::Lat, i, j, ie));
      cslat[i][j] = std::cos(
          spherep(Spherical_Polar_e::Lat, i, j, ie));
      snlon[i][j] = std::sin(
          spherep(Spherical_Polar_e::Lon, i, j, ie));
      cslon[i][j] = std::cos(
          spherep(Spherical_Polar_e::Lon, i, j, ie));
    }
  }
  real V1, V2;
  for(int i = 0; i < np; i++) {
    for(int h = 0; h < np; h++) {
      V1 = u0 * (cslat[h][i] * csalpha +
                 snlat[h][i] * cslon[h][i] * snalpha);
      V2 = -u0 * (snlon[h][i] * snalpha);
      for(int g = 0; g < dim; g++) {
        v(h, i, g, k, n0, ie) =
            V1 * d(h, i, g, 0, ie) + V2 * d(h, i, g, 1, ie);
      }
    }
  }
}

void swirl_velocity(int ie, int n0, int k, real time,
                    SphereP &spherep, D &d, V &v) {
  /* TODO: Implement this */
}

void gradient_sphere(int ie, int e, const ScalarField &s,
                     const derivative &deriv, const D &dinv,
                     VectorField &grad) {
  Array<real, dim> dsd;
  Array<Array<Array<real, np>, np>, dim> v;
  for(int j = 0; j < np; j++) {
    for(int l = 0; l < np; l++) {
      for(int k = 0; k < dim; k++) {
        dsd[k] = 0.0;
      }
      for(int i = 0; i < np; i++) {
        dsd[0] += deriv.Dvv[i][l] * s[i][j];
      }
      v[0][l][j] = dsd[0] * rrearth;
      v[1][j][l] = dsd[1] * rrearth;
    }
  }
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int h = 0; h < np; h++) {
        grad(i, j, h) = dinv(i, j, 0, h, ie) * v[0][i][j] +
                        dinv(i, j, 1, h, ie) * v[1][i][j];
      }
    }
  }
}

void team_parallel_ex(const int &nets, const int &nete,
                      const int &n0, const int &nelemd,
                      const real &pmean, const real &u0,
                      const real &real_time,
                      const char *&topology,
                      const char *&test_case, real *&d_ptr,
                      real *&dinv_ptr, real *&fcor_ptr,
                      real *&spheremp_ptr, real *&p_ptr,
                      real *&ps_ptr, real *&v_ptr) {
  constexpr const unsigned dim = 2;

  D d(d_ptr, np, np, dim, dim, nelemd);
  D dinv(dinv_ptr, np, np, dim, dim, nelemd);
  FCor fcor(fcor_ptr, np, np, nelemd);
  SphereMP spheremp(np, np, nelemd);
  P p(p_ptr, np, np, nlev, timelevels, nelemd);
  PS ps(ps_ptr, np, np, nelemd);
  V v(v_ptr, np, np, dim, nlev, timelevels, nelemd);
  Zeta zeta(np, np);

  for(int ie = nets - 1; ie < nete; ie++) {
    if(strcmp(topology, "cube") == 0) {
      if(strcmp(test_case, "swtc1") == 0) {
        for(int k = 0; k < nlev; k++) {
          tc1_velocity(ie, n0, k, u0, spherep, d, v);
        }
      } else if(strcmp(test_case, "swirl") == 0) {
        for(int k = 0; k < nlev; k++) {
          swirl_velocity(ie, n0, k, time, spherep, dinv);
        }
      }
    }

    for(int k = 0; k < nlev; k++) {
      // ulatlon(np, np, dim), local variable
      VectorField ulatlon(np, np, dim);
      // E(np, np), local variable
      ScalarField e(np, np);
      // pv(np, np, dim)
      VectorField pv(np, np, dim);

      for(int j = 0; j < np; j++) {
        for(int i = 0; i < np; i++) {
          real v1 = v(i, j, 0, k, n0, ie);
          real v2 = v(i, j, 1, k, n0, ie);
          e(i, j) = 0.0;
          for(int h = 0; h < dim; h++) {
            ulatlon(i, j, h) =
                d(i, j, h, 0) * v1 + d(i, j, h, 1) * v2;
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
      VectorField grade(np, np, dim);
      // TODO: Implement gradient_sphere
      gradient_sphere(ie, e, deriv, dinv, grade);
      // TODO: Implement vorticity_sphere
      // TODO: Change the elem parameter to a type we can
      // actually use in C++
      vorticity_sphere(ulatlon, deriv, elem(ie), zeta);
      // latlon vector -> scalar
      // gradh(np, np, dim)
      VectorField gradh(np, np, dim);
      // div(np, np)
      Div div(np, np);
      if(tracer_advection_formulation == TRACERADV_UGRADQ) {
        gradient_sphere(ie, p, deriv, Dinv, gradh);
        for(int j = 0; j < np; j++) {
          for(int i = 0; i < np; i++) {
            div(i, j) = ulatlon(i, j, 0) * gradh(i, j, 0) +
                        ulatlon(i, j, 1) * gradh(i, j, 1);
          }
        }
      } else {
        // TODO: Implement
        // divergence_sphere
        // TODO: Change the
        // elem parameter to a
        // type we can
        // actually use in C++
        divergence_sphere(pv, deriv, elem, div);
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
          vtens(i, j, 0, k, ie) =
              vtens(i, j, 0, k, ie) +
              (ulatlon(i, j, 1) *
                   (fcor(i, j) + zeta(i, j)) -
               grade(i, j, 0));
          vtens(i, j, 1, k, ie) =
              vtens(i, j, 1, k, ie) +
              (-ulatlon(i, j, 0) *
                   (fcor(i, j) + zeta(i, j)) -
               grade(i, j, 1));
          // take the local element timestep
          for(int h = 0; h < dim; h++) {
            vtens(i, j, h, k, ie) =
                ulatlon(i, j, h) +
                dtstage * vtens(i, j, h, k, ie);
          }
          ptens(i, j, k, ie) +=
              dtstage * ptens(i, j, k, ie);
        }
      }
      if(limiter_option == 8 || limiter_option == 81 ||
         limiter_option == 84) {
        if(limiter_option == 81) {
          for(int i = 0; i < dim; i++) {
            pmax(i, ie) = pmax( :, ie) + 9e19;
          }
        } else if(limiter_option == 84) {
          for(int i = 0; i < dim; i++) {
            pmin(i, ie) = 0.0;
          }
          if(strcmp(test_case, 'swirl') == 0) {
            pmin(1, ie) = 0.1;
            if(nlev >= 3) {
              k = 3;
              pmin(k, ie) = 0.1;
            }
          }
          for(int i = 0; i < dim; i++) {
            pmax(i, ie) = pmax(i, ie) + 9e19;
          }
        }
        // TODO: Implement limiter_optim_wrap
        limiter_optim_wrap(kmass, ie, ptens, spheremp, pmin,
                           pmax);
      } else if(limiter_option == 4) {
        limiter2d_zero(kmass, ie, ptens, spheremp);
      }
      for(int k = 0; k < nlev; k++) {
        for(int j = 0; j < np; j++) {
          for(int i = 0; i < np; i++) {
            ptens(i, j, k, ie) =
                ptens(i, j, k, ie) * spheremp(i, j, ie);
            vtens(i, j, 0, k, ie) =
                vtens(i, j, 0, k, ie) * spheremp(i, j, ie);
            vtens(i, j, 1, k, ie) =
                vtens(i, j, 1, k, ie) * spheremp(i, j, ie);
          }
        }
      }
    }
  }
}

}  // Homme
