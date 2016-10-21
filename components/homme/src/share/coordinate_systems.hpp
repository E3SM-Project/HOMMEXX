
#ifndef _COORDINATE_SYSTEMS_HPP_
#define _COORDINATE_SYSTEMS_HPP_

constexpr const real DIST_THRESHOLD = 1.0e-9;
constexpr const real one = 1.0, two = 2.0;

struct cartesian2D_t {
  real x;
  real y;
};

struct cartesian3D_t {
  real x;
  real y;
  real z;
};

struct spherical_polar_t {
  real r;
  real lon;
  real lat;
};

#endif
