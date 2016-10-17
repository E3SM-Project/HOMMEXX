
#include <catch/catch.hpp>

#include <fortran_binding.hpp>

void divergence_sphere(const void *, const void *, const void *, void *);

extern void f90_divergence_sphere(const void *, const void *, const void *, void *) FORTRAN(divergence_sphere);

TEST_CASE("Divergence Test Case", "divergence_cxx") {
  f90_divergence_sphere(NULL, NULL, NULL, NULL);
}
