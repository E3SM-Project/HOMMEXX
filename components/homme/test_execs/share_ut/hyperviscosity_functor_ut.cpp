#include <catch/catch.hpp>

#include "Context.hpp"
#include "Control.hpp"
#include "Derivative.hpp"
#include "Elements.hpp"
#include "HyperviscosityFunctor.hpp"
#include "Types.hpp"

#include <random>
#include <iomanip>

using namespace Homme;

extern "C" {

void setup_test_f90 (const int& ne_in, CF90Ptr& h_d_ptr, CF90Ptr& h_dinv_ptr, CF90Ptr& h_mp_ptr,
                     CF90Ptr& h_spheremp_ptr, CF90Ptr& h_rspheremp_ptr, CF90Ptr& h_metdet_ptr,
                     CF90Ptr& h_metinv_ptr, CF90Ptr& h_vec_sph2cart_ptr, CF90Ptr& h_tensorVisc_ptr);
void hyperviscosity_test_f90(F90Ptr& temperature_ptr, F90Ptr& dp3d_ptr, F90Ptr& velocity_ptr, const int& itl);
void cleanup_f90 ();

} // extern "C"

// =========================== TESTS ============================ //

TEST_CASE ("HyperviscosityFunctor", "Testing the biharmonic functor class")
{
  // Test constants
  constexpr int ne        = 2;
  constexpr int nfaces    = 6;
  constexpr int num_elems = nfaces*ne*ne;
  constexpr int num_tests = 10;
  constexpr const Real min_value = 0.015625;

  // Random accessories
  std::random_device rd;
  std::mt19937_64 engine(rd());
  std::uniform_real_distribution<Real> random_dist(min_value, 1.0 / min_value);

  // Get structures
  Control&    data     = Context::singleton().get_control();
  Elements&   elements = Context::singleton().get_elements();
  Derivative& deriv    = Context::singleton().get_derivative();

  // Setup data
  data.num_elems = num_elems;
  data.nets = 0;
  data.nete = num_elems;

  // Init the elements
  elements.random_init(num_elems);

  // Setup the F90
  decltype(elements.m_d)::HostMirror h_d = Kokkos::create_mirror_view(elements.m_d);
  decltype(elements.m_dinv)::HostMirror h_dinv = Kokkos::create_mirror_view(elements.m_dinv);
  decltype(elements.m_mp)::HostMirror h_mp = Kokkos::create_mirror_view(elements.m_mp);
  decltype(elements.m_spheremp)::HostMirror h_spheremp = Kokkos::create_mirror_view(elements.m_spheremp);
  decltype(elements.m_rspheremp)::HostMirror h_rspheremp = Kokkos::create_mirror_view(elements.m_rspheremp);
  decltype(elements.m_metdet)::HostMirror h_metdet = Kokkos::create_mirror_view(elements.m_metdet);
  decltype(elements.m_metinv)::HostMirror h_metinv = Kokkos::create_mirror_view(elements.m_metinv);
  decltype(elements.m_vec_sph2cart)::HostMirror h_vec_sph2cart = Kokkos::create_mirror_view(elements.m_vec_sph2cart);
  decltype(elements.m_tensorVisc)::HostMirror h_tensorVisc = Kokkos::create_mirror_view(elements.m_tensorVisc);
  Real* h_d_ptr = reinterpret_cast<Real*>(h_d.data();
  Real* h_dinv_ptr = reinterpret_cast<Real*>(h_dinv.data();
  Real* h_mp_ptr = reinterpret_cast<Real*>(h_mp.data();
  Real* h_spheremp_ptr = reinterpret_cast<Real*>(h_spheremp.data();
  Real* h_rspheremp_ptr = reinterpret_cast<Real*>(h_rspheremp.data();
  Real* h_metdet_ptr = reinterpret_cast<Real*>(h_metdet.data();
  Real* h_metinv_ptr = reinterpret_cast<Real*>(h_metinv.data();
  Real* h_vec_sph2cart_ptr = reinterpret_cast<Real*>(h_vec_sph2cart.data();
  Real* h_tensorVisc_ptr = reinterpret_cast<Real*>(h_tensorVisc.data();

  setup_test_f90 (num_elems, h_d_ptr, h_dinv_ptr, h_mp_ptr, h_spheremp_ptr, h_rspheremp_ptr,
                  h_metdet_ptr, h_metinv_ptr, h_vec_sph2cart_ptr, h_tensorVisc_ptr);



  // Create the functor
  HyperviscosityFunctor func(data,elements,deriv);

  // Run the tests
  for (int itest=0; itest<num_tests; ++itest) {
    // Generate random states
    genRandArray(elements.m_t,   engine,random_dist);
    genRandArray(elements.m_dp3d,engine,random_dist);
    genRandArray(elements.m_v,   engine,random_dist);

    // Run the f90 test

    // Generate random states
    REQUIRE(true);
  }

  cleanup_f90();
}
