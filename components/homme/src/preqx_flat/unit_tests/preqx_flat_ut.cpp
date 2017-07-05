#include <catch/catch.hpp>

#include <limits>

#include <CaarControl.hpp>
#include <CaarFunctor.hpp>
#include <CaarRegion.hpp>
#include <Dimensions.hpp>
#include <Types.hpp>

#include <assert.h>
#include <stdio.h>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

void caar_compute_energy_grad_c_int(const Real (&dvv)[NP][NP], Real *Dinv,
                                    Real *const &pecnd, Real *const &phi,
                                    Real *const &velocity,
                                    Real (&vtemp)[2][NP][NP]);
}

Real compare_answers(Real target, Real computed, Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}

template <typename TestFunctor_T> class compute_subfunctor_test {
public:
  compute_subfunctor_test(int num_elems)
      : vector_results_1("Vector 1 output", num_elems),
        vector_results_2("Vector 2 output", num_elems),
        scalar_results_1("Scalar 1 output", num_elems),
        scalar_results_2("Scalar 2 output", num_elems),
        vector_output_1(Kokkos::create_mirror_view(vector_results_1)),
        vector_output_2(Kokkos::create_mirror_view(vector_results_2)),
        scalar_output_1(Kokkos::create_mirror_view(scalar_results_1)),
        scalar_output_2(Kokkos::create_mirror_view(scalar_results_2)),
        functor(), velocity("Velocity", num_elems),
        temperature("Temperature", num_elems), dp3d("DP3D", num_elems),
        phi("Phi", num_elems), pecnd("PE_CND", num_elems),
        omega_p("Omega_P", num_elems), derived_v("Derived V?", num_elems),
        eta_dpdn("Eta dot dp/deta", num_elems), qdp("QDP", num_elems),
        dinv("DInv", num_elems) {
    nets = 1;
    nete = num_elems;

    Real hybrid_a[NUM_LEV_P] = {0};
    functor.m_data.init(0, num_elems, num_elems, nm1, n0, np1, qn0, ps0, dt2,
                        false, eta_ave_w, hybrid_a);

    get_derivative().dvv(reinterpret_cast<Real *>(dvv));

    get_region().push_to_f90_pointers(velocity.data(), temperature.data(),
                                      dp3d.data(), phi.data(), pecnd.data(),
                                      omega_p.data(), derived_v.data(),
                                      eta_dpdn.data(), qdp.data());
    for (int ie = 0; ie < num_elems; ++ie) {
      get_region().dinv(Kokkos::subview(dinv, ie, Kokkos::ALL, Kokkos::ALL,
                                        Kokkos::ALL, Kokkos::ALL)
                            .data(),
                        ie);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    CaarFunctor::KernelVariables kv(team);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int &level) {
          kv.ilev = level;
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              for (int dim = 0; dim < 2; ++dim) {
                kv.vector_buf_1(dim, igp, jgp) = 0.0;
                kv.vector_buf_2(dim, igp, jgp) = 0.0;
              }
              kv.scalar_buf_1(igp, jgp) = 0.0;
              kv.scalar_buf_2(igp, jgp) = 0.0;
            }
          }
          TestFunctor_T::test_functor(functor, kv);
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              for (int dim = 0; dim < 2; ++dim) {
                vector_results_1(kv.ie, kv.ilev, dim, igp, jgp) =
                    kv.vector_buf_1(dim, igp, jgp);
                vector_results_2(kv.ie, kv.ilev, dim, igp, jgp) =
                    kv.vector_buf_2(dim, igp, jgp);
              }
              scalar_results_1(kv.ie, kv.ilev, igp, jgp) =
                  kv.scalar_buf_1(igp, jgp);
              scalar_results_2(kv.ie, kv.ilev, igp, jgp) =
                  kv.scalar_buf_2(igp, jgp);
            }
          }
        });
  }

  void run_functor() const {
    Kokkos::TeamPolicy<ExecSpace> policy(functor.m_data.num_elems, 1, 1);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(vector_output_1, vector_results_1);
    Kokkos::deep_copy(vector_output_2, vector_results_2);
    Kokkos::deep_copy(scalar_output_1, scalar_results_1);
    Kokkos::deep_copy(scalar_output_2, scalar_results_2);
  }

  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]> vector_results_1;
  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]> vector_results_2;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]> scalar_results_1;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]> scalar_results_2;

  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]>::HostMirror vector_output_1;
  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]>::HostMirror vector_output_2;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]>::HostMirror scalar_output_1;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]>::HostMirror scalar_output_2;

  CaarFunctor functor;

  // Arrays used to pass data to and from Fortran
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_LEV][2][NP][NP]> velocity;
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_LEV][NP][NP]> temperature;
  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_LEV][NP][NP]> dp3d;
  HostViewManaged<Real * [NUM_LEV][NP][NP]> phi;
  HostViewManaged<Real * [NUM_LEV][NP][NP]> pecnd;
  HostViewManaged<Real * [NUM_LEV][NP][NP]> omega_p;
  HostViewManaged<Real * [NUM_LEV][2][NP][NP]> derived_v;
  HostViewManaged<Real * [NUM_LEV_P][NP][NP]> eta_dpdn;
  HostViewManaged<Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP]> qdp;
  HostViewManaged<Real * [2][2][NP][NP]> dinv;

  Real dvv[NP][NP];

  static constexpr const int nm1 = 0;
  static constexpr const int n0 = 1;
  static constexpr const int np1 = 2;
  static constexpr const int qn0 = -1;
  static constexpr const int ps0 = 1;
  static constexpr const Real dt2 = 1.0;
  static constexpr const Real eta_ave_w = 1.0;

  int nets = -1;
  int nete = -1;
};

class compute_energy_grad_test {
public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor,
                           CaarFunctor::KernelVariables &kv) {
    functor.compute_energy_grad(kv);
  }
};

TEST_CASE("monolithic compute_and_apply_rhs", "compute_energy_grad") {
  constexpr const Real rel_threshold = 1E-15;
  constexpr const int num_elems = 10;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are initialized in the
  // singleton
  CaarRegion &region = get_region();
  region.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  compute_subfunctor_test<compute_energy_grad_test> test_functor(num_elems);

  test_functor.run_functor();

  Real vtemp[2][NP][NP];

  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_LEV; ++level) {
      Real(*const pressure)[NP] = reinterpret_cast<Real(*)[NP]>(
          region.get_3d_buffer(ie, CaarFunctor::PRESSURE, level).data());
      caar_compute_energy_grad_c_int(
          test_functor.dvv,
          Kokkos::subview(test_functor.dinv, ie, Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL)
              .data(),
          Kokkos::subview(test_functor.pecnd, ie, level, Kokkos::ALL,
                          Kokkos::ALL)
              .data(),
          Kokkos::subview(test_functor.phi, ie, level, Kokkos::ALL, Kokkos::ALL)
              .data(),
          Kokkos::subview(test_functor.velocity, ie, test_functor.n0, level,
                          Kokkos::ALL, Kokkos::ALL, Kokkos::ALL)
              .data(),
          vtemp);
      for (int dim = 0; dim < 2; ++dim) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            REQUIRE(!std::isnan(vtemp[dim][igp][jgp]));
            REQUIRE(!std::isnan(
                test_functor.vector_output_2(ie, level, dim, jgp, igp)));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        vtemp[dim][igp][jgp],
                        test_functor.vector_output_2(ie, level, dim, jgp, igp),
                        4.0));
          }
        }
      }
    }
  }
}
