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

void caar_compute_energy_grad_c_int(Real((*const &dvv)[NP]), const Real *&Dinv,
                                    Real((*const &pecnd)[NP]),
                                    Real((*const &phi)[NP]),
                                    Real((*const &velocity)[NP][NP]),
                                    Real (*&vtemp)[NP][NP]);
}

Real compare_answers(Real target, Real computed, Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}

template <typename Results_T>
class compute_subfunctor_test {
public:
  using SubFunctor_T = void (CaarFunctor::*)(CaarFunctor::KernelVariables &);

  compute_subfunctor_test(int num_elems, rngAlg &engine, SubFunctor_T subfunctor)
      : results("Kokkos results", num_elems), functor(),
        energy_grad("Energy gradient", num_elems) {
    using udi_type = std::uniform_int_distribution<int>;

    nets = 1;
    nete = num_elems;

    Real hybrid_a[NUM_LEV_P] = {0};
    functor.m_data.init(0, num_elems, num_elems, nm1, n0, np1, qn0, ps0, dt2,
                        false, eta_ave_w, hybrid_a);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    CaarFunctor::KernelVariables kv(team);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int &level) {
                           kv.ilev = level;
                           for (int dim = 0; dim < 2; ++dim) {
                             for (int igp = 0; igp < NP; ++igp) {
                               for (int jgp = 0; jgp < NP; ++jgp) {
                                 kv.vector_buf_2(dim, igp, jgp) = 0.0;
                               }
                             }
                           }
                           functor.compute_energy_grad(kv);
                           for (int dim = 0; dim < 2; ++dim) {
                             for (int igp = 0; igp < NP; ++igp) {
                               for (int jgp = 0; jgp < NP; ++jgp) {
                                 energy_grad(kv.ie, kv.ilev, dim, igp, jgp) =
                                     kv.vector_buf_2(dim, igp, jgp);
                               }
                             }
                           }
                         });
  }

  void run_functor() const {
    Kokkos::TeamPolicy<ExecSpace> policy(functor.m_data.num_elems, 1, 1);
    Kokkos::parallel_for(policy, *this);
    Kokkos::deep_copy(results, energy_grad);
  }

  ExecViewManaged<Results_T>::HostMirror results;

  CaarFunctor functor;

  ExecViewManaged<Results_T> energy_grad;

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

TEST_CASE("monolithic compute_and_apply_rhs", "compute_energy_grad") {
  constexpr const Real rel_threshold = 1E-15;
  constexpr const int num_elems = 1;

  std::random_device rd;
  rngAlg engine(rd());

  Real(*velocity)[NUM_TIME_LEVELS][NUM_LEV][2][NP][NP] =
      new Real[num_elems][NUM_TIME_LEVELS][NUM_LEV][2][NP][NP];
  Real(*temperature)[NUM_TIME_LEVELS][NUM_LEV][NP][NP] =
      new Real[num_elems][NUM_TIME_LEVELS][NUM_LEV][NP][NP];
  Real(*dp3d)[NUM_TIME_LEVELS][NUM_LEV][NP][NP] =
      new Real[num_elems][NUM_TIME_LEVELS][NUM_LEV][NP][NP];
  Real(*phi)[NUM_LEV][NP][NP] = new Real[num_elems][NUM_LEV][NP][NP];
  Real(*pecnd)[NUM_LEV][NP][NP] = new Real[num_elems][NUM_LEV][NP][NP];
  Real(*omega_p)[NUM_LEV][NP][NP] = new Real[num_elems][NUM_LEV][NP][NP];
  Real(*derived_v)[NUM_LEV][2][NP][NP] =
      new Real[num_elems][NUM_LEV][2][NP][NP];
  Real(*eta_dpdn)[NUM_LEV_P][NP][NP] = new Real[num_elems][NUM_LEV_P][NP][NP];
  Real(*qdp)[Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP] =
      new Real[num_elems][Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP];
  Real(*dvv)[NP] = new Real[NP][NP];
  Real(*vtemp)[NP][NP] = new Real[2][NP][NP];

  // This must be a reference to ensure the views are initialized in the
  // singleton
  CaarRegion &region = get_region();
  region.random_init(num_elems, engine);
  get_derivative().random_init(engine);
  get_derivative().dvv(reinterpret_cast<Real *>(dvv));

  region.push_to_f90_pointers(
      reinterpret_cast<Real *>(velocity), reinterpret_cast<Real *>(temperature),
      reinterpret_cast<Real *>(dp3d), reinterpret_cast<Real *>(phi),
      reinterpret_cast<Real *>(pecnd), reinterpret_cast<Real *>(omega_p),
      reinterpret_cast<Real *>(derived_v), reinterpret_cast<Real *>(eta_dpdn),
      reinterpret_cast<Real *>(qdp));

  compute_energy_grad_test test_functor(num_elems, engine);

  test_functor.run_functor();

  Real *dinv = reinterpret_cast<Real *>(new Real[2][2][NP][NP]);
  const Real *const_dinv = dinv;
  for (int ie = 0; ie < num_elems; ++ie) {
    region.dinv(dinv, ie);
    for (int level = 0; level < NUM_LEV; ++level) {
      Real(*const pressure)[NP] = reinterpret_cast<Real(*)[NP]>(
          region.get_3d_buffer(ie, CaarFunctor::PRESSURE, level).data());
      caar_compute_energy_grad_c_int(
          dvv, const_dinv, pecnd[ie][level], phi[ie][level],
          velocity[ie][test_functor.n0][level], vtemp);
      for (int dim = 0; dim < 2; ++dim) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        vtemp[dim][igp][jgp],
                        test_functor.results(ie, level, dim, jgp, igp), 4.0));
          }
        }
      }
    }
  }
  delete[] dinv;

  delete[] temperature;
  delete[] dp3d;
  delete[] phi;
  delete[] pecnd;
  delete[] omega_p;
  delete[] derived_v;
  delete[] eta_dpdn;
  delete[] qdp;
  delete[] dvv;
}
