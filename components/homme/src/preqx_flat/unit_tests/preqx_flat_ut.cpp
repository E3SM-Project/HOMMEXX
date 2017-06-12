#include <catch/catch.hpp>

#include <CaarControl.hpp>
#include <CaarFunctor.hpp>
#include <CaarRegion.hpp>
#include <Dimensions.hpp>
#include <Types.hpp>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

void caar_compute_energy_grad_c_int(
    const int &n0, const int &level, Real((*const &dvv)[NP]), const Real *&Dinv,
    Real((*const &pecnd)[NP][NP]), Real((*const &temperature)[NP][NP]),
    const Real *&pressure, Real((*const &phi)[NP][NP]),
    Real((*const &velocity)[NUM_LEV][2][NP][NP]), Real (*&vtemp)[NP][NP]);
}

Real compare_answers(Real target, Real computed, Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}

class compute_energy_grad_test {
public:
  compute_energy_grad_test(int num_elems, rngAlg &engine)
      : results("Kokkos results", num_elems), control(), functor(control),
        energy_grad("Energy gradient", num_elems) {
    using udi_type = std::uniform_int_distribution<int>;

    nets = udi_type(0, num_elems - 1)(engine);
    nete = udi_type(nets + 1, num_elems)(engine);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    CaarFunctor::KernelVariables kv(team);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int &level) {
                           kv.ilev = level;
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
    Kokkos::deep_copy(Kokkos::subview(results, kv.ie, Kokkos::ALL, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL),
                      Kokkos::subview(energy_grad, kv.ie, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }

  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]>::HostMirror results;

  CaarControl control;
  CaarFunctor functor;

  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]> energy_grad;

  static constexpr const int num_elems = 10;
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
  constexpr const int num_elems = 10;

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

  get_region().random_init(num_elems, engine);
  get_derivative().random_init(engine);

  get_derivative().dvv(reinterpret_cast<Real *>(dvv));

  compute_energy_grad_test test_functor(10, engine);

  Kokkos::TeamPolicy<ExecSpace> policy(num_elems, 8, 1);
  Kokkos::parallel_for(policy, test_functor);

  CaarRegion region = get_region();
  region.push_to_f90_pointers(
      reinterpret_cast<Real *>(velocity), reinterpret_cast<Real *>(temperature),
      reinterpret_cast<Real *>(dp3d), reinterpret_cast<Real *>(phi),
      reinterpret_cast<Real *>(pecnd), reinterpret_cast<Real *>(omega_p),
      reinterpret_cast<Real *>(derived_v), reinterpret_cast<Real *>(eta_dpdn),
      reinterpret_cast<Real *>(qdp));

  for (int ie = 0; ie < num_elems; ie++) {
    const Real *Dinv = region.DINV(ie).data();
    for (int level = 0; level < NUM_LEV; level++) {
      const Real *pressure =
          region.get_3d_buffer(ie, CaarFunctor::PRESSURE, level).data();
      caar_compute_energy_grad_c_int(test_functor.n0, level, dvv, Dinv,
                                     pecnd[ie],
                                     temperature[ie][test_functor.n0], pressure,
                                     phi[ie], velocity[ie], vtemp);
      for (int dim = 0; dim < 2; ++dim) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            REQUIRE(vtemp[dim][igp][jgp] ==
                    test_functor.results(ie, level, dim, igp, jgp));
          }
        }
      }
    }
  }
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


// dvv:         0xbe0f630
// dinv:        0xbe11280
// pecnd:       0xbdee7f0
// pressure:    0xbe55880
// temperature: 0xbddcf30 *** Different from Fortran
// phi:         0xbdebfb0
// velocity:    0xbdcdef0
// vtemp:       0xbe0f6f0



// dvv         199292464
// dinv        199299712
// pecnd       199157744
// pressure    199579776
// temperature 199086896
// phi         199147440
// velocity    199024368 
// vtemp       199292656
