#include <catch/catch.hpp>

#include <limits>

#include "CaarControl.hpp"
#include "CaarFunctor.hpp"
#include "CaarRegion.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"

#include <assert.h>
#include <stdio.h>
#include <random>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

void caar_compute_energy_grad_c_int(
    const Real *const &dvv, const Real *const &Dinv,
    const Real *const &pecnd, const Real *const &phi,
    const Real *const &velocity,
    Real *const &vtemp);  //(&vtemp)[2][NP][NP]);

// inserting names of params for _c_callable functions.
// instead of using names as in function descriptions, using
// verbose names
void laplace_simple_c_callable(const Real *input,
                          const Real *dvv, const Real *dinv,
                          const Real *metdet, Real *output);

void gradient_sphere_c_callable(const Real *input,
                                const Real *dvv,
                                const Real *dinv,
                                Real *output);

void divergence_sphere_wk_c_callable(const Real *input,
                                     const Real *dvv,
                                     const Real *spheremp,
                                     const Real *dinv,
                                     Real *output);

//This is an interface for the most general 2d laplace in F. In c++ we will ignore
//'var coef' HV. If one uses var_coef in below, then both 'var_coef' and 'tensor' HV
//are possible. the switch between them is whether hvpower is <>0. To exclude
//'var coef' HV we will always set hvpower to 0. If hvscaling is <>0, then tensor is used.
//So, settings for usual HV are var_coef=FALSE.
//Settings for tensor HV are var_coef=TRUE, hvpower=0, hvscaling >0.
//(don't ask me why this is so).
//Update, copypasting from stackoverflow: 
//Typical Fortran implementations pass all arguments by reference, 
//while in C(++) the default is by value. This fixes an issue with hvpower, etc.
void laplace_sphere_wk_c_callable(const Real * input,
                                  const Real * dvv,
                                  const Real * dinv,
                                  const Real * spheremp,
                                  const Real * tensorVisc,
                                  const Real &hvpower,//should be set to 0 always
                                  const Real &hvscaling,//should be set to !=0 value
                                  const bool &var_coef,//should be set to 1 for tensor HV
                                  Real * output);

void curl_sphere_wk_testcov_c_callable(const Real * input, //s(np,np)
                                       const Real * dvv,// dvv(np, np)
                                       const Real * D,//D(np, np, 2, 2)
                                       const Real * mp,//mp(np, np)
                                       Real * output); //ds(np,np,2)


void gradient_sphere_wk_testcov_c_callable(const Real * input, //s(np,np)
                                       const Real * dvv,// dvv(np, np)
                                       const Real * metinv,//metinv(np, np, 2, 2)
                                       const Real * metdet,//metdet(np, np)
                                       const Real * D,//D(np, np, 2, 2)
                                       const Real * mp,//mp(np, np)
                                       Real * output); //ds(np,np,2)

}  // extern C

Real compare_answers(Real target, Real computed,
                     Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if(relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }
  return std::fabs(target - computed) / denom;
}  // end of definition of compare_answers()

void genRandArray(
    Real *arr, int arr_len, rngAlg &engine,
    std::uniform_real_distribution<Real> pdf) {
  for(int i = 0; i < arr_len; ++i) {
    arr[i] = pdf(engine);
  }
}  // end of definition of genRandArray()

/* compute_subfunctor_test
 *
 * Randomly initializes all of the input data
 * Calls the static method test_functor in TestFunctor_T,
 * passing in a correctly initialized KernelVariables object
 *
 * Ideally to use this structure you won't touch anything in
 * it
 */
template <typename TestFunctor_T>
class compute_subfunctor_test {
 public:
  compute_subfunctor_test(int num_elems)
      : functor(),
        velocity("Velocity", num_elems),
        temperature("Temperature", num_elems),
        dp3d("DP3D", num_elems),
        phi("Phi", num_elems),
        pecnd("PE_CND", num_elems),
        omega_p("Omega_P", num_elems),
        derived_v("Derived V?", num_elems),
        eta_dpdn("Eta dot dp/deta", num_elems),
        qdp("QDP", num_elems),
        dinv("DInv", num_elems),
        dvv("dvv"),
        nets(1),
        nete(num_elems) {
    Real hybrid_a[NUM_LEV_P] = {0};
    functor.m_data.init(0, num_elems, num_elems, nm1, n0,
                        np1, qn0, ps0, dt2, false,
                        eta_ave_w, hybrid_a);

    get_derivative().dvv(dvv.data());

    get_region().push_to_f90_pointers(
        velocity.data(), temperature.data(), dp3d.data(),
        phi.data(), pecnd.data(), omega_p.data(),
        derived_v.data(), eta_dpdn.data(), qdp.data());
    for(int ie = 0; ie < num_elems; ++ie) {
      get_region().dinv(
          Kokkos::subview(dinv, ie, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL)
              .data(),
          ie);
    }
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    return KernelVariables::shmem_size(team_size);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    KernelVariables kv(team);
    TestFunctor_T::test_functor(functor, kv);
  }

  void run_functor() const {
    Kokkos::TeamPolicy<ExecSpace> policy(
        functor.m_data.num_elems, 16, 4);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
  }

  CaarFunctor functor;

  // host
  // Arrays used to pass data to and from Fortran
  HostViewManaged<
      Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]>
      velocity;
  HostViewManaged<
      Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>
      temperature;
  HostViewManaged<
      Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>
      dp3d;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> phi;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> pecnd;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]>
      omega_p;
  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]>
      derived_v;
  HostViewManaged<Real * [NUM_INTERFACE_LEV][NP][NP]>
      eta_dpdn;
  HostViewManaged<Real *
                  [Q_NUM_TIME_LEVELS]
                      [QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>
      qdp;
  HostViewManaged<Real * [2][2][NP][NP]> dinv;
  HostViewManaged<Real[NP][NP]> dvv;

  const int nets;
  const int nete;

  static constexpr const int nm1 = 0;
  static constexpr const int n0 = 1;
  static constexpr const int np1 = 2;
  static constexpr const int qn0 = -1;
  static constexpr const int ps0 = 1;
  static constexpr const Real dt2 = 1.0;
  static constexpr const Real eta_ave_w = 1.0;
};

class compute_energy_grad_test {
 public:
  KOKKOS_INLINE_FUNCTION
  static void test_functor(const CaarFunctor &functor,
                           KernelVariables &kv) {
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          functor.compute_energy_grad(kv);
        });
  }
};

TEST_CASE("monolithic compute_and_apply_rhs",
          "compute_energy_grad") {
  printf(
      "Q_NUM_TIME_LEVELS: %d\n"
      "QSIZE_D: %d\n"
      "NUM_PHYSICAL_LEV: %d\n"
      "VECTOR_SIZE: %d\n"
      "LEVEL_PADDING: %d\n"
      "NUM_LEV: %d\n",
      Q_NUM_TIME_LEVELS, QSIZE_D, NUM_PHYSICAL_LEV,
      VECTOR_SIZE, LEVEL_PADDING, NUM_LEV);
  constexpr const Real rel_threshold = 1E-15;
  constexpr const int num_elems = 1;

  std::random_device rd;
  rngAlg engine(rd());

  // This must be a reference to ensure the views are
  // initialized in the singleton
  CaarRegion &region = get_region();
  region.random_init(num_elems, engine);
  get_derivative().random_init(engine);

  compute_subfunctor_test<compute_energy_grad_test>
      test_functor(num_elems);
  test_functor.run_functor();
  HostViewManaged<Scalar * [2][NP][NP][NUM_LEV]>
      energy_grad("energy_grad", num_elems);
  Kokkos::deep_copy(energy_grad,
                    region.buffers.energy_grad);

  for(int ie = 0; ie < num_elems; ++ie) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real vtemp[2][NP][NP];

        caar_compute_energy_grad_c_int(
            reinterpret_cast<Real *>(
                test_functor.dvv.data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.dinv, ie,
                                Kokkos::ALL, Kokkos::ALL,
                                Kokkos::ALL, Kokkos::ALL)
                    .data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.pecnd, ie,
                                level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL)
                    .data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.phi, ie,
                                level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL)
                    .data()),
            reinterpret_cast<Real *>(
                Kokkos::subview(test_functor.velocity, ie,
                                test_functor.n0,
                                level * VECTOR_SIZE + v,
                                Kokkos::ALL, Kokkos::ALL,
                                Kokkos::ALL)
                    .data()),
            &vtemp[0][0][0]);
        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            REQUIRE(!std::isnan(vtemp[0][igp][jgp]));
            REQUIRE(!std::isnan(vtemp[1][igp][jgp]));
            REQUIRE(!std::isnan(
                energy_grad(ie, 0, igp, jgp, level)[v]));
            REQUIRE(!std::isnan(
                energy_grad(ie, 1, igp, jgp, level)[v]));
            REQUIRE(
                std::numeric_limits<Real>::epsilon() >=
                compare_answers(
                    vtemp[0][igp][jgp],
                    energy_grad(ie, 0, igp, jgp, level)[v],
                    128.0));
            REQUIRE(
                std::numeric_limits<Real>::epsilon() >=
                compare_answers(
                    vtemp[1][igp][jgp],
                    energy_grad(ie, 1, igp, jgp, level)[v],
                    128.0));
          }
        }
      }
    }
  }
  std::cout << "test finished.\n";
};  // end of TEST_CASE(...,"compute_energy_grad")

class compute_sphere_operator_test_ml {
 public:
  compute_sphere_operator_test_ml(int num_elems)
      : scalar_input_d("scalar input", num_elems),
        vector_input_d("vector input", num_elems),
        d_d("d", num_elems),
        dinv_d("dinv", num_elems),
        metinv_d("metinv", num_elems),
        metdet_d("metdet", num_elems),
        spheremp_d("spheremp", num_elems),
        mp_d("mp", num_elems), 
        dvv_d("dvv"),
        tensor_d("tensor", num_elems),
        scalar_output_d("scalar output", num_elems),
        vector_output_d("vector output", num_elems),
        temp1_d("temp1", num_elems),
        temp2_d("temp2", num_elems),
        temp3_d("temp3", num_elems),
        temp4_d("temp4"),
        scalar_input_host(
            Kokkos::create_mirror_view(scalar_input_d)),
        vector_input_host(
            Kokkos::create_mirror_view(vector_input_d)),
        d_host(Kokkos::create_mirror_view(d_d)),
        dinv_host(Kokkos::create_mirror_view(dinv_d)),
        metinv_host(Kokkos::create_mirror_view(metinv_d)),
        metdet_host(Kokkos::create_mirror_view(metdet_d)),
        spheremp_host(
            Kokkos::create_mirror_view(spheremp_d)),
        mp_host(
            Kokkos::create_mirror_view(mp_d)),
        dvv_host(Kokkos::create_mirror_view(dvv_d)),
        tensor_host(Kokkos::create_mirror_view(tensor_d)),
        scalar_output_host(
            Kokkos::create_mirror_view(scalar_output_d)),
        vector_output_host(
            Kokkos::create_mirror_view(vector_output_d)),
        temp1_host(Kokkos::create_mirror_view(temp1_d)),
        temp2_host(Kokkos::create_mirror_view(temp2_d)),
        temp3_host(Kokkos::create_mirror_view(temp3_d)),
        _num_elems(num_elems) {
    std::random_device rd;
    rngAlg engine(rd());
    genRandArray(
        reinterpret_cast<Real *>(scalar_input_host.data()),
        scalar_input_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-1000.0, 1000.0));
    genRandArray(
        reinterpret_cast<Real *>(vector_input_host.data()),
        vector_input_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-1000.0,
                                             1000.0));
    genRandArray(
        d_host.data(), d_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-100.0, 100.0));
    genRandArray(
        dinv_host.data(), dinv_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-100.0, 100.0));
    genRandArray(
        metinv_host.data(), metinv_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-100.0, 100.0));
    genRandArray(
        metdet_host.data(), metdet_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-100.0, 100.0));
    genRandArray(
        spheremp_host.data(), spheremp_len * _num_elems,
        engine,
        std::uniform_real_distribution<Real>(-100.0, 100.0));
    genRandArray(
        mp_host.data(), mp_len * _num_elems,
        engine,
        std::uniform_real_distribution<Real>(-100.0, 100.0));
    genRandArray(
        dvv_host.data(), dvv_len, engine,
        std::uniform_real_distribution<Real>(-100.0, 100.0));
    genRandArray(
        tensor_host.data(), tensor_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-1000, 1000.0));


//setting everything to 1 is good for debugging
#if 1
for(int i1=0; i1<_num_elems; i1++)
for(int i2=0; i2<NP; i2++)
for(int i3=0; i3<NP; i3++){
//d_host(i1,0,0,i2,i3)=1.0;
//d_host(i1,1,1,i2,i3)=1.0;
//d_host(i1,1,0,i2,i3)=1.0;
//d_host(i1,0,1,i2,i3)=1.0;
//metdet_host(i1,i2,i3)=1.0;
//spheremp_host(i1,i2,i3)=1.0;
//dvv_host(i2,i3)=1.0;
//mp_host(i1,i2,i3)=1.0;
//           -//Real aa = i2+i3;
//            -//scalar_input_host(i1,i2,i3) = aa;
//             -//vector_input_host(i1,0,i2,i3) = aa;
//              -//vector_input_host(i1,1,i2,i3) = aa;
}
#endif


  }  // end of constructor

  int _num_elems;  // league size, serves as ie index

  // device
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>
      scalar_input_d;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>
      vector_input_d;
  ExecViewManaged<Real * [2][2][NP][NP]> d_d;
  ExecViewManaged<Real * [2][2][NP][NP]> dinv_d;
  ExecViewManaged<Real * [2][2][NP][NP]> metinv_d;
  ExecViewManaged<Real * [NP][NP]> spheremp_d;
  ExecViewManaged<Real * [NP][NP]> mp_d;
  ExecViewManaged<Real * [NP][NP]> metdet_d;
  ExecViewManaged<Real[NP][NP]> dvv_d;
  ExecViewManaged<Real * [2][2][NP][NP]> tensor_d;
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>
      scalar_output_d;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>
      vector_output_d;
  // making temp vars with leading dimension 'ie' to avoid
  // thread sharing issues  in the ie loop
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]> temp1_d,
      temp2_d, temp3_d;

  ExecViewManaged<Scalar [NP][NP][NUM_LEV]>
      temp4_d;

  // host
  // rely on fact NUM_PHYSICAL_LEV=NUM_LEV*VECTOR_SIZE
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror
      scalar_input_host;
  const int scalar_input_len =
      NUM_PHYSICAL_LEV * NP * NP;  // temp code

  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>::HostMirror
      vector_input_host;
  const int vector_input_len =
      NUM_PHYSICAL_LEV * 2 * NP * NP;

  ExecViewManaged<Real * [2][2][NP][NP]>::HostMirror d_host;
  const int d_len = 2 * 2 * NP * NP;  // temp code

  ExecViewManaged<Real * [2][2][NP][NP]>::HostMirror
      dinv_host;
  const int dinv_len = 2 * 2 * NP * NP;  // temp code

  ExecViewManaged<Real * [2][2][NP][NP]>::HostMirror
      metinv_host;
  const int metinv_len = 2 * 2 * NP * NP;  // temp code

  ExecViewManaged<Real * [NP][NP]>::HostMirror metdet_host;
  const int metdet_len = NP * NP;

  ExecViewManaged<Real * [NP][NP]>::HostMirror
      spheremp_host;
  const int spheremp_len = NP * NP;

  ExecViewManaged<Real * [NP][NP]>::HostMirror
      mp_host;
  const int mp_len = NP * NP;

  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_host;
  const int dvv_len = NP * NP;

  ExecViewManaged<Real * [2][2][NP][NP]>::HostMirror
      tensor_host;
  const int tensor_len = 2 * 2 * NP * NP;  // temp code

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror
      scalar_output_host;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>::HostMirror
      vector_output_host;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>::HostMirror
      temp1_host,
      temp2_host, temp3_host;

  // tag for laplace_simple()
  struct TagSimpleLaplaceML {};
  // tag for gradient_sphere()
  struct TagGradientSphereML {};
  // tag for divergence_sphere_wk
  struct TagDivergenceSphereWkML {};
  // tag for laplace_tensor
  struct TagTensorLaplaceML {};
  // tag for laplace_tensor
  struct TagTensorLaplaceReplaceML {};
  // tag for curl_sphere_wk_testcov
  struct TagCurlSphereWkTestCovML {};
  // tag for grad_sphere_wk_testcov
  struct TagGradSphereWkTestCovML {};
  // tag for default, a dummy
  struct TagDefault {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagDefault &,
                  TeamMember team) const {
      // do nothing or print a message
  };

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagGradientSphereML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_input_d = Kokkos::subview(
            scalar_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);
    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_vector_output_d = Kokkos::subview(
            vector_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          gradient_sphere(kv, dinv_d, dvv_d,
                          local_scalar_input_d,
                          local_vector_output_d);
        });  // end parallel_for for level

  }  // end of op() for grad_sphere_ml

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagDivergenceSphereWkML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_vector_input_d = Kokkos::subview(
            vector_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_output_d = Kokkos::subview(
            scalar_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          divergence_sphere_wk(kv, dinv_d, spheremp_d,
                               dvv_d, local_vector_input_d,
                               local_scalar_output_d);
        });  // end parallel_for for level

  }  // end of op() for divergence_sphere_wk_ml

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagSimpleLaplaceML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_input_d = Kokkos::subview(
            scalar_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_output_d = Kokkos::subview(
            scalar_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_temp1_d = Kokkos::subview(
            temp1_d, _index, Kokkos::ALL, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          laplace_simple(kv, dinv_d, spheremp_d, dvv_d,
                     local_temp1_d, local_scalar_input_d,
                     local_scalar_output_d);
        });  // end parallel_for for level

  }  // end of op() for laplace_wk_ml

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagTensorLaplaceML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_input_d = Kokkos::subview(
            scalar_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_output_d = Kokkos::subview(
            scalar_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_temp1_d = Kokkos::subview(
            temp1_d, _index, Kokkos::ALL, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          laplace_tensor(kv, dinv_d, spheremp_d, dvv_d, tensor_d,
                     local_temp1_d, local_scalar_input_d,
                     local_scalar_output_d);
        });  // end parallel_for for level

  }  // end of op() for laplace_tensor multil


  KOKKOS_INLINE_FUNCTION
  void operator()(const TagTensorLaplaceReplaceML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_input_d = Kokkos::subview(
            scalar_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_output_d = Kokkos::subview(
            scalar_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

//temp vars could have no ie dim from the start...
    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_temp1_d = Kokkos::subview(
            temp1_d, _index, Kokkos::ALL, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    for(int i=0; i< NP; i++)
    for(int j=0; j< NP; j++)
    for(int k = 0; k< NUM_LEV; k++){
       temp4_d(i,j,k) = local_scalar_input_d(i,j,k);
    }

//NOT YET WORKING 
//here is a problem, we'd like to store input in the usual variable
//for F and output in the usual variable for comparison.
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          laplace_tensor_replace(kv, dinv_d, spheremp_d, dvv_d, tensor_d,
                     local_temp1_d, local_scalar_input_d);
        });  // end parallel_for for level
//record local_input to output, replace local_input
    for(int i=0; i< NP; i++)
    for(int j=0; j< NP; j++)
    for(int k = 0; k< NUM_LEV; k++){
       local_scalar_output_d(i,j,k) = local_scalar_input_d(i,j,k);
       local_scalar_input_d(i,j,k) = temp4_d(i,j,k);
    }


  }  // end of op() for laplace_tensor multil


  KOKKOS_INLINE_FUNCTION
  void operator()(const TagCurlSphereWkTestCovML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_input_d = Kokkos::subview(
            scalar_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);
    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_vector_output_d = Kokkos::subview(
            vector_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          curl_sphere_wk_testcov(kv, d_d, mp_d, dvv_d,
                          local_scalar_input_d,
                          local_vector_output_d);
        });  // end parallel_for for level

  }  // end of op() for curl_sphere_wk_testcov

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagGradSphereWkTestCovML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_input_d = Kokkos::subview(
            scalar_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);
    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_vector_output_d = Kokkos::subview(
            vector_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          grad_sphere_wk_testcov(kv, d_d, mp_d, metinv_d,
                          metdet_d, dvv_d,
                          local_scalar_input_d,
                          local_vector_output_d);
        });  // end parallel_for for level

  }  // end of op() for grad_sphere_wk_testcov



  void run_functor_gradient_sphere() const {
    // league, team, vector_length_request=1
    Kokkos::TeamPolicy<ExecSpace, TagGradientSphereML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    // TO FROM
    Kokkos::deep_copy(vector_output_host, vector_output_d);
  };

  void run_functor_divergence_sphere_wk() const {
    Kokkos::TeamPolicy<ExecSpace, TagDivergenceSphereWkML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(scalar_output_host, scalar_output_d);
  };

  void run_functor_laplace_wk() const {
    Kokkos::TeamPolicy<ExecSpace, TagSimpleLaplaceML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(scalar_output_host, scalar_output_d);
  };

  void run_functor_tensor_laplace() const {
    Kokkos::TeamPolicy<ExecSpace, TagTensorLaplaceML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(scalar_output_host, scalar_output_d);
  };

  void run_functor_tensor_laplace_replace() const {
    Kokkos::TeamPolicy<ExecSpace, TagTensorLaplaceReplaceML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(scalar_output_host, scalar_output_d);
  };

  void run_functor_curl_sphere_wk_testcov() const {
    Kokkos::TeamPolicy<ExecSpace, TagCurlSphereWkTestCovML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(vector_output_host, vector_output_d);
  };

  void run_functor_grad_sphere_wk_testcov() const {
    Kokkos::TeamPolicy<ExecSpace, TagGradSphereWkTestCovML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(vector_output_host, vector_output_d);
  };

};  // end of class def compute_sphere_op_test_ml

class compute_sphere_operator_test {
 public:
  compute_sphere_operator_test(int num_elems)
      : scalar_input_d("scalar input", num_elems),
        vector_input_d("vector input", num_elems),
        d_d("d", num_elems),
        dinv_d("dinv", num_elems),
        metdet_d("metdet", num_elems),
        spheremp_d("spheremp", num_elems),
        dvv_d("dvv"),
        scalar_output_d("scalar output", num_elems),
        vector_output_d("vector output", num_elems),
        temp1_d("temp1", num_elems),
        temp2_d("temp2", num_elems),
        temp3_d("temp3", num_elems),
        scalar_input_host(
            Kokkos::create_mirror_view(scalar_input_d)),
        vector_input_host(
            Kokkos::create_mirror_view(vector_input_d)),
        d_host(Kokkos::create_mirror_view(d_d)),
        dinv_host(Kokkos::create_mirror_view(dinv_d)),
        metdet_host(Kokkos::create_mirror_view(metdet_d)),
        spheremp_host(
            Kokkos::create_mirror_view(spheremp_d)),
        dvv_host(Kokkos::create_mirror_view(dvv_d)),
        scalar_output_host(
            Kokkos::create_mirror_view(scalar_output_d)),
        vector_output_host(
            Kokkos::create_mirror_view(vector_output_d)),
        temp1_host(Kokkos::create_mirror_view(temp1_d)),
        temp2_host(Kokkos::create_mirror_view(temp2_d)),
        temp3_host(Kokkos::create_mirror_view(temp3_d)),
        _num_elems(num_elems) {
    // constructor's body
    // init randonly

    std::random_device rd;
    rngAlg engine(rd());

    // check singularities? divergence_wk uses both D and
    // Dinv, does it matter if ther are  not inverses of each
    // other?
    genRandArray(
        scalar_input_host.data(),
        scalar_input_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(0, 100.0));
    genRandArray(vector_input_host.data(),
                 vector_input_len * _num_elems, engine,
                 std::uniform_real_distribution<Real>(
                     -100.0, 100.0));
    genRandArray(
        d_host.data(), d_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(0, 1.0));
    genRandArray(
        dinv_host.data(), dinv_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(0, 1.0));
    genRandArray(
        metdet_host.data(), metdet_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(0, 1.0));
    genRandArray(
        spheremp_host.data(), spheremp_len * _num_elems,
        engine,
        std::uniform_real_distribution<Real>(0, 1.0));
    genRandArray(
        dvv_host.data(), dvv_len, engine,
        std::uniform_real_distribution<Real>(0, 1.0));
  }
  int _num_elems;  // league size, serves as ie index

  // device views
  ExecViewManaged<Real * [NP][NP]> scalar_input_d;
  ExecViewManaged<Real * [2][NP][NP]> vector_input_d;
  ExecViewManaged<Real * [2][2][NP][NP]> d_d;
  ExecViewManaged<Real * [2][2][NP][NP]> dinv_d;
  ExecViewManaged<Real * [NP][NP]> spheremp_d;
  ExecViewManaged<Real * [NP][NP]> metdet_d;
  ExecViewManaged<Real[NP][NP]> dvv_d;
  ExecViewManaged<Real * [NP][NP]> scalar_output_d;
  ExecViewManaged<Real * [2][NP][NP]> vector_output_d;
  ExecViewManaged<Real * [2][NP][NP]> temp1_d, temp2_d,
      temp3_d;

  // host views, one dim is num_elems. Spherical operators
  // do not take ie or nlev fields,  but to make it a more
  // reasonable test and to have parallel_for we ise another
  // dimension.
  ExecViewManaged<Real * [NP][NP]>::HostMirror
      scalar_input_host;
  // how to get total length of view? use dim0*dim1*...till
  // dim7
  const int scalar_input_len = NP * NP;  // temp code
  ExecViewManaged<Real * [2][NP][NP]>::HostMirror
      vector_input_host;
  // how to get total length of view? use dim0*dim1*...till
  // dim7
  const int vector_input_len = 2 * NP * NP;  // temp code
  ExecViewManaged<Real * [2][2][NP][NP]>::HostMirror d_host;
  const int d_len = 2 * 2 * NP * NP;  // temp code
  ExecViewManaged<Real * [2][2][NP][NP]>::HostMirror
      dinv_host;
  const int dinv_len = 2 * 2 * NP * NP;  // temp code
  ExecViewManaged<Real * [NP][NP]>::HostMirror metdet_host;
  const int metdet_len = NP * NP;
  ExecViewManaged<Real * [NP][NP]>::HostMirror
      spheremp_host;
  const int spheremp_len = NP * NP;
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_host;
  const int dvv_len = NP * NP;
  ExecViewManaged<Real * [NP][NP]>::HostMirror
      scalar_output_host;
  ExecViewManaged<Real * [2][NP][NP]>::HostMirror
      vector_output_host;
  ExecViewManaged<Real * [2][NP][NP]>::HostMirror
      temp1_host,
      temp2_host, temp3_host;

  // tag for laplace_simple()
  struct TagSimpleLaplace {};
  // tag for gradient_sphere()
  struct TagGradientSphere {};
  // tag for divergence_sphere_wk
  struct TagDivergenceSphereWk {};
  // tag for default, a dummy
  struct TagDefault {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagDefault &,
                  TeamMember team) const {
      // do nothing or print a message
  };

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagSimpleLaplace &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Real[NP][NP]> local_scalar_input_d =
        Kokkos::subview(scalar_input_d, _index, Kokkos::ALL,
                        Kokkos::ALL);
    ExecViewManaged<Real[2][NP][NP]> local_temp1_d =
        Kokkos::subview(temp1_d, _index, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);
    ExecViewManaged<Real[NP][NP]> local_scalar_output_d =
        Kokkos::subview(scalar_output_d, _index,
                        Kokkos::ALL, Kokkos::ALL);

    laplace_wk_sl(kv, dinv_d, spheremp_d, dvv_d,
                  local_temp1_d, local_scalar_input_d,
                  local_scalar_output_d);

  };  // end of op() for laplace_simple

  /*
   * A comment on how these tests work:
   * Consider 160 threads available, with _num_elems=10;
   *
   * Below are lines
   *     Kokkos::TeamPolicy<ExecSpace, TagSimpleLaplace>
   * policy(_num_elems, 16); Kokkos::parallel_for(policy,
   * *this); this one will call operator() with, say,
   * weak divergence tag. Then first 160 threads will
   * be clustered into 10 leagues (as many as _num_elems).
   * Each league will contain 16 threads (16 as the second
   * argument in policy().) Each league will have its
   * separate subview input and subview output (subview of
   * global arrays based on team.league_rank), so, league 2
   * will have input from local_vector_input_d(2,:,:), etc.
   * When divergence_sphere_wk is called, it will be
   * executed by 16 threads in league, each sharing input
   * and output. So, it is not a perfect situation and not a
   * perfect test, because 16 team threads run the same code
   * and OVERWRITE the same output. A better test should
   * have another level of parallelism, a loop with
   * TeamThreadRange. Also, one should note that
   * divergence_sphere_wk as well as other SphereOperators
   * should be called from loop with aeamThreadRange.
   */

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagDivergenceSphereWk &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Real[2][NP][NP]> local_vector_input_d =
        Kokkos::subview(vector_input_d, _index, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);
    ExecViewManaged<Real[NP][NP]> local_scalar_output_d =
        Kokkos::subview(scalar_output_d, _index,
                        Kokkos::ALL, Kokkos::ALL);

    divergence_sphere_wk_sl(kv, dinv_d, spheremp_d, dvv_d,
                            local_vector_input_d,
                            local_scalar_output_d);
  };  // end of op() for divergence_sphere_wk

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagGradientSphere &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Real[NP][NP]> local_scalar_input_d =
        Kokkos::subview(scalar_input_d, _index, Kokkos::ALL,
                        Kokkos::ALL);
    ExecViewManaged<Real[2][NP][NP]> local_vector_output_d =
        Kokkos::subview(vector_output_d, _index,
                        Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL);

    gradient_sphere_sl(kv, dinv_d, dvv_d,
                       local_scalar_input_d,
                       local_vector_output_d);
  };

  // this could be even nicer,
  // put in a param in run_functor(param) to only branch
  // policy type
  void run_functor_simple_laplace() const {
    // league, team, vector_length_request=1
    Kokkos::TeamPolicy<ExecSpace, TagSimpleLaplace> policy(
        _num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    // TO FROM
    Kokkos::deep_copy(scalar_output_host, scalar_output_d);
  };

  void run_functor_gradient_sphere() const {
    // league, team, vector_length_request=1
    Kokkos::TeamPolicy<ExecSpace, TagGradientSphere> policy(
        _num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    // TO FROM
    Kokkos::deep_copy(vector_output_host, vector_output_d);
  };

  void run_functor_div_wk() const {
    // league, team, vector_length_request=1
    Kokkos::TeamPolicy<ExecSpace, TagDivergenceSphereWk>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    // TO FROM
    // remember to copy correct output
    Kokkos::deep_copy(scalar_output_host, scalar_output_d);
  };

};  // end of definition of compute_sphere_operator_test()

TEST_CASE("Testing laplace_simple_sl()",
          "laplace_simple_sl") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test testing_laplace(elements);

  testing_laplace.run_functor_simple_laplace();

  for(int _index = 0; _index < elements; _index++) {
    Real local_fortran_output[NP][NP];

    HostViewManaged<Real[NP][NP]> local_scalar_input =
        Kokkos::subview(testing_laplace.scalar_input_host,
                        _index, Kokkos::ALL, Kokkos::ALL);

    HostViewManaged<Real[2][2][NP][NP]> local_dinv =
        Kokkos::subview(testing_laplace.dinv_host, _index,
                        Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);

    HostViewManaged<Real[NP][NP]> local_spheremp =
        Kokkos::subview(testing_laplace.spheremp_host,
                        _index, Kokkos::ALL, Kokkos::ALL);

    // F input declared
    Real sf[NP][NP];
    Real dvvf[NP][NP];
    Real dinvf[2][2][NP][NP];
    Real sphf[NP][NP];

    // flip arrays for F
    for(int _i = 0; _i < NP; _i++)
      for(int _j = 0; _j < NP; _j++) {
        sf[_i][_j] = local_scalar_input(_i, _j);
        dvvf[_i][_j] = testing_laplace.dvv_host(_i, _j);
        sphf[_i][_j] = local_spheremp(_i, _j);
        for(int _d1 = 0; _d1 < 2; _d1++)
          for(int _d2 = 0; _d2 < 2; _d2++)
            dinvf[_d1][_d2][_i][_j] =
                local_dinv(_d1, _d2, _i, _j);
      }

    // run F code
    laplace_simple_c_callable(
        &(sf[0][0]), &(dvvf[0][0]), &(dinvf[0][0][0][0]),
        &(sphf[0][0]), &(local_fortran_output[0][0]));

    // compare answers
    for(int igp = 0; igp < NP; ++igp) {
      for(int jgp = 0; jgp < NP; ++jgp) {
        REQUIRE(
            !std::isnan(local_fortran_output[igp][jgp]));
        REQUIRE(
            !std::isnan(testing_laplace.scalar_output_host(
                _index, igp, jgp)));
        REQUIRE(std::numeric_limits<Real>::epsilon() >=
                compare_answers(
                    local_fortran_output[igp][jgp],
                    testing_laplace.scalar_output_host(
                        _index, igp, jgp)));
      }  // jgp
    }    // igp
  }      // end of for loop for elements

  std::cout << "simple_laplace_sl single level test finished.\n";

};  // end of TEST_CASE(..., "simple laplace")

TEST_CASE("Testing div_wk_sl()", "div_wk_sl") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 1;

  compute_sphere_operator_test testing_divwk(elements);

  testing_divwk.run_functor_div_wk();

  Real local_fortran_output[NP][NP];

  for(int _index = 0; _index < elements; _index++) {
    Real local_fortran_output[NP][NP];

    HostViewManaged<Real[2][NP][NP]> local_vector_input =
        Kokkos::subview(testing_divwk.vector_input_host,
                        _index, Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL);

    HostViewManaged<Real[2][2][NP][NP]> local_dinv =
        Kokkos::subview(testing_divwk.dinv_host, _index,
                        Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);

    HostViewManaged<Real[NP][NP]> local_spheremp =
        Kokkos::subview(testing_divwk.spheremp_host, _index,
                        Kokkos::ALL, Kokkos::ALL);

    Real vf[2][NP][NP];
    Real dvvf[NP][NP];
    Real dinvf[2][2][NP][NP];
    Real metf[NP][NP];
    Real sphf[NP][NP];

    for(int _i = 0; _i < NP; _i++)
      for(int _j = 0; _j < NP; _j++) {
        dvvf[_i][_j] = testing_divwk.dvv_host(_i, _j);
        sphf[_i][_j] = local_spheremp(_i, _j);
        for(int _d1 = 0; _d1 < 2; _d1++) {
          vf[_d1][_i][_j] = local_vector_input(_d1, _i, _j);
          for(int _d2 = 0; _d2 < 2; _d2++)
            dinvf[_d1][_d2][_i][_j] =
                local_dinv(_d1, _d2, _i, _j);
        }
      }

    divergence_sphere_wk_c_callable(
        &(vf[0][0][0]), &(dvvf[0][0]), &(sphf[0][0]),
        &(dinvf[0][0][0][0]),
        &(local_fortran_output[0][0]));

    for(int igp = 0; igp < NP; ++igp) {
      for(int jgp = 0; jgp < NP; ++jgp) {
        REQUIRE(
            !std::isnan(local_fortran_output[igp][jgp]));
        REQUIRE(
            !std::isnan(testing_divwk.scalar_output_host(
                _index, igp, jgp)));
        REQUIRE(std::numeric_limits<Real>::epsilon() >=
                compare_answers(
                    local_fortran_output[igp][jgp],
                    testing_divwk.scalar_output_host(
                        _index, igp, jgp)));
      }  // jgp
    }    // igp
  };     // end of elements loop

  std::cout << "div_wk single level test finished.\n";

}  // end of TEST_CASE(...,"divergence_sphere_wk")

TEST_CASE("Testing gradient_sphere_sl()",
          "gradient_sphere") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test testing_grad(elements);

  // running kokkos version of operator
  testing_grad.run_functor_gradient_sphere();

  for(int _index = 0; _index < elements; _index++) {
    Real local_fortran_output[2][NP][NP];

    HostViewManaged<Real[NP][NP]> local_scalar_input =
        Kokkos::subview(testing_grad.scalar_input_host,
                        _index, Kokkos::ALL, Kokkos::ALL);

    HostViewManaged<Real[2][2][NP][NP]> local_dinv =
        Kokkos::subview(testing_grad.dinv_host, _index,
                        Kokkos::ALL, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);

    Real sf[NP][NP];
    Real dvvf[NP][NP];
    Real dinvf[2][2][NP][NP];

    // flipping arrays -- WRITE DOWN HOW THIS SHOULD BE DONE
    for(int _i = 0; _i < NP; _i++)
      for(int _j = 0; _j < NP; _j++) {
        sf[_i][_j] = local_scalar_input(_i, _j);
        dvvf[_i][_j] = testing_grad.dvv_host(_i, _j);
        for(int _d1 = 0; _d1 < 2; _d1++)
          for(int _d2 = 0; _d2 < 2; _d2++)
            dinvf[_d1][_d2][_i][_j] =
                local_dinv(_d1, _d2, _i, _j);
      }

    // running F version of operator
    gradient_sphere_c_callable(
        &(sf[0][0]), &(dvvf[0][0]), &(dinvf[0][0][0][0]),
        &(local_fortran_output[0][0][0]));

    // comparing answers from kokkos and F
    for(int igp = 0; igp < NP; ++igp)
      for(int jgp = 0; jgp < NP; ++jgp)
        for(int _d = 0; _d < 2; ++_d) {
          REQUIRE(!std::isnan(
              local_fortran_output[_d][igp][jgp]));
          REQUIRE(
              !std::isnan(testing_grad.vector_output_host(
                  _index, _d, igp, jgp)));
          REQUIRE(std::numeric_limits<Real>::epsilon() >=
                  compare_answers(
                      local_fortran_output[_d][igp][jgp],
                      testing_grad.vector_output_host(
                          _index, _d, igp, jgp)));
        }  // end of comparing answers

  }  // end of loop for elements

  std::cout << "grad single level test finished.\n";

};  // end of TEST_CASE(..., "gradient_sphere")

// SHMEM ????

TEST_CASE("Testing gradient_sphere()",
          "gradient_sphere") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_grad_ml(elements);
  testing_grad_ml.run_functor_gradient_sphere();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        // fortran output
        Real local_fortran_output[2][NP][NP];
        // F input
        Real sf[NP][NP];
        Real dvvf[NP][NP];
        Real dinvf[2][2][NP][NP];

        // since i don't need flipping arrays, maybe, it is
        // enough to pass data() pointer?
        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            sf[_i][_j] = testing_grad_ml.scalar_input_host(
                _index, _i, _j, level)[v];
            dvvf[_i][_j] = testing_grad_ml.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++)
                dinvf[_d1][_d2][_i][_j] =
                    testing_grad_ml.dinv_host(_index, _d1,
                                              _d2, _i, _j);
          }

        // running F version of operator
        gradient_sphere_c_callable(
            &(sf[0][0]), &(dvvf[0][0]),
            &(dinvf[0][0][0][0]),
            &(local_fortran_output[0][0][0]));

        // compare with the part from C run
        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 =
                testing_grad_ml.vector_output_host(
                    _index, 0, igp, jgp, level)[v];
            Real coutput1 =
                testing_grad_ml.vector_output_host(
                    _index, 1, igp, jgp, level)[v];
            REQUIRE(!std::isnan(
                local_fortran_output[0][igp][jgp]));
            REQUIRE(!std::isnan(
                local_fortran_output[1][igp][jgp]));
            REQUIRE(!std::isnan(coutput0));
            REQUIRE(!std::isnan(coutput1));
            // what is 128 here?
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[0][igp][jgp],
                        coutput0, 128.0));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[1][igp][jgp],
                        coutput1, 128.0));
          }  // jgp
        }    // igp
      }      // v
    }        // level
  }          //_index

  std::cout << "test grad multilevel finished. \n";

}  // end fo test grad_sphere_ml

TEST_CASE("Testing divergence_sphere_wk()",
          "divergence_sphere_wk") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_div_ml(elements);
  testing_div_ml.run_functor_divergence_sphere_wk();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        // fortran output
        Real local_fortran_output[NP][NP];
        // F input
        Real vf[2][NP][NP];
        Real dvvf[NP][NP];
        Real dinvf[2][2][NP][NP];
        Real sphf[NP][NP];

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            sphf[_i][_j] = testing_div_ml.spheremp_host(
                _index, _i, _j);
            dvvf[_i][_j] = testing_div_ml.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++) {
              vf[_d1][_i][_j] =
                  testing_div_ml.vector_input_host(
                      _index, _d1, _i, _j, level)[v];
              for(int _d2 = 0; _d2 < 2; _d2++)
                dinvf[_d1][_d2][_i][_j] =
                    testing_div_ml.dinv_host(_index, _d1,
                                             _d2, _i, _j);
            }
          }
        divergence_sphere_wk_c_callable(
            &(vf[0][0][0]), &(dvvf[0][0]), &(sphf[0][0]),
            &(dinvf[0][0][0][0]),
            &(local_fortran_output[0][0]));
        // compare with the part from C run
        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 =
                testing_div_ml.scalar_output_host(
                    _index, igp, jgp, level)[v];
            REQUIRE(!std::isnan(
                local_fortran_output[igp][jgp]));
            REQUIRE(!std::isnan(coutput0));
            // what is 128 here?
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[igp][jgp],
                        coutput0, 128.0));
          }  // jgp
        }    // igp
      }      // v
    }        // level
  }          //_index

  std::cout << "test div_wk multilevel finished. \n";

}  // end of test div_sphere_wk_ml

TEST_CASE("Testing simple laplace_wk()",
          "laplace_wk") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_laplace_ml(
      elements);
  testing_laplace_ml.run_functor_laplace_wk();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        // fortran output
        Real local_fortran_output[NP][NP];
        // F input
        Real sf[NP][NP];
        Real dvvf[NP][NP];
        Real dinvf[2][2][NP][NP];
        Real sphf[NP][NP];

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            sf[_i][_j] =
                testing_laplace_ml.scalar_input_host(
                    _index, _i, _j, level)[v];
            sphf[_i][_j] = testing_laplace_ml.spheremp_host(
                _index, _i, _j);
            dvvf[_i][_j] =
                testing_laplace_ml.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++)
                dinvf[_d1][_d2][_i][_j] =
                    testing_laplace_ml.dinv_host(
                        _index, _d1, _d2, _i, _j);
          }

        laplace_simple_c_callable(&(sf[0][0]), &(dvvf[0][0]),
                             &(dinvf[0][0][0][0]),
                             &(sphf[0][0]),
                             &(local_fortran_output[0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 =
                testing_laplace_ml.scalar_output_host(
                    _index, igp, jgp, level)[v];
            REQUIRE(!std::isnan(
                local_fortran_output[igp][jgp]));
            REQUIRE(!std::isnan(coutput0));
            // what is 128 here?
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[igp][jgp],
                        coutput0, 128.0));
          }  // jgp
        }    // igp
      }      // v
    }        // level
  }          //_index

  std::cout << "test laplace_simple multilevel finished. \n";

}  // end of test laplace_simple multilevel


TEST_CASE("Testing laplace_tensor() multilevel",
          "laplace_tensor") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_tensor_laplace(
      elements);
  testing_tensor_laplace.run_functor_tensor_laplace();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real local_fortran_output[NP][NP];
        // F input
        Real sf[NP][NP];
        Real dvvf[NP][NP];
        Real dinvf[2][2][NP][NP];
        Real tensorf[2][2][NP][NP];
        Real sphf[NP][NP];

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            sf[_i][_j] =
                testing_tensor_laplace.scalar_input_host(
                    _index, _i, _j, level)[v];
            sphf[_i][_j] = testing_tensor_laplace.spheremp_host(
                _index, _i, _j);
            dvvf[_i][_j] =
                testing_tensor_laplace.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++){

                dinvf[_d1][_d2][_i][_j] = 
     testing_tensor_laplace.dinv_host( _index, _d1, _d2, _i, _j);

                tensorf[_d1][_d2][_i][_j] =
     testing_tensor_laplace.tensor_host( _index, _d1, _d2, _i, _j);

              }//end of d2 loop
          }

Real _hp = 0.0;
Real _hs = 1.0;
bool _vc = true;

        laplace_sphere_wk_c_callable(&(sf[0][0]), &(dvvf[0][0]),
                             &(dinvf[0][0][0][0]),
                             &(sphf[0][0]),&(tensorf[0][0][0][0]),
                             _hp, _hs, 
                             &_vc,
                             &(local_fortran_output[0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {

            Real coutput0 =
                testing_tensor_laplace.scalar_output_host(
                    _index, igp, jgp, level)[v];

//std::cout << igp << "," << jgp << " F output  = " <<
//local_fortran_output[igp][jgp] << ", C output" << coutput0 << "\n";

            REQUIRE(!std::isnan(
                local_fortran_output[igp][jgp]));
            REQUIRE(!std::isnan(coutput0));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[igp][jgp],
                        coutput0, 128.0));
          }  // jgp
        }    // igp
      }      // v
    }        // level
  }          //_index

  std::cout << "test laplace_tensor multilevel finished. \n";

}  // end of test laplace_tensor multilevel



TEST_CASE("Testing laplace_tensor_replace() multilevel",
          "laplace_tensor_replace") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_tensor_laplace(
      elements);
std::cout << "here 1 \n";
  testing_tensor_laplace.run_functor_tensor_laplace_replace();

std::cout << "here 2 \n";
  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real local_fortran_output[NP][NP];
        Real sf[NP][NP];
        Real dvvf[NP][NP];
        Real dinvf[2][2][NP][NP];
        Real tensorf[2][2][NP][NP];
        Real sphf[NP][NP];

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            sf[_i][_j] =
                testing_tensor_laplace.scalar_input_host(
                    _index, _i, _j, level)[v];
            sphf[_i][_j] = testing_tensor_laplace.spheremp_host(
                _index, _i, _j);
            dvvf[_i][_j] =
                testing_tensor_laplace.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++){

                dinvf[_d1][_d2][_i][_j] =
     testing_tensor_laplace.dinv_host( _index, _d1, _d2, _i, _j);

                tensorf[_d1][_d2][_i][_j] =
     testing_tensor_laplace.tensor_host( _index, _d1, _d2, _i, _j);

              }//end of d2 loop
          }

Real _hp = 0.0;
Real _hs = 1.0;
bool _vc = true;

        laplace_sphere_wk_c_callable(&(sf[0][0]), &(dvvf[0][0]),
                             &(dinvf[0][0][0][0]),
                             &(sphf[0][0]),&(tensorf[0][0][0][0]),
                             _hp, _hs,
                             &_vc,
                             &(local_fortran_output[0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {

            Real coutput0 =
                testing_tensor_laplace.scalar_output_host(
                    _index, igp, jgp, level)[v];

           REQUIRE(!std::isnan(
                local_fortran_output[igp][jgp]));
            REQUIRE(!std::isnan(coutput0));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[igp][jgp],
                        coutput0, 128.0));
          }  // jgp
        }    // igp
      }      // v
    }        // level
  }          //_index

  std::cout << "test laplace_tensor_replace multilevel finished. \n";

}  // end of test laplace_tensor_replace multilevel







TEST_CASE("Testing curl_sphere_wk_testcov() multilevel",
          "curl_sphere_wk_testcov") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 1;

  compute_sphere_operator_test_ml testing_curl(
      elements);
  testing_curl.run_functor_curl_sphere_wk_testcov();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real local_fortran_output[2][NP][NP];
        Real sf[NP][NP];
        Real dvvf[NP][NP];
        Real df[2][2][NP][NP];
        Real mpf[NP][NP];

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            sf[_i][_j] =
                testing_curl.scalar_input_host(
                    _index, _i, _j, level)[v];
            mpf[_i][_j] = testing_curl.mp_host(
                _index, _i, _j);
            dvvf[_i][_j] =
                testing_curl.dvv_host(_i, _j);

        for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++){

                df[_d1][_d2][_i][_j] =
     testing_curl.d_host( _index, _d1, _d2, _i, _j);

              }//end of d2 loop
          }
          curl_sphere_wk_testcov_c_callable(&(sf[0][0]), &(dvvf[0][0]),
                             &(df[0][0][0][0]),
                             &(mpf[0][0]),
                             &(local_fortran_output[0][0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {

            Real coutput0 =
                testing_curl.vector_output_host(
                    _index, 0, igp, jgp, level)[v];

            Real coutput1 =
                testing_curl.vector_output_host(
                    _index, 1, igp, jgp, level)[v];

/*
std::cout << igp << "," << jgp << " F output0  = " <<
local_fortran_output[0][igp][jgp] << ", C output0 = " << coutput0 << "\n";
std::cout << "difference=" << local_fortran_output[0][igp][jgp] - coutput0 << "\n";
std::cout << "rel difference=" << (local_fortran_output[0][igp][jgp] - coutput0)/coutput0 << "\n";

std::cout << "difference=" << local_fortran_output[1][igp][jgp] - coutput1 << "\n";
std::cout << "rel difference=" << (local_fortran_output[1][igp][jgp] - coutput1)/coutput1 << "\n";

std::cout << "epsilon = " << std::numeric_limits<Real>::epsilon() << "\n";
*/

            REQUIRE(!std::isnan(
                local_fortran_output[0][igp][jgp]));

            REQUIRE(!std::isnan(
                local_fortran_output[1][igp][jgp]));

            REQUIRE(!std::isnan(coutput0));
            REQUIRE(!std::isnan(coutput1));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[0][igp][jgp],
                        coutput0, 128.0));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[1][igp][jgp],
                        coutput1, 128.0));

          }  // jgp
        }    // igp
      }      // v
    }        // level
  }          //_index

  std::cout << "test curl_sphere_wk_testcov multilevel finished. \n";

}  // end of test laplace_tensor multilevel



TEST_CASE("Testing grad_sphere_wk_testcov() multilevel",
          "grad_sphere_wk_testcov") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 1;

std::cout << "here 1 \n";

  compute_sphere_operator_test_ml testing_grad(
      elements);
  testing_grad.run_functor_grad_sphere_wk_testcov();

std::cout << "here 2 \n";

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real local_fortran_output[2][NP][NP];
        Real sf[NP][NP];
        Real dvvf[NP][NP];
        Real df[2][2][NP][NP];
        Real mpf[NP][NP];
        Real metdetf[NP][NP];
        Real metinvf[2][2][NP][NP];

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            sf[_i][_j] =
                testing_grad.scalar_input_host(
                    _index, _i, _j, level)[v];
            mpf[_i][_j] = testing_grad.mp_host(
                _index, _i, _j);
            metdetf[_i][_j] = testing_grad.metdet_host(
                _index, _i, _j);
            dvvf[_i][_j] =
                testing_grad.dvv_host(_i, _j);

        for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++){

                df[_d1][_d2][_i][_j] =
     testing_grad.d_host( _index, _d1, _d2, _i, _j);

                metinvf[_d1][_d2][_i][_j] =
     testing_grad.metinv_host( _index, _d1, _d2, _i, _j);
              }//end of d2 loop
          }
          gradient_sphere_wk_testcov_c_callable(&(sf[0][0]), &(dvvf[0][0]),
                             &(metinvf[0][0][0][0]),
                             &(metdetf[0][0]),
                             &(df[0][0][0][0]),
                             &(mpf[0][0]),
                             &(local_fortran_output[0][0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {

            Real coutput0 =
                testing_grad.vector_output_host(
                    _index, 0, igp, jgp, level)[v];

            Real coutput1 =
                testing_grad.vector_output_host(
                    _index, 1, igp, jgp, level)[v];

/*
 * std::cout << igp << "," << jgp << " F output0  = " <<
 * local_fortran_output[0][igp][jgp] << ", C output0 = " << coutput0 << "\n";
 * std::cout << "difference=" << local_fortran_output[0][igp][jgp] - coutput0 << "\n";
 * std::cout << "rel difference=" << (local_fortran_output[0][igp][jgp] - coutput0)/coutput0 << "\n";
 *
 * std::cout << "difference=" << local_fortran_output[1][igp][jgp] - coutput1 << "\n";
 * std::cout << "rel difference=" << (local_fortran_output[1][igp][jgp] - coutput1)/coutput1 << "\n";
 *
 * std::cout << "epsilon = " << std::numeric_limits<Real>::epsilon() << "\n";
 * */

            REQUIRE(!std::isnan(
                local_fortran_output[0][igp][jgp]));

            REQUIRE(!std::isnan(
                local_fortran_output[1][igp][jgp]));

            REQUIRE(!std::isnan(coutput0));
            REQUIRE(!std::isnan(coutput1));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[0][igp][jgp],
                        coutput0, 128.0));
            REQUIRE(std::numeric_limits<Real>::epsilon() >=
                    compare_answers(
                        local_fortran_output[1][igp][jgp],
                        coutput1, 128.0));

          }  // jgp
        }    // igp
      }      // v
    }        // level
  }          //_index

  std::cout << "test grad_sphere_wk_testcov multilevel finished. \n";

}  // end of test laplace_tensor multilevel






