#include <catch/catch.hpp>

#include <limits>

#include "Control.hpp"
#include "CaarFunctor.hpp"
#include "Elements.hpp"
#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"
#include "Utility.hpp"

#include <assert.h>
#include <stdio.h>
#include <random>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

// inserting names of params for _c_callable functions.
// instead of using names as in function descriptions, using
// verbose names
void laplace_simple_c_callable(const Real *input,
                               const Real *dvv,
                               const Real *dinv,
                               const Real *metdet,
                               Real *output);

void gradient_sphere_c_callable(const Real *input,
                                const Real *dvv,
                                const Real *dinv,
                                Real *output);

void divergence_sphere_wk_c_callable(const Real *input,
                                     const Real *dvv,
                                     const Real *spheremp,
                                     const Real *dinv,
                                     Real *output);

// This is an interface for the most general 2d laplace in
// F. In c++ we will ignore 'var coef' HV. If one uses
//var_coef in below, then both 'var_coef' and 'tensor' HV
// are possible. the switch between them is whether hvpower
// is <>0. To exclude 'var coef' HV we will always set
//hvpower to 0. If hvscaling is <>0, then tensor is used.
// So, settings for usual HV are var_coef=FALSE.
// Settings for tensor HV are var_coef=TRUE, hvpower=0,
// hvscaling >0. (don't ask me why this is so).  Update,
// copypasting from stackoverflow:  Typical Fortran
// implementations pass all arguments by reference,  while in
// C(++) the default is by value. This fixes an issue with
// hvpower, etc.
void laplace_sphere_wk_c_callable(
    const Real *input, const Real *dvv, const Real *dinv,
    const Real *spheremp, const Real *tensorVisc,
    const Real &hvpower,    // should be set to 0 always
    const Real &hvscaling,  // should be set to !=0 value
    const bool
        var_coef,  // should be set to 1 for tensor HV
    Real *output);

void curl_sphere_wk_testcov_c_callable(
    const Real *input,  // s(np,np)
    const Real *dvv,    // dvv(np, np)
    const Real *D,      // D(np, np, 2, 2)
    const Real *mp,     // mp(np, np)
    Real *output);      // ds(np,np,2)

void gradient_sphere_wk_testcov_c_callable(
    const Real *input,   // s(np,np)
    const Real *dvv,     // dvv(np, np)
    const Real *metinv,  // metinv(np, np, 2, 2)
    const Real *metdet,  // metdet(np, np)
    const Real *D,       // D(np, np, 2, 2)
    const Real *mp,      // mp(np, np)
    Real *output);       // ds(np,np,2)

void vlaplace_sphere_wk_cartesian_c_callable(
    const Real *input, const Real *dvv, const Real *dinv,
    const Real *spheremp, const Real *tensorVisc,
    const Real *vec_sph2cart, const Real &hvpower,
    const Real &hvscaling, const bool var_coef,
    Real *output);

void vlaplace_sphere_wk_contra_c_callable(
    const Real *input, const Real *dvv, const Real *d,
    const Real *dinv, const Real *mp, const Real *spheremp,
    const Real *metinv, const Real *metdet,
    const Real *rmetdet, const Real &nu_ratio,
    Real *output);

void vorticity_sphere_c_callable(
    const Real *input, const Real *dvv,
    const Real *metdet, const Real *d,
    Real *output);

}  // extern C

class compute_sphere_operator_test_ml {
 public:
  compute_sphere_operator_test_ml(int num_elems)
      : scalar_input_d("scalar input", num_elems),
        scalar_input_COPY2_d("scalar input 2", num_elems),
        vector_input_d("vector input", num_elems),
        d_d("d", num_elems),
        dinv_d("dinv", num_elems),
        metinv_d("metinv", num_elems),
        metdet_d("metdet", num_elems),
        spheremp_d("spheremp", num_elems),
        mp_d("mp", num_elems),
        dvv_d("dvv"),
        tensor_d("tensor", num_elems),
        vec_sph2cart_d("ver_sph2cart", num_elems),
        scalar_output_d("scalar output", num_elems),
        vector_output_d("vector output", num_elems),
        temp1_d("temp1", num_elems),
        temp2_d("temp2", num_elems),
        temp3_d("temp3", num_elems),
        temp4_d("temp4", num_elems),
        temp5_d("temp5", num_elems),
        temp6_d("temp6", num_elems),
        sphere_buf("Spherical buffer", num_elems),

        // Make certain Kokkos doesn't use the same arrays
        // as used on the device when they are mutable
        scalar_input_host("scalar input host", num_elems),
        scalar_input_COPY2_host("scalar input host copy", num_elems),
        vector_input_host("vector input host", num_elems),
        d_host(Kokkos::create_mirror_view(d_d)),
        dinv_host(Kokkos::create_mirror_view(dinv_d)),
        metinv_host(Kokkos::create_mirror_view(metinv_d)),
        metdet_host(Kokkos::create_mirror_view(metdet_d)),
        spheremp_host(
            Kokkos::create_mirror_view(spheremp_d)),
        mp_host(Kokkos::create_mirror_view(mp_d)),
        dvv_host(Kokkos::create_mirror_view(dvv_d)),
        tensor_host(Kokkos::create_mirror_view(tensor_d)),
        vec_sph2cart_host(
            Kokkos::create_mirror_view(vec_sph2cart_d)),
        scalar_output_host(
            Kokkos::create_mirror_view(scalar_output_d)),
        vector_output_host(
            Kokkos::create_mirror_view(vector_output_d)),
        _num_elems(num_elems) {
    std::random_device rd;
    rngAlg engine(rd());
    genRandArray(
        scalar_input_host, engine,
        std::uniform_real_distribution<Real>(-1000.0,
                                             1000.0));
    Kokkos::deep_copy(scalar_input_d, scalar_input_host);
    Kokkos::deep_copy(scalar_input_COPY2_host, scalar_input_host);

    genRandArray(
        vector_input_host, engine,
        std::uniform_real_distribution<Real>(-1000.0,
                                             1000.0));
    Kokkos::deep_copy(vector_input_d, vector_input_host);

    genRandArray(d_host, engine,
                 std::uniform_real_distribution<Real>(
                     -100.0, 100.0));
    Kokkos::deep_copy(d_d, d_host);

    genRandArray(dinv_host,
                 engine,
                 std::uniform_real_distribution<Real>(
                     -100.0, 100.0));
    Kokkos::deep_copy(dinv_d, dinv_host);

    genRandArray(metinv_host, engine,
                 std::uniform_real_distribution<Real>(
                     -100.0, 100.0));
    Kokkos::deep_copy(metinv_d, metinv_host);

    genRandArray(metdet_host, engine,
                 std::uniform_real_distribution<Real>(
                     -100.0, 100.0));
    Kokkos::deep_copy(metdet_d, metdet_host);

    genRandArray(spheremp_host, engine,
                 std::uniform_real_distribution<Real>(
                     -100.0, 100.0));
    Kokkos::deep_copy(spheremp_d, spheremp_host);

    genRandArray(mp_host,
                 engine,
                 std::uniform_real_distribution<Real>(
                     -100.0, 100.0));
    Kokkos::deep_copy(mp_d, mp_host);

    genRandArray(dvv_host, engine,
                 std::uniform_real_distribution<Real>(
                     -100.0, 100.0));
    Kokkos::deep_copy(dvv_d, dvv_host);

    genRandArray(tensor_host, engine,
                 std::uniform_real_distribution<Real>(
                     -1000, 1000.0));
    Kokkos::deep_copy(tensor_d, tensor_host);

    genRandArray(vec_sph2cart_host, engine,
                 std::uniform_real_distribution<Real>(
                     -1000, 1000.0));
    Kokkos::deep_copy(vec_sph2cart_d, vec_sph2cart_host);

    genRandArray(&nu_ratio, 1, engine,
                 std::uniform_real_distribution<Real>(
                     -1000, 1000.0));
// setting everything to 1 is good for debugging
#if 0
    for(int i1=0; i1<_num_elems; i1++) {
      for(int i2=0; i2<NP; i2++) {
        for(int i3=0; i3<NP; i3++) {
          dinv_host(i1,0,0,i2,i3)=1.0;
          dinv_host(i1,1,1,i2,i3)=1.0;
          dinv_host(i1,1,0,i2,i3)=1.0;
          dinv_host(i1,0,1,i2,i3)=1.0;
          tensor_host(i1,0,0,i2,i3)=1.0;
          tensor_host(i1,1,1,i2,i3)=1.0;
          tensor_host(i1,1,0,i2,i3)=1.0;
          tensor_host(i1,0,1,i2,i3)=1.0;

          vec_sph2cart_host(i1,0,0,i2,i3)=1.0;
          vec_sph2cart_host(i1,1,0,i2,i3)=1.0;
          vec_sph2cart_host(i1,0,1,i2,i3)=1.0;
          vec_sph2cart_host(i1,1,1,i2,i3)=1.0;
          vec_sph2cart_host(i1,0,2,i2,i3)=1.0;
          vec_sph2cart_host(i1,1,2,i2,i3)=1.0;
          //metdet_host(i1,i2,i3)=1.0;
          spheremp_host(i1,i2,i3)=1.0;
          dvv_host(i2,i3)=1.0;
          //mp_host(i1,i2,i3)=1.0;
          //           -//Real aa = i2+i3;
          //            -//scalar_input_host(i1,i2,i3) = aa;
          //
          //vector_input_host(i1,0,i2,i3)[...] = 1;//aa;
          //vector_input_host(i1,1,i2,i3)[...] = 1;//aa;
        }
      }
    }
#endif

  }  // end of constructor

  int _num_elems;  // league size, serves as ie index

  // device
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>
      scalar_input_d;
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>
      scalar_input_COPY2_d;
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
  ExecViewManaged<Real * [2][3][NP][NP]> vec_sph2cart_d;
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>
      scalar_output_d;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>
      vector_output_d;
  // making temp vars with leading dimension 'ie' to avoid
  // thread sharing issues  in the ie loop
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]> temp1_d,
      temp2_d, temp3_d;

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> temp4_d,
      temp5_d, temp6_d;

  ExecViewManaged<Scalar * [NUM_LEV][2][NP][NP]> sphere_buf;

  // host
  // rely on fact NUM_PHYSICAL_LEV=NUM_LEV*VECTOR_SIZE
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror
      scalar_input_host,
      scalar_input_COPY2_host;
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

  ExecViewManaged<Real * [NP][NP]>::HostMirror mp_host;
  const int mp_len = NP * NP;

  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_host;
  const int dvv_len = NP * NP;

  ExecViewManaged<Real * [2][2][NP][NP]>::HostMirror
      tensor_host;
  const int tensor_len = 2 * 2 * NP * NP;  // temp code

  ExecViewManaged<Real * [2][3][NP][NP]>::HostMirror
      vec_sph2cart_host;
  const int vec_sph2cart_len =
      2 * 3 * NP * NP;  // temp code

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror
      scalar_output_host;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>::HostMirror
      vector_output_host;

  Real nu_ratio;

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
  // tag for vlaplace_sphere_wk_cartesian_reduced
  struct TagVLaplaceCartesianReducedML {};
  // tag for vlaplace_sphere_wk_contra
  struct TagVLaplaceContraML {};
  // tag for vorticity_sphere
  struct TagVorticityVectorML {};
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
                          sphere_buf,
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
                               sphere_buf,
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
          laplace_simple(
              kv, dinv_d, spheremp_d, dvv_d, local_temp1_d,
              local_scalar_input_d, sphere_buf, local_scalar_output_d);
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
          laplace_tensor(kv, dinv_d, spheremp_d, dvv_d,
                         tensor_d, local_temp1_d,
                         local_scalar_input_d,
                         sphere_buf,
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
            scalar_input_COPY2_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]>
        local_scalar_output_d = Kokkos::subview(
            scalar_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_temp1_d = Kokkos::subview(
            temp1_d, _index, Kokkos::ALL, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    if(kv.ie == 0) {
      Kokkos::single(Kokkos::PerTeam(kv.team),
          [&]() {
            for(int igp = 0; igp < NP; ++igp) {
              for(int jgp = 0; jgp < NP; ++jgp) {
                printf("scalar input %d %d c % .17e\n",
                       igp, jgp,
                       local_scalar_input_d(igp, jgp, 0)[0]);
              }
            }
          });
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          laplace_tensor_replace(
              kv, dinv_d, spheremp_d, dvv_d, tensor_d,
              local_temp1_d, sphere_buf, local_scalar_input_d);

          // if team rank == 0? only 1 thread is allowed to
          // overwrite?
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                [&](const int idx) {
                  const int igp = idx / NP;
                  const int jgp = idx % NP;
                local_scalar_output_d(igp, jgp, level) =
                    local_scalar_input_d(igp, jgp, level);
              });
        });  // end of par_for for level

    if(kv.ie == 0) {
      Kokkos::single(Kokkos::PerTeam(kv.team),
          [&]() {
            for(int igp = 0; igp < NP; ++igp) {
              for(int jgp = 0; jgp < NP; ++jgp) {
                printf("scalar output %d %d c % .17e\n",
                       igp, jgp,
                       local_scalar_input_d(igp, jgp, 0)[0]);
              }
            }
          });
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
                                 sphere_buf,
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
          grad_sphere_wk_testcov(
              kv, d_d, mp_d, metinv_d, metdet_d, dvv_d,
              local_scalar_input_d, sphere_buf, local_vector_output_d);
        });  // end parallel_for for level

  }  // end of op() for grad_sphere_wk_testcov

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagVLaplaceCartesianReducedML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_vector_input_d = Kokkos::subview(
            vector_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_vector_output_d = Kokkos::subview(
            vector_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_temp1_d = Kokkos::subview(
            temp1_d, _index, Kokkos::ALL, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]> local_temp4_d =
        Kokkos::subview(temp4_d, _index, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]> local_temp5_d =
        Kokkos::subview(temp5_d, _index, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]> local_temp6_d =
        Kokkos::subview(temp6_d, _index, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          vlaplace_sphere_wk_cartesian_reduced(
              kv, dinv_d, spheremp_d, tensor_d,
              vec_sph2cart_d, dvv_d, local_temp1_d,
              local_temp4_d, local_temp5_d, local_temp6_d,
              sphere_buf,
              local_vector_input_d, local_vector_output_d);
        });  // end parallel_for for level

  }  // end of op() for laplace_tensor multil

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagVLaplaceContraML &,
                  TeamMember team) const {
    KernelVariables kv(team);
    int _index = team.league_rank();

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_vector_input_d = Kokkos::subview(
            vector_input_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_vector_output_d = Kokkos::subview(
            vector_output_d, _index, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_temp1_d = Kokkos::subview(
            temp1_d, _index, Kokkos::ALL, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[2][NP][NP][NUM_LEV]>
        local_temp2_d = Kokkos::subview(
            temp2_d, _index, Kokkos::ALL, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]> local_temp4_d =
        Kokkos::subview(temp4_d, _index, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);

    ExecViewManaged<Scalar[NP][NP][NUM_LEV]> local_temp5_d =
        Kokkos::subview(temp5_d, _index, Kokkos::ALL,
                        Kokkos::ALL, Kokkos::ALL);

    // don't forget to introduce nu_ratio
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(kv.team, NUM_LEV),
        [&](const int &level) {
          kv.ilev = level;
          vlaplace_sphere_wk_contra(
              kv, d_d, dinv_d, mp_d, spheremp_d, metinv_d,
              metdet_d, dvv_d, nu_ratio, local_temp4_d,
              local_temp5_d, local_temp1_d, local_temp2_d,
              sphere_buf,
              local_vector_input_d, local_vector_output_d);
        });  // end parallel_for for level

  }  // end of op() for laplace_tensor multil


  KOKKOS_INLINE_FUNCTION
  void operator()(const TagVorticityVectorML &,
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
          vorticity_sphere_vector(kv, d_d, metdet_d,
                               dvv_d, local_vector_input_d,
                               sphere_buf,
                               local_scalar_output_d);
        });  // end parallel_for for level

  }  // end of op() for vorticity_sphere_vector multilevel


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

  void run_functor_vlaplace_cartesian_reduced() const {
    Kokkos::TeamPolicy<ExecSpace,
                       TagVLaplaceCartesianReducedML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(vector_output_host, vector_output_d);
  };

  void run_functor_vlaplace_contra() const {
    Kokkos::TeamPolicy<ExecSpace, TagVLaplaceContraML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(vector_output_host, vector_output_d);
  };

  void run_functor_vorticity_sphere_vector() const {
    Kokkos::TeamPolicy<ExecSpace, TagVorticityVectorML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(scalar_output_host, scalar_output_d);
  };

};  // end of class def compute_sphere_op_test_ml

// SHMEM ????

TEST_CASE("Testing_gradient_sphere", "gradient_sphere") {
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

        // since i don't need flipping arrays, maybe, it is
        // enough to pass data() pointer?
        for(int _i = 0; _i < NP; _i++) {
          for(int _j = 0; _j < NP; _j++) {
            sf[_i][_j] = testing_grad_ml.scalar_input_host(
                _index, _i, _j, level)[v];
          }
        }

        // running F version of operator
        gradient_sphere_c_callable(
            &(sf[0][0]), testing_grad_ml.dvv_host.data(),
            &testing_grad_ml.dinv_host(_index, 0, 0, 0, 0),
            &(local_fortran_output[0][0][0]));

        // compare with the part from C run
        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            const Real coutput0 =
                testing_grad_ml.vector_output_host(
                    _index, 0, igp, jgp, level)[v];
            const Real coutput1 =
                testing_grad_ml.vector_output_host(
                    _index, 1, igp, jgp, level)[v];
            REQUIRE(!std::isnan(coutput0));
            REQUIRE(!std::isnan(coutput1));

            REQUIRE(!std::isnan(
                local_fortran_output[0][igp][jgp]));
            REQUIRE(!std::isnan(
                local_fortran_output[1][igp][jgp]));

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

TEST_CASE("Testing simple laplace_wk()", "laplace_wk") {
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

        laplace_simple_c_callable(
            &(sf[0][0]), &(dvvf[0][0]),
            &(dinvf[0][0][0][0]), &(sphf[0][0]),
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

  std::cout
      << "test laplace_simple multilevel finished. \n";

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
            sphf[_i][_j] =
                testing_tensor_laplace.spheremp_host(
                    _index, _i, _j);
            dvvf[_i][_j] =
                testing_tensor_laplace.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++) {
                dinvf[_d1][_d2][_i][_j] =
                    testing_tensor_laplace.dinv_host(
                        _index, _d1, _d2, _i, _j);

                tensorf[_d1][_d2][_i][_j] =
                    testing_tensor_laplace.tensor_host(
                        _index, _d1, _d2, _i, _j);

              }  // end of d2 loop
          }

        Real _hp = 0.0;
        Real _hs = 1.0;
        bool _vc = true;

        laplace_sphere_wk_c_callable(
            &(sf[0][0]), &(dvvf[0][0]),
            &(dinvf[0][0][0][0]), &(sphf[0][0]),
            &(tensorf[0][0][0][0]), _hp, _hs, _vc,
            &(local_fortran_output[0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 =
                testing_tensor_laplace.scalar_output_host(
                    _index, igp, jgp, level)[v];

            // std::cout << igp << "," << jgp << " F output
            // = " <<  local_fortran_output[igp][jgp] << ", C
            // output" << coutput0 << "\n";

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

  std::cout
      << "test laplace_tensor multilevel finished. \n";

}  // end of test laplace_tensor multilevel

TEST_CASE("Testing laplace_tensor_replace() multilevel",
          "laplace_tensor_replace") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_tensor_laplace(
      elements);
  testing_tensor_laplace
      .run_functor_tensor_laplace_replace();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real local_fortran_output[NP][NP];
        Real sf[NP][NP];
        Real dvvf[NP][NP];
        Real sphf[NP][NP];

        for(int _i = 0; _i < NP; _i++) {
          for(int _j = 0; _j < NP; _j++) {

            sf[_i][_j] =
                testing_tensor_laplace.scalar_input_host(
                    _index, _i, _j, level)[v];

            sphf[_i][_j] =
                testing_tensor_laplace.spheremp_host(
                    _index, _i, _j);
            dvvf[_i][_j] =
                testing_tensor_laplace.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++) {
              for(int _d2 = 0; _d2 < 2; _d2++) {
                printf("tensor %d %d %d %d f % .17e\n",
                       _d1, _d2, _i, _j,
                       testing_tensor_laplace.tensor_host(_index, _d1, _d2, _i, _j));
              }  // end of d2 loop
            }
          }
        }
        for(int _i = 0; _i < NP; _i++) {
          for(int _j = 0; _j < NP; _j++) {
            printf("scalar input %d %d f % .17e\n",
                   _i, _j,
                   testing_tensor_laplace.scalar_input_host(_index, _i, _j, level)[v]);
          }
        }

        Kokkos::deep_copy(testing_tensor_laplace.scalar_input_host,
                          testing_tensor_laplace.scalar_input_d);
        for(int _i = 0; _i < NP; _i++) {
          for(int _j = 0; _j < NP; _j++) {
            printf("scalar input %d %d f % .17e\n",
                   _i, _j,
                   testing_tensor_laplace.scalar_input_host(_index, _i, _j, level)[v]);
          }
        }

        Real _hp = 0.0;
        Real _hs = 1.0;
        bool _vc = true;

        laplace_sphere_wk_c_callable(
            &(sf[0][0]), &(dvvf[0][0]),
            &testing_tensor_laplace.dinv_host(_index, 0, 0, 0, 0),
            &(sphf[0][0]),
            &testing_tensor_laplace.tensor_host(_index, 0, 0, 0, 0),
            _hp, _hs, _vc,
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

  std::cout << "test laplace_tensor_replace multilevel "
               "finished. \n";

}  // end of test laplace_tensor_replace multilevel

TEST_CASE("Testing curl_sphere_wk_testcov() multilevel",
          "curl_sphere_wk_testcov") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_curl(elements);
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
            sf[_i][_j] = testing_curl.scalar_input_host(
                _index, _i, _j, level)[v];
            mpf[_i][_j] =
                testing_curl.mp_host(_index, _i, _j);
            dvvf[_i][_j] = testing_curl.dvv_host(_i, _j);

            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++) {
                df[_d1][_d2][_i][_j] = testing_curl.d_host(
                    _index, _d1, _d2, _i, _j);

              }  // end of d2 loop
          }
        curl_sphere_wk_testcov_c_callable(
            &(sf[0][0]), &(dvvf[0][0]), &(df[0][0][0][0]),
            &(mpf[0][0]), &(local_fortran_output[0][0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 = testing_curl.vector_output_host(
                _index, 0, igp, jgp, level)[v];

            Real coutput1 = testing_curl.vector_output_host(
                _index, 1, igp, jgp, level)[v];

            /*
            std::cout << igp << "," << jgp << " F output0  =
            " << local_fortran_output[0][igp][jgp] << ", C
            output0 = " << coutput0 << "\n"; std::cout <<
            "difference=" <<
            local_fortran_output[0][igp][jgp] - coutput0 <<
            "\n"; std::cout << "rel difference=" <<
            (local_fortran_output[0][igp][jgp] -
            coutput0)/coutput0 << "\n";

            std::cout << "difference=" <<
            local_fortran_output[1][igp][jgp] - coutput1 <<
            "\n"; std::cout << "rel difference=" <<
            (local_fortran_output[1][igp][jgp] -
            coutput1)/coutput1 << "\n";

            std::cout << "epsilon = " <<
            std::numeric_limits<Real>::epsilon() << "\n";
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

  std::cout << "test curl_sphere_wk_testcov multilevel "
               "finished. \n";

}  // end of test laplace_tensor multilevel

TEST_CASE("Testing grad_sphere_wk_testcov() multilevel",
          "grad_sphere_wk_testcov") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_grad(elements);
  testing_grad.run_functor_grad_sphere_wk_testcov();

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
            sf[_i][_j] = testing_grad.scalar_input_host(
                _index, _i, _j, level)[v];
            mpf[_i][_j] =
                testing_grad.mp_host(_index, _i, _j);
            metdetf[_i][_j] =
                testing_grad.metdet_host(_index, _i, _j);
            dvvf[_i][_j] = testing_grad.dvv_host(_i, _j);

            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++) {
                df[_d1][_d2][_i][_j] = testing_grad.d_host(
                    _index, _d1, _d2, _i, _j);

                metinvf[_d1][_d2][_i][_j] =
                    testing_grad.metinv_host(_index, _d1,
                                             _d2, _i, _j);
              }  // end of d2 loop
          }
        gradient_sphere_wk_testcov_c_callable(
            &(sf[0][0]), &(dvvf[0][0]),
            &(metinvf[0][0][0][0]), &(metdetf[0][0]),
            &(df[0][0][0][0]), &(mpf[0][0]),
            &(local_fortran_output[0][0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 = testing_grad.vector_output_host(
                _index, 0, igp, jgp, level)[v];

            Real coutput1 = testing_grad.vector_output_host(
                _index, 1, igp, jgp, level)[v];

            /*
             * std::cout << igp << "," << jgp << " F output0
             * = " << local_fortran_output[0][igp][jgp] <<
             * ", C output0 = " << coutput0 << "\n";
             * std::cout << "difference=" <<
             * local_fortran_output[0][igp][jgp] - coutput0
             * << "\n"; std::cout << "rel difference=" <<
             * (local_fortran_output[0][igp][jgp] -
             * coutput0)/coutput0 << "\n";
             *
             * std::cout << "difference=" <<
             * local_fortran_output[1][igp][jgp] - coutput1
             * << "\n"; std::cout << "rel difference=" <<
             * (local_fortran_output[1][igp][jgp] -
             * coutput1)/coutput1 << "\n";
             *
             * std::cout << "epsilon = " <<
             * std::numeric_limits<Real>::epsilon() << "\n";
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

  std::cout << "test grad_sphere_wk_testcov multilevel "
               "finished. \n";

}  // end of test laplace_tensor multilevel

TEST_CASE(
    "Testing vlaplace_sphere_wk_cartesian() multilevel",
    "vlaplace_sphere_wk_cartesian") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_vlaplace(
      elements);
  testing_vlaplace.run_functor_vlaplace_cartesian_reduced();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real local_fortran_output[2][NP][NP];
        Real vf[2][NP][NP];  // input
        Real dvvf[NP][NP];
        Real dinvf[2][2][NP][NP];
        Real tensorf[2][2][NP][NP];
        Real vec_sph2cartf[2][3][NP][NP];
        Real sphf[NP][NP];

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            vf[0][_i][_j] =
                testing_vlaplace.vector_input_host(
                    _index, 0, _i, _j, level)[v];
            vf[1][_i][_j] =
                testing_vlaplace.vector_input_host(
                    _index, 1, _i, _j, level)[v];

            sphf[_i][_j] = testing_vlaplace.spheremp_host(
                _index, _i, _j);
            dvvf[_i][_j] =
                testing_vlaplace.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++) {
                dinvf[_d1][_d2][_i][_j] =
                    testing_vlaplace.dinv_host(_index, _d1,
                                               _d2, _i, _j);

                tensorf[_d1][_d2][_i][_j] =
                    testing_vlaplace.tensor_host(
                        _index, _d1, _d2, _i, _j);

              }  // end of d2 loop
            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 3; _d2++) {
                vec_sph2cartf[_d1][_d2][_i][_j] =
                    testing_vlaplace.vec_sph2cart_host(
                        _index, _d1, _d2, _i, _j);
              }  // end of d2 loop
          }      // end of j loop

        Real _hp = 0.0;
        Real _hs = 1.0;
        bool _vc = true;

        vlaplace_sphere_wk_cartesian_c_callable(
            &(vf[0][0][0]), &(dvvf[0][0]),
            &(dinvf[0][0][0][0]), &(sphf[0][0]),
            &(tensorf[0][0][0][0]),
            &(vec_sph2cartf[0][0][0][0]), _hp, _hs, _vc,
            &(local_fortran_output[0][0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 =
                testing_vlaplace.vector_output_host(
                    _index, 0, igp, jgp, level)[v];
            Real coutput1 =
                testing_vlaplace.vector_output_host(
                    _index, 1, igp, jgp, level)[v];

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

  std::cout << "test vlaplace_sphere_wk_cartesian "
               "multilevel finished. \n";

}  // end of test laplace_sphere_wk_contra multilevel

TEST_CASE("Testing vlaplace_sphere_wk_contra() multilevel",
          "vlaplace_sphere_wk_contra") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_vlaplace(
      elements);
  testing_vlaplace.run_functor_vlaplace_contra();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real local_fortran_output[2][NP][NP];
        Real vf[2][NP][NP];  // input
        Real dvvf[NP][NP];
        Real df[2][2][NP][NP];
        Real dinvf[2][2][NP][NP];
        Real mpf[NP][NP];
        Real sphf[NP][NP];
        Real metinvf[2][2][NP][NP];
        Real metdetf[NP][NP];
        Real rmetdetf[NP][NP];
        // let's test with 1 first, then we need to test
        // with random...
        Real nu_ratio = testing_vlaplace.nu_ratio;

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            vf[0][_i][_j] =
                testing_vlaplace.vector_input_host(
                    _index, 0, _i, _j, level)[v];
            vf[1][_i][_j] =
                testing_vlaplace.vector_input_host(
                    _index, 1, _i, _j, level)[v];

            mpf[_i][_j] =
                testing_vlaplace.mp_host(_index, _i, _j);
            sphf[_i][_j] = testing_vlaplace.spheremp_host(
                _index, _i, _j);
            metdetf[_i][_j] = testing_vlaplace.metdet_host(
                _index, _i, _j);
            rmetdetf[_i][_j] = 1.0 / metdetf[_i][_j];

            dvvf[_i][_j] =
                testing_vlaplace.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++)
              for(int _d2 = 0; _d2 < 2; _d2++) {
                df[_d1][_d2][_i][_j] =
                    testing_vlaplace.d_host(_index, _d1,
                                            _d2, _i, _j);
                dinvf[_d1][_d2][_i][_j] =
                    testing_vlaplace.dinv_host(_index, _d1,
                                               _d2, _i, _j);

                metinvf[_d1][_d2][_i][_j] =
                    testing_vlaplace.metinv_host(
                        _index, _d1, _d2, _i, _j);
              }  // end of d2 loop
          }      // end of j loop

        vlaplace_sphere_wk_contra_c_callable(
            &(vf[0][0][0]), &(dvvf[0][0]),
            &(df[0][0][0][0]), &(dinvf[0][0][0][0]),
            &(mpf[0][0]), &(sphf[0][0]),
            &(metinvf[0][0][0][0]), &(metdetf[0][0]),
            &(rmetdetf[0][0]), nu_ratio,
            &(local_fortran_output[0][0][0]));

        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 =
                testing_vlaplace.vector_output_host(
                    _index, 0, igp, jgp, level)[v];
            Real coutput1 =
                testing_vlaplace.vector_output_host(
                    _index, 1, igp, jgp, level)[v];
            // std::cout << igp << "," << jgp << " F output0
            // = " <<  local_fortran_output[0][igp][jgp] << ",
            // C output0 = " << coutput0 << "\n";
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

  std::cout << "test vlaplace_sphere_wk_contra multilevel "
               "finished. \n";

}  // end of test vlaplace_contra multilevel

TEST_CASE("Testing vorticity_sphere_vector()",
          "vorticity_sphere_vector") {
  constexpr const Real rel_threshold =
      1E-15;  // let's move this somewhere in *hpp?
  constexpr const int elements = 10;

  compute_sphere_operator_test_ml testing_vort(elements);
  testing_vort.run_functor_vorticity_sphere_vector();

  for(int _index = 0; _index < elements; _index++) {
    for(int level = 0; level < NUM_LEV; ++level) {
      for(int v = 0; v < VECTOR_SIZE; ++v) {
        Real local_fortran_output[NP][NP];
        Real vf[2][NP][NP];
        Real dvvf[NP][NP];
        Real df[2][2][NP][NP];
        Real metdetf[NP][NP];

        for(int _i = 0; _i < NP; _i++)
          for(int _j = 0; _j < NP; _j++) {
            metdetf[_i][_j] = testing_vort.metdet_host(
                _index, _i, _j);
            dvvf[_i][_j] = testing_vort.dvv_host(_i, _j);
            for(int _d1 = 0; _d1 < 2; _d1++) {
              vf[_d1][_i][_j] =
                  testing_vort.vector_input_host(
                      _index, _d1, _i, _j, level)[v];
              for(int _d2 = 0; _d2 < 2; _d2++)
                df[_d1][_d2][_i][_j] =
                    testing_vort.d_host(_index, _d1,
                                             _d2, _i, _j);
            }
          }
        vorticity_sphere_c_callable(
            &(vf[0][0][0]), &(dvvf[0][0]), &(metdetf[0][0]),
            &(df[0][0][0][0]),
            &(local_fortran_output[0][0]));
        for(int igp = 0; igp < NP; ++igp) {
          for(int jgp = 0; jgp < NP; ++jgp) {
            Real coutput0 =
                testing_vort.scalar_output_host(
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

  std::cout << "test vorticity_sphere_vector multilevel finished. \n";

}  // end of test div_sphere_wk_ml











