#include <catch/catch.hpp>


//?
#include <limits>

#include "Dimensions.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"

#include "utils_flat_ut.cpp"

#include <assert.h>
#include <stdio.h>
#include <random>

using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

void compute_ppm_grids_c_callable(
const Real * dx,
Real rslt,
int alg);

}  // extern C

class remap_test {
 public:
  compute_sphere_operator_test_ml(int num_elems)
      : scalar_input_d("scalar input", num_elems),
        vector_input_d("vector input", num_elems),
        scalar_output_d("scalar output", num_elems),
        vector_output_d("vector output", num_elems),
        scalar_input_host(
            Kokkos::create_mirror_view(scalar_input_d)),
        vector_input_host(
            Kokkos::create_mirror_view(vector_input_d)),
        scalar_output_host(
            Kokkos::create_mirror_view(scalar_output_d)),
        vector_output_host(
            Kokkos::create_mirror_view(vector_output_d)),
        _num_elems(num_elems) {
    std::random_device rd;
    rngAlg engine(rd());
    genRandArray(
        reinterpret_cast<Real *>(scalar_input_host.data()),
        scalar_input_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-1000.0,
                                             1000.0));
    genRandArray(
        reinterpret_cast<Real *>(vector_input_host.data()),
        vector_input_len * _num_elems, engine,
        std::uniform_real_distribution<Real>(-1000.0,
                                             1000.0));
// setting everything to 1 is good for debugging
#if 0
for(int i1=0; i1<_num_elems; i1++)
for(int i2=0; i2<NP; i2++)
for(int i3=0; i3<NP; i3++){
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
#endif

  }  // end of constructor

  int _num_elems;  // league size, serves as ie index

  // device
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>
      scalar_input_d;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>
      vector_input_d;
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>
      scalar_output_d;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>
      vector_output_d;
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

  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror
      scalar_output_host;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>::HostMirror
      vector_output_host;

  // tag for laplace_simple()
  struct TagSimpleLaplaceML {};
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

  void run_functor_gradient_sphere() const {
    // league, team, vector_length_request=1
    Kokkos::TeamPolicy<ExecSpace, TagGradientSphereML>
        policy(_num_elems, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    // TO FROM
    Kokkos::deep_copy(vector_output_host, vector_output_d);
  };


};  // end of class def compute_sphere_op_test_ml

// SHMEM ????

TEST_CASE("Testing gradient_sphere()", "gradient_sphere") {
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




