#include <catch/catch.hpp>

#include <limits>

#include <CaarControl.hpp>
#include <CaarFunctor.hpp>
#include <CaarRegion.hpp>
#include <Dimensions.hpp>
#include <Types.hpp>

#include <assert.h>
#include <stdio.h>
#include <random>



using namespace Homme;

using rngAlg = std::mt19937_64;

extern "C" {

void caar_compute_energy_grad_c_int(const Real (&dvv)[NP][NP], Real *Dinv,
                                    Real *const &pecnd, Real *const &phi,
                                    Real *const &velocity,
                                    Real (&vtemp)[2][NP][NP]);
//what is _int here for?
//why dvv is passed as real&[][], Dinv is real* , and phi as *const &?
/*
void laplace_simple_c_int(const Real * scalar_field,
                          const Real * dvv,
                          const Real * dinv,
                          const Real * metdet,
                                Real (&laplace)[NP][NP]);
*/
void laplace_simple_c_int(const Real (&scalar_field)[NP][NP],
                          const Real (&dvv)[NP][NP],
                          const Real (&dinv)[NP][NP][2][2],
                          const Real (&metdet)[NP][NP],
                                Real (&laplace)[NP][NP]);
/*
void gradient_sphere_c_callable(const Real (&s)[NP][NP],
                                const Real (&dvv)[NP][NP],
                                const Real (&dinv)[NP][NP][2][2],
                                Real (&grad)[NP][NP][2]);
*/

void gradient_sphere_c_callable(const Real* ,
                                const Real* ,
                                const Real* ,
                                Real*);




}//extern C

Real compare_answers(Real target, Real computed, Real relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

std::cout<< "in compute answers " << target << " and " << computed <<"\n";

  return std::fabs(target - computed) / denom;
}//compare_answers

void genRandArray(Real *arr, int arr_len, rngAlg &engine, std::uniform_real_distribution<Real> pdf){
  for(int i = 0; i < arr_len; ++i) {
    arr[i] = pdf(engine);
  }
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
                vector_output_1(kv.ie, kv.ilev, dim, igp, jgp) =
                    kv.vector_buf_1(dim, igp, jgp);
                vector_output_2(kv.ie, kv.ilev, dim, igp, jgp) =
                    kv.vector_buf_2(dim, igp, jgp);
              }
              scalar_output_1(kv.ie, kv.ilev, igp, jgp) =
                  kv.scalar_buf_1(igp, jgp);
              scalar_output_2(kv.ie, kv.ilev, igp, jgp) =
                  kv.scalar_buf_2(igp, jgp);
            }
          }
        });
  }

  void run_functor() const {
    Kokkos::TeamPolicy<ExecSpace> policy(functor.m_data.num_elems, 16, 4);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    Kokkos::deep_copy(vector_output_1, vector_results_1);
    Kokkos::deep_copy(vector_output_2, vector_results_2);
    Kokkos::deep_copy(scalar_output_1, scalar_results_1);
    Kokkos::deep_copy(scalar_output_2, scalar_results_2);
  }
//device
  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]> vector_results_1;
  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]> vector_results_2;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]> scalar_results_1;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]> scalar_results_2;
//host mirrors
  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]>::HostMirror vector_output_1;
  ExecViewManaged<Real * [NUM_LEV][2][NP][NP]>::HostMirror vector_output_2;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]>::HostMirror scalar_output_1;
  ExecViewManaged<Real * [NUM_LEV][NP][NP]>::HostMirror scalar_output_2;

  CaarFunctor functor;

//host
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
};//compute_subfunctor_test


/*
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

  class compute_energy_grad_test {
  public:
    KOKKOS_INLINE_FUNCTION
    static void test_functor(const CaarFunctor &functor,
                             CaarFunctor::KernelVariables &kv) {
      functor.compute_energy_grad(kv);
    }
  };
  compute_subfunctor_test<compute_energy_grad_test> test_functor(num_elems);

  test_functor.run_functor();

  Real vtemp[2][NP][NP];

  for (int ie = 0; ie < num_elems; ++ie) {
    for (int level = 0; level < NUM_LEV; ++level) {
//pressure is not used here
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
}//end of TEST_CASE(...,"compute_energy_grad")
*/

/*
  subroutine laplace_simple_c_int(s,dvv,dinv,metdet,laplace) bind(c)
    use kinds, only: real_kind
    use dimensions_mod, only: np

    real(kind=real_kind), intent(in) :: s(np,np)
    real(kind=real_kind), intent(in) :: dvv(np,np)
    real(kind=real_kind), intent(in) :: dinv(np,np,2,2)
    real(kind=real_kind), intent(in) :: metdet(np, np)
    real(kind=real_kind), intent(out):: laplace(np,np)
*/


//template <typename TestFunctor_T> class compute_sphop_test {
class compute_sphop_test {
public:
  compute_sphop_test(int some_index)
    : scalar_input_d("scalar input", some_index),
      dinv_d("dinv", some_index),
      metdet_d("metdet", some_index),
      dvv_d("dvv", some_index),
      scalar_output_d("scalar output", some_index),
      vector_output_d("vector output", some_index),
      temp1_d("temp1", some_index),
      temp2_d("temp2", some_index),
      temp3_d("temp3", some_index),
      scalar_input_host(Kokkos::create_mirror_view(scalar_input_d)),
      dinv_host(Kokkos::create_mirror_view(dinv_d)),
      metdet_host(Kokkos::create_mirror_view(metdet_d)),
      dvv_host(Kokkos::create_mirror_view(dvv_d)),
      scalar_output_host(Kokkos::create_mirror_view(scalar_output_d)),
      vector_output_host(Kokkos::create_mirror_view(vector_output_d)),
      temp1_host(Kokkos::create_mirror_view(temp1_d)),
      temp2_host(Kokkos::create_mirror_view(temp2_d)),
      temp3_host(Kokkos::create_mirror_view(temp3_d)),
      _some_index(some_index)
       {

//constructor s body
//init randonly

  std::random_device rd;
  rngAlg engine(rd());

//for dimensions use product dim0*...

//check singularities?
  genRandArray(scalar_input_host.data(), scalar_input_len*_some_index, engine, std::uniform_real_distribution<Real>(0, 100.0));
  genRandArray(dinv_host.data(), dinv_len*_some_index, engine, std::uniform_real_distribution<Real>(0, 1.0));
//  genRandArray(metdet_host.data(), metdet_len*_some_index, engine, std::uniform_real_distribution<Real>(0, 1.0));
  genRandArray(dvv_host.data(), dvv_len*_some_index, engine, std::uniform_real_distribution<Real>(0, 1.0));

for(int i1=0; i1<_some_index; i1++)
for(int i2=0; i2<NP; i2++)
for(int i3=0; i3<NP; i3++){
//dinv_host(i1,0,0,i2,i3)=1.0;
//dinv_host(i1,1,1,i2,i3)=1.0;
//dinv_host(i1,1,0,i2,i3)=1.0;
//dinv_host(i1,0,1,i2,i3)=1.0;

//dvv_host(i1,i2,i3)=1.0;

//Real aa = i2+i3;

//scalar_input_host(i1,i2,i3) = aa;

}


  }
  int _some_index;//league size

//device views
  ExecViewManaged<Real * [NP][NP]> scalar_input_d;
  ExecViewManaged<Real * [2][2][NP][NP]> dinv_d;
  ExecViewManaged<Real * [NP][NP]> metdet_d;
  ExecViewManaged<Real * [NP][NP]> dvv_d;
  ExecViewManaged<Real * [NP][NP]> scalar_output_d;
  ExecViewManaged<Real * [2][NP][NP]> vector_output_d;
  ExecViewManaged<Real * [2][NP][NP]> temp1_d, temp2_d, temp3_d;  

//host views, one dim is some_index. sp ops do not take ie or nlev fields, but to make it 
//more reasonable test and to have parallel_for
  ExecViewManaged<Real * [NP][NP]>::HostMirror scalar_input_host;
//how to get total length of view?
  const int scalar_input_len = NP*NP;
  ExecViewManaged<Real * [2][2][NP][NP]>::HostMirror dinv_host;
  const int dinv_len = 2*2*NP*NP;
  ExecViewManaged<Real * [NP][NP]>::HostMirror metdet_host;
  const int metdet_len = NP*NP;
  ExecViewManaged<Real * [NP][NP]>::HostMirror dvv_host;
  const int dvv_len = NP*NP;
  ExecViewManaged<Real * [NP][NP]>::HostMirror scalar_output_host;
  ExecViewManaged<Real * [2][NP][NP]>::HostMirror vector_output_host;
  ExecViewManaged<Real * [2][NP][NP]>::HostMirror temp1_host, temp2_host, temp3_host;

//tag for simple laplace
  struct TagSimpleLaplace{};
//tag for gradient_sphere
  struct TagGradientSphere{};

  KOKKOS_INLINE_FUNCTION
  void operator()( const TagSimpleLaplace &, TeamMember team) const {

        int _index = team.league_rank();

        ExecViewManaged<Real [NP][NP]> local_scalar_input_d = 
          Kokkos::subview(scalar_input_d, _index, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [NP][NP]> local_dvv_d = 
          Kokkos::subview(dvv_d, _index, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [2][2][NP][NP]> local_dinv_d = 
          Kokkos::subview(dinv_d, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [NP][NP]> local_metdet_d =
          Kokkos::subview(metdet_d, _index, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [2][NP][NP]> local_temp1_d = 
          Kokkos::subview(temp1_d, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [2][NP][NP]> local_temp2_d =
          Kokkos::subview(temp2_d, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [2][NP][NP]> local_temp3_d =
          Kokkos::subview(temp3_d, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [NP][NP]> local_scalar_output_d = 
          Kokkos::subview(scalar_output_d, _index, Kokkos::ALL, Kokkos::ALL);
        
        laplace_wk(team, local_scalar_input_d, local_dvv_d, local_dinv_d,
                         local_metdet_d, local_temp1_d, local_temp2_d, local_temp3_d, local_scalar_output_d);

      };

//this could be even nicer, put in a param in run_functor(param) to only branch 
//policy type
  void run_functor_simple_laplace() const {
//league, team, vector_length_request=1
    Kokkos::TeamPolicy<ExecSpace, TagSimpleLaplace> policy(_some_index, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    //TO FROM 
    Kokkos::deep_copy(scalar_output_host, scalar_output_d);
  };

  void run_functor_gradient_sphere() const {
//league, team, vector_length_request=1
    Kokkos::TeamPolicy<ExecSpace, TagGradientSphere> policy(_some_index, 16);
    Kokkos::parallel_for(policy, *this);
    ExecSpace::fence();
    //TO FROM 
    Kokkos::deep_copy(vector_output_host, vector_output_d);
  };

  KOKKOS_INLINE_FUNCTION
  void operator()( const TagGradientSphere &, TeamMember team) const {
    //here is a flaw
    //grad is :
    //gradient_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
    //                const ExecViewUnmanaged<const Real[NP][NP]> scalar,
    //                const ExecViewUnmanaged<const Real[NP][NP]> dvv,
    //                const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
    //                ExecViewUnmanaged<Real[2][NP][NP]> temp_v,
    //                ExecViewUnmanaged<Real[2][NP][NP]> grad_s)
    //
        int _index = team.league_rank();
        
        ExecViewManaged<Real [NP][NP]> local_scalar_input_d =  
          Kokkos::subview(scalar_input_d, _index, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [NP][NP]> local_dvv_d =  
          Kokkos::subview(dvv_d, _index, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [2][2][NP][NP]> local_dinv_d = 
          Kokkos::subview(dinv_d, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [2][NP][NP]> local_temp1_d = 
          Kokkos::subview(temp1_d, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        ExecViewManaged<Real [2][NP][NP]> local_vector_output_d =  
          Kokkos::subview(vector_output_d, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        
        gradient_sphere(team, local_scalar_input_d, local_dvv_d, local_dinv_d,
                              local_temp1_d, local_vector_output_d);
      
      };




}; //end of compute_sphop_test

/*
TEST_CASE("Testing laplace_simple()", "laplace_simple") {

 constexpr const Real rel_threshold = 1E-15;//let's move this somewhere in *hpp?
 constexpr const int some_index = 1;

 compute_sphop_test testing_laplace(some_index);

//result will be in testing_grad.vector_output_host
 testing_laplace.run_functor_simple_laplace();

 HostViewManaged<Real * [NP][NP]> fortran_scalar_output("Fortran scalar output", some_index);
 Real loc_f_out[NP][NP];
//cannot put fortran testing in the class... enough problems with cpp functors 
//run fortran
 for(int _index = 0; _index < some_index; _index++){
   HostViewManaged<Real [NP][NP]> local_scalar_input = 
     Kokkos::subview(testing_laplace.scalar_input_host, _index, Kokkos::ALL, Kokkos::ALL);
   
   HostViewManaged<Real [NP][NP]> local_dvv =
     Kokkos::subview(testing_laplace.dvv_host, _index, Kokkos::ALL, Kokkos::ALL);
  
   HostViewManaged<Real [2][2][NP][NP]> local_dinv =
     Kokkos::subview(testing_laplace.dinv_host, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

   HostViewManaged<Real [NP][NP]> local_metdet =
     Kokkos::subview(testing_laplace.metdet_host, _index, Kokkos::ALL, Kokkos::ALL);

   Real sf[NP][NP];
   Real dvvf[NP][NP];
   Real dinvf[NP][NP][2][2];
   Real metf[NP][NP];

   for(int _i = 0; _i < NP; _i++)
      for(int _j = 0; _j < NP; _j++){
   
        sf[_i][_j] = local_scalar_input(_i,_j);
        dvvf[_i][_j] = local_dvv(_i,_j);
        metf[_i][_j] = local_metdet(_i,_j);
        for(int _d1 = 0; _d1 < 2; _d1++)
           for(int _d2 = 0; _d2 < 2; _d2++)
              dinvf[_i][_j][_d1][_d2] = local_dinv(_d1,_d2,_i,_j);
   }

//old call, arrays need to be flipped
//   laplace_simple_c_int(local_scalar_input.data(), local_dvv.data(),
//                        local_dinv.data(), local_metdet.data(), loc_f_out);

   laplace_simple_c_int(sf, dvvf, dinvf, metf, loc_f_out);

   for(int _i = 0; _i < NP; _i++)
      for(int _j = 0; _j < NP; _j++){

//std::cout << "i = " << _i << ", j= " << _j << "\n";
//std::cout << "scalar input from C " << testing_grad.scalar_input_host(_index,_i,_j) << "\n";
//std::cout << "scalar input from F " << sf[_i][_j] << "\n";


}
//         fortran_scalar_output(_index, _j, _i) = loc_f_out[_i][_j];

//now compare answers
   for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {

std::cout << "i = " << igp << ", j= " << jgp << "\n";
std::cout << "F = " << loc_f_out[igp][jgp] << ", C = " << testing_laplace.scalar_output_host(_index, igp, jgp) <<"\n";
}
}
   for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {

          REQUIRE(!std::isnan(loc_f_out[igp][jgp]));
          REQUIRE(!std::isnan(
              testing_laplace.scalar_output_host(_index, jgp, igp)));
          REQUIRE(std::numeric_limits<Real>::epsilon() >=
              compare_answers(
                        loc_f_out[jgp][igp],
                        testing_laplace.scalar_output_host(_index, jgp, igp)));
      }
    }

 }
};//end of TEST_CASE(..., "simple laplace")
*/


TEST_CASE("Testing gradient_sphere()", "gradient_sphere") {

 constexpr const Real rel_threshold = 1E-15;//let's move this somewhere in *hpp?
 constexpr const int some_index = 1;

 compute_sphop_test testing_grad(some_index);

 testing_grad.run_functor_gradient_sphere();

//not used
 Real loc_f_out[2][NP][NP];

 for(int _index = 0; _index < some_index; _index++){
   HostViewManaged<Real [NP][NP]> local_scalar_input =
     Kokkos::subview(testing_grad.scalar_input_host, _index, Kokkos::ALL, Kokkos::ALL);

   HostViewManaged<Real [NP][NP]> local_dvv =
     Kokkos::subview(testing_grad.dvv_host, _index, Kokkos::ALL, Kokkos::ALL);

   HostViewManaged<Real [2][2][NP][NP]> local_dinv =
     Kokkos::subview(testing_grad.dinv_host, _index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

   Real sf[NP][NP];
   Real dvvf[NP][NP];
   Real dinvf[2][2][NP][NP];

//flipping arrays -- i dont know how to do it here
   for( int _i = 0; _i < NP; _i++)
      for(int _j = 0; _j < NP; _j++){

        sf[_j][_i] = local_scalar_input(_i,_j);
        dvvf[_j][_i] = local_dvv(_i,_j);

        for(int _d1 = 0; _d1 < 2; _d1++)
           for(int _d2 = 0; _d2 < 2; _d2++)

              dinvf[_d2][_d1][_j][_i] = local_dinv(_d1,_d2,_i,_j);
   }

   gradient_sphere_c_callable(&(sf[0][0]), &(dvvf[0][0]), &(dinvf[0][0][0][0]), &(loc_f_out[0][0][0]));


   for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
       //  for (int _d = 0; _d < 2; ++_d){
        
std::cout << "here i= " << igp << ", j= " << jgp << "\n";
std::cout << "F real* = " << loc_f_out[0][jgp][igp] <<  ", " << loc_f_out[1][jgp][igp] <<"\n";
std::cout << "C = " << testing_grad.vector_output_host(_index, 0, igp, jgp) <<
", " << testing_grad.vector_output_host(_index, 1, igp, jgp) << "\n";
//}   
}   
}
   for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
         for (int _d = 0; _d < 2; ++_d){
          REQUIRE(!std::isnan(loc_f_out[_d][jgp][igp]));
          REQUIRE(!std::isnan(testing_grad.vector_output_host(_index, _d, igp, jgp)));
          REQUIRE(std::numeric_limits<Real>::epsilon() >=
              compare_answers(
                        loc_f_out[_d][igp][jgp],
                        testing_grad.vector_output_host(_index, _d, jgp, igp)));
      }
    }
  }
 }
};//end of TEST_CASE(..., "gradient_sphere")





