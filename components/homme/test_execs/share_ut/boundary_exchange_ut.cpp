#include <catch/catch.hpp>

#include "Context.hpp"
#include "BuffersManager.hpp"
#include "BoundaryExchange.hpp"
#include "Connectivity.hpp"
#include "Utility.hpp"
#include "Types.hpp"

#include <random>
#include <iomanip>

using namespace Homme;

extern "C" {

void initmp_f90 ();
void init_cube_geometry_f90 (const int& ne);
void init_connectivity_f90 (const int& num_scalar_fields_2d, const int& num_scalar_fields_3d,
                            const int& num_vector_fields_3d, const int& vector_dim);
void cleanup_f90 ();
void boundary_exchange_test_f90 (F90Ptr& field_2d_ptr, F90Ptr& field_3d_ptr, F90Ptr& field_4d_ptr,
                                 const int& inner_dim_4d, const int& num_time_levels,
                                 const int& idim_2d, const int& idim_3d, const int& idim_4d);

} // extern "C"

// =========================== TESTS ============================ //

TEST_CASE ("Boundary Exchange", "Testing the boundary exchange framework")
{
  //std::random_device rd;
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dreal(-1.0, 1.0);

  constexpr int ne        = 2;
  //constexpr int nfaces    = 6;
  //constexpr int num_elems = nfaces*ne*ne;
  constexpr int num_tests = 1;
  constexpr int DIM       = 2;
  constexpr double test_tolerance = 1e-13;
  constexpr int num_scalar_fields_2d = 1;
  constexpr int num_scalar_fields_3d = 1;
  constexpr int num_vector_fields_3d = 1;
  constexpr int field_2d_idim = 0;
  constexpr int field_3d_idim = 1;
  constexpr int field_4d_outer_idim = 2;

  // Initialize f90 mpi stuff
  initmp_f90();

  // Create cube geometry
  init_cube_geometry_f90(ne);

  // Create connectivity
  init_connectivity_f90(num_scalar_fields_2d, num_scalar_fields_3d, num_vector_fields_3d, DIM);
  std::shared_ptr<Connectivity> connectivity = Context::singleton().get_connectivity();

  // Retrieve local number of elements
  int num_elements = connectivity->get_num_elements();
  int rank = connectivity->get_comm().m_rank;

  // Create input data arrays
  HostViewManaged<Real*[NUM_TIME_LEVELS][NP][NP]> field_2d_f90("", num_elements);
  ExecViewManaged<Real*[NUM_TIME_LEVELS][NP][NP]> field_2d_cxx("", num_elements);
  ExecViewManaged<Real*[NUM_TIME_LEVELS][NP][NP]>::HostMirror field_2d_cxx_host;
  field_2d_cxx_host = Kokkos::create_mirror_view(field_2d_cxx);

  HostViewManaged<Real*[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]> field_3d_f90("", num_elements);
  ExecViewManaged<Scalar*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]> field_3d_cxx ("", num_elements);
  ExecViewManaged<Scalar*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror field_3d_cxx_host;
  field_3d_cxx_host = Kokkos::create_mirror_view(field_3d_cxx);

  HostViewManaged<Real*[NUM_TIME_LEVELS][DIM][NUM_PHYSICAL_LEV][NP][NP]> field_4d_f90 ("", num_elements);
  ExecViewManaged<Scalar*[NUM_TIME_LEVELS][DIM][NP][NP][NUM_LEV]> field_4d_cxx ("", num_elements);
  ExecViewManaged<Scalar*[NUM_TIME_LEVELS][DIM][NP][NP][NUM_LEV]>::HostMirror field_4d_cxx_host;
  field_4d_cxx_host = Kokkos::create_mirror_view(field_4d_cxx);

  // Get the buffers manager
  std::shared_ptr<BuffersManager> buffers_manager = Context::singleton().get_buffers_manager();
  buffers_manager->set_connectivity(connectivity);

  // Create boundary exchanges
  std::shared_ptr<BoundaryExchange> be1 = std::make_shared<BoundaryExchange>(connectivity,buffers_manager);
  std::shared_ptr<BoundaryExchange> be2 = std::make_shared<BoundaryExchange>(connectivity,buffers_manager);

  // Setup the be objects
  be1->set_num_fields(num_scalar_fields_2d,DIM*num_vector_fields_3d);
  be1->register_field(field_2d_cxx,1,field_2d_idim);
  be1->register_field(field_4d_cxx,  field_4d_outer_idim,DIM,0);
  be1->registration_completed();

  be2->set_num_fields(0,num_scalar_fields_3d);
  be2->register_field(field_3d_cxx,1,field_3d_idim);
  be2->registration_completed();

  for (int itest=0; itest<num_tests; ++itest)
  {
    // Initialize input data to random values
    genRandArray(field_2d_f90,engine,dreal);
    for (int ie=0; ie<num_elements; ++ie) {
      for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
        for (int igp=0; igp<NP; ++igp) {
          for (int jgp=0; jgp<NP; ++jgp) {
            field_2d_cxx_host(ie,itl,igp,jgp) = field_2d_f90(ie,itl,igp,jgp);
    }}}}
    Kokkos::deep_copy(field_2d_cxx, field_2d_cxx_host);

    genRandArray(field_3d_f90,engine,dreal);
    for (int ie=0; ie<num_elements; ++ie) {
      for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
        for (int level=0; level<NUM_PHYSICAL_LEV; ++level) {
          const int ilev = level / VECTOR_SIZE;
          const int ivec = level % VECTOR_SIZE;
          for (int igp=0; igp<NP; ++igp) {
            for (int jgp=0; jgp<NP; ++jgp) {
              field_3d_cxx_host(ie,itl,igp,jgp,ilev)[ivec] = field_3d_f90(ie,itl,level,igp,jgp);
    }}}}}
    Kokkos::deep_copy(field_3d_cxx, field_3d_cxx_host);

    genRandArray(field_4d_f90,engine,dreal);
    for (int ie=0; ie<num_elements; ++ie) {
      for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
        for (int idim=0; idim<DIM; ++idim) {
          for (int level=0; level<NUM_PHYSICAL_LEV; ++level) {
            const int ilev = level / VECTOR_SIZE;
            const int ivec = level % VECTOR_SIZE;
            for (int igp=0; igp<NP; ++igp) {
              for (int jgp=0; jgp<NP; ++jgp) {
                field_4d_cxx_host(ie,itl,idim,igp,jgp,ilev)[ivec] = field_4d_f90(ie,itl,idim,level,igp,jgp);
    }}}}}}
    Kokkos::deep_copy(field_4d_cxx, field_4d_cxx_host);

    // Perform boundary exchange
    boundary_exchange_test_f90(field_2d_f90.data(), field_3d_f90.data(), field_4d_f90.data(), DIM, NUM_TIME_LEVELS, field_2d_idim+1, field_3d_idim+1, field_4d_outer_idim+1);
    be1->exchange();
    be2->exchange();
    Kokkos::deep_copy(field_2d_cxx_host, field_2d_cxx);
    Kokkos::deep_copy(field_3d_cxx_host, field_3d_cxx);
    Kokkos::deep_copy(field_4d_cxx_host, field_4d_cxx);

    // Compare answers
    for (int ie=0; ie<num_elements; ++ie) {
      for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
        for (int igp=0; igp<NP; ++igp) {
          for (int jgp=0; jgp<NP; ++jgp) {
            if(compare_answers(field_2d_f90(ie,itl,igp,jgp),field_2d_cxx_host(ie,itl,igp,jgp)) >= test_tolerance) {
              std::cout << "rank,ie,itl,igp,jgp: " << rank << ", " << ie << ", " << itl << ", " << igp << ", " << jgp << "\n";
              std::cout << "f90: " << field_2d_f90(ie,itl,igp,jgp) << "\n";
              std::cout << "cxx: " << field_2d_cxx_host(ie,itl,igp,jgp) << "\n";
            }
            REQUIRE(compare_answers(field_2d_f90(ie,itl,igp,jgp),field_2d_cxx_host(ie,itl,igp,jgp)) < test_tolerance);
    }}}}

    for (int ie=0; ie<num_elements; ++ie) {
      for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
        for (int level=0; level<NUM_PHYSICAL_LEV; ++level) {
          const int ilev = level / VECTOR_SIZE;
          const int ivec = level % VECTOR_SIZE;
          for (int igp=0; igp<NP; ++igp) {
            for (int jgp=0; jgp<NP; ++jgp) {
              if(compare_answers(field_3d_f90(ie,itl,level,igp,jgp),field_3d_cxx_host(ie,itl,igp,jgp,ilev)[ivec]) >= test_tolerance) {
                std::cout << std::setprecision(17) << "ie,itl,igp,jgp,ilev,iv: " << ie << ", " << itl << ", " << igp << ", " << jgp << ", " << ilev << ", " << ivec << "\n";
                std::cout << std::setprecision(17) << "f90: " << field_3d_f90(ie,itl,level,igp,jgp) << "\n";
                std::cout << std::setprecision(17) << "cxx: " << field_3d_cxx_host(ie,itl,igp,jgp,ilev)[ivec] << "\n";
              }
              REQUIRE(compare_answers(field_3d_f90(ie,itl,level,igp,jgp),field_3d_cxx_host(ie,itl,igp,jgp,ilev)[ivec]) < test_tolerance);
    }}}}}

    for (int ie=0; ie<num_elements; ++ie) {
      for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
        for (int idim=0; idim<DIM; ++idim) {
          for (int level=0; level<NUM_PHYSICAL_LEV; ++level) {
            const int ilev = level / VECTOR_SIZE;
            const int ivec = level % VECTOR_SIZE;
            for (int igp=0; igp<NP; ++igp) {
              for (int jgp=0; jgp<NP; ++jgp) {
                if(compare_answers(field_4d_f90(ie,itl,idim,level,igp,jgp),field_4d_cxx_host(ie,itl,idim,igp,jgp,ilev)[ivec]) >= test_tolerance) {
                  std::cout << std::setprecision(17) << "rank,ie,itl,idim,igp,jgp,ilev,iv: " << rank << ", " << ie << ", " << itl << ", " << idim << ", " << igp << ", " << jgp << ", " << ilev << ", " << ivec << "\n";
                  std::cout << std::setprecision(17) << "f90: " << field_4d_f90(ie,itl,idim,level,igp,jgp) << "\n";
                  std::cout << std::setprecision(17) << "cxx: " << field_4d_cxx_host(ie,itl,idim,igp,jgp,ilev)[ivec] << "\n";
                }
                REQUIRE(compare_answers(field_4d_f90(ie,itl,idim,level,igp,jgp),field_4d_cxx_host(ie,itl,idim,igp,jgp,ilev)[ivec]) < test_tolerance);
    }}}}}}
  }

  // Cleanup
  cleanup_f90();  // Deallocate stuff in the F90 module
  be1->clean_up();
  be2->clean_up();
}
