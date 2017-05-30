
#define CATCH_CONFIG_RUNNER

#include "catch/catch.hpp"

#include <Kokkos_Core.hpp>

int main(int argc, char **argv) {
  Kokkos::initialize();
  Kokkos::print_configuration(std::cout,true);
  int result = Catch::Session().run(argc, argv);
  Kokkos::finalize();
  return result;
}
