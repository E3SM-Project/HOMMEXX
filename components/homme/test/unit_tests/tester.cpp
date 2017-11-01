
#define CATCH_CONFIG_RUNNER

#include "catch/catch.hpp"

#include <Kokkos_Core.hpp>

#include "Context.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize();
  Kokkos::DefaultExecutionSpace::print_configuration(std::cout, true);
  int result = Catch::Session().run(argc, argv);
  Homme::Context::finalize_singleton();
  Kokkos::finalize();
  return result;
}
