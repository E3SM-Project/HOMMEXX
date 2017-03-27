#include <catch/catch.hpp>

#include <Types.hpp>

using namespace Homme;

extern "C"
{

}

TEST_CASE ("dummy", "dummy") {
  SECTION ("dummy check") {
    REQUIRE(true);
  }
}
