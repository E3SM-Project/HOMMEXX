#include <catch/catch.hpp>

#include "Context.hpp"
#include "Control.hpp"
#include "Derivative.hpp"
#include "Elements.hpp"
#include "BiharmonicFunctor.hpp"
#include "Types.hpp"

#include <random>
#include <iomanip>

using namespace Homme;

// =========================== TESTS ============================ //

TEST_CASE ("BiharmonicFunctor", "Testing the biharmonic functor class")
{
  Control&    data     = Context::singleton().get_control();
  Elements&   elements = Context::singleton().get_elements();
  Derivative& deriv    = Context::singleton().get_derivative();

  BiharmonicFunctor func(data,elements,deriv);

  REQUIRE(true);
}
