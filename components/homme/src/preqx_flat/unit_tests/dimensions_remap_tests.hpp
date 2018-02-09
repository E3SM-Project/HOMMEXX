#ifndef HOMMEXX_DIMENSIONS_REMAP_TESTS_HPP
#define HOMMEXX_DIMENSIONS_REMAP_TESTS_HPP

#include <Dimensions.hpp>

namespace Homme {

//for unit tests
constexpr const int DIM3 = 3,
                    DIM10 = 10,
                    NLEV = NUM_PHYSICAL_LEV,
                    NLEVP1 = NUM_PHYSICAL_LEV + 1, 
                    NLEVP2 = NUM_PHYSICAL_LEV + 2,
                    NLEVP3 = NUM_PHYSICAL_LEV + 3, 
                    NLEVP4 = NUM_PHYSICAL_LEV + 4;

constexpr const int QSIZETEST=10;

} // namespace Homme

#endif // HOMMEXX_DIMENSIONS_REMAP_TESTS_HPP
