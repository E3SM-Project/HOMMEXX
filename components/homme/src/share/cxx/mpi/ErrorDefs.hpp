#ifndef HOMMEXX_ERRORDEFS_HPP
#define HOMMEXX_ERRORDEFS_HPP

#include <string>

namespace Homme {
namespace Errors {

void runtime_abort(std::string message, int code);

static constexpr int err_negative_layer_thickness = 101;
static constexpr int functionality_not_yet_implemented= 7;
static constexpr int unknown_option = 11;
static constexpr int unsupported_option = 12;

} // namespace Errors
} // namespace Homme

#endif // HOMMEXX_ERRORDEFS_HPP
