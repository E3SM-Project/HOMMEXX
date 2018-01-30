#ifndef HOMMEXX_ERRORDEFS_HPP
#define HOMMEXX_ERRORDEFS_HPP

#include <string>

namespace Homme {
namespace Errors {

void runtime_check(bool cond, const std::string& message, int code);
void runtime_abort(const std::string& message, int code);

static constexpr int err_unknown_option           = 11;
static constexpr int err_unsupported_option       = 12;
static constexpr int err_negative_layer_thickness = 101;

} // namespace Homme
} // namespace Errors

#endif // HOMMEXX_ERRORDEFS_HPP
