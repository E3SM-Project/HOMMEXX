
#ifndef _ERRORDEFS_HPP_
#define _ERRORDEFS_HPP_

namespace Homme {
namespace Errors {

void runtime_abort(std::string message, int code);

static constexpr int err_negative_layer_thickness = 101;
}
}

#endif // _ERRORDEFS_HPP_
