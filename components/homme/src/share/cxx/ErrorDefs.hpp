
#ifndef _ERRORDEFS_HPP_
#define _ERRORDEFS_HPP_

#ifndef NDEBUG
#define DEBUG_PRINT(...) \
  do {                   \
    printf(__VA_ARGS__); \
  } while(false)
// This macro always evaluates eval, but
// This enables us to define variables specifically for use
// in asserts Note this can still cause issues
#define DEBUG_EXPECT(eval, expected) \
do {                                 \
  auto v = eval;                     \
  assert(v == expected);             \
} while(false)
#else
#define DEBUG_PRINT(...) \
do {                     \
} while(false)
#define DEBUG_EXPECT(eval, expected) \
do {                                 \
  eval;                              \
} while(false)
#endif

#ifdef DEBUG_TRACE
#define TRACE_PRINT(...) \
do {                     \
  printf(__VA_ARGS__);   \
} while(false)
#else
#define TRACE_PRINT(...) \
do {                     \
} while(false)
#endif

namespace Homme {
namespace Errors {

void runtime_abort(std::string message, int code);

static constexpr int err_negative_layer_thickness = 101;
}  // namespace Errors
}

#endif  // _ERRORDEFS_HPP_
