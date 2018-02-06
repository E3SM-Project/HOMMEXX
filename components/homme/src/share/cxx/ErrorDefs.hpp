#ifndef HOMME_ERRORDEFS_HPP
#define HOMME_ERRORDEFS_HPP

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

#include <string>

namespace Homme {
namespace Errors {

void runtime_check(bool cond, const std::string& message, int code);
void runtime_abort(const std::string& message, int code);

static constexpr int err_unknown_option           = 11;
static constexpr int err_not_implemented          = 12;
static constexpr int err_negative_layer_thickness = 101;

} // namespace Errors
} // namespace Homme

#endif // HOMME_ERRORDEFS_HPP
