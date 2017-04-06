#ifndef HOMMEXX_ASSERT_HPP
#define HOMMEXX_ASSERT_HPP

#include <Hommexx_config.h>

#define HOMMEXX_ERROR_MSG(msg)        \
  printf("Error! %s", msg)

#define HOMMEXX_DO_ASSERT(cond, msg)  \
  do {                                \
    if (!(cond)) {                    \
      HOMMEXX_ERROR_MSG(msg);         \
      std::abort();                   \
    }                                 \
  } while (0)

#ifdef HOMMEXX_DEBUG
#define HOMMEXX_ASSERT(cond, msg) HOMMEXX_DO_ASSERT(cond, msg)
#else
#define HOMMEXX_ASSERT(cond, msg)
#endif

#endif // HOMMEXX_ASSERT_HPP
