
#ifndef _FORTRANBINDING_HPP_
#define _FORTRANBINDING_HPP_

#define CAT_HELPER(a, b) a##b
#define CAT(a, b) CAT_HELPER(a, b)

#define QUOTE_HELPER(a) #a
#define QUOTE(a) QUOTE_HELPER(a)

/* The following definitions are compiler specific */

/* GCC only: */
#define FORTRAN_FUNC(modname, fname) \
  __asm__(QUOTE(CAT(CAT(modname, _), CAT(fname, _))))

#define FORTRAN_C_FUNC(fname) __asm__(QUOTE(fname))

#define FORTRAN_VAR(modname, vname) \
  __asm__(QUOTE(CAT(CAT(CAT(__, modname), _MOD_), vname)))

#endif
