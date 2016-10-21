
#ifndef _FORTRANBINDING_HPP_
#define _FORTRANBINDING_HPP_

#define CAT_HELPER(a, b) a##b
#define CAT(a, b) CAT_HELPER(a, b)

#define QUOTE_HELPER(a) #a
#define QUOTE(a) QUOTE_HELPER(a)

#define FORTRAN(modname, fname) \
  __asm__(QUOTE(CAT(CAT(modname, _), CAT(fname, _))))
#define FORTRAN_C(fname) __asm__(QUOTE(fname))

#endif
