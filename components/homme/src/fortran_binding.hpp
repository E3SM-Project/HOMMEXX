
#ifndef _FORTRANBINDING_HPP_
#define _FORTRANBINDING_HPP_

#define CAT_HELPER(a, b) a ## b
#define CAT(a, b) CAT_HELPER(a, b)

#define Q(a) #a
#define QUOTE(a) Q(a)

#define FORTRAN(modname, fname) __asm__(QUOTE(CAT(CAT(modname, _), CAT(fname, _))))
#endif
