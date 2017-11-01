#ifndef HOMMEXX_CONTEXT_HPP
#define HOMMEXX_CONTEXT_HPP

#include <memory>

namespace Homme {

class Control;
class Elements;
class Derivative;

/* A Context manages resources previously treated as singletons. Context is
 * meant to have two roles. First, a Context singleton is the only singleton in
 * the program. Second, a context need not be a singleton, and each Context
 * object can have different Elements, Control, Derivative, etc., objects. (That
 * probably isn't needed, but Context immediately supports it.)
 *  
 * Finally, Context has two singleton functions: singleton(), which returns
 * Context&, and finalize_singleton(). The second is called in a unit test exe
 * main before Kokkos::finalize().
 */
class Context {
  std::shared_ptr<Control> control_;
  std::shared_ptr<Elements> elements_;
  std::shared_ptr<Derivative> derivative_;

  // Clear the objects Context manages.
  void clear();

public:
  Context();
  virtual ~Context();

  // Getters for each managed object.
  Control& get_control();
  Elements& get_elements();
  Derivative& get_derivative();

  // Exactly one singleton.
  static Context& singleton();
  // Call this to clean up the resources in this Context prior to calling, e.g.,
  // Kokkos::finalize().
  static void finalize_singleton();
};

}

#endif // HOMMEXX_CONTEXT_HPP
