#ifndef HOMMEXX_CONTEXT_HPP
#define HOMMEXX_CONTEXT_HPP

#include <string>
#include <map>
#include <memory>

namespace Homme {

class Control;
class Elements;
class Derivative;
class Connectivity;
class BoundaryExchange;

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
public:
  using BEMap = std::map<std::string,BoundaryExchange>;

private:
  // Note: using uniqe_ptr disables copy construction
  std::unique_ptr<Control>      control_;
  std::unique_ptr<Elements>     elements_;
  std::unique_ptr<Derivative>   derivative_;
  std::unique_ptr<Connectivity> connectivity_;
  std::unique_ptr<BEMap>        boundary_exchanges_;

  // Clear the objects Context manages.
  void clear();

public:
  Context();
  virtual ~Context();

  // Getters for each managed object.
  Control& get_control();
  Elements& get_elements();
  Derivative& get_derivative();
  Connectivity& get_connectivity();
  BEMap& get_boundary_exchanges();
  BoundaryExchange& get_boundary_exchange(const std::string& name);

  // Exactly one singleton.
  static Context& singleton();

  static void finalize_singleton();
};

}

#endif // HOMMEXX_CONTEXT_HPP
