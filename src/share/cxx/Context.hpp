#ifndef HOMMEXX_CONTEXT_HPP
#define HOMMEXX_CONTEXT_HPP

#include <string>
#include <map>
#include <memory>

namespace Homme {

class Comm;
class Control;
class Elements;
class Derivative;
class Connectivity;
class BoundaryExchange;
class BuffersManager;
class SimulationParams;
class TimeLevel;
class VerticalRemapManager;

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
  using BEMap = std::map<std::string,std::shared_ptr<BoundaryExchange>>;
  using BMMap = std::map<int,std::shared_ptr<BuffersManager>>;

private:
  // Note: using uniqe_ptr disables copy construction
  std::unique_ptr<Comm>               comm_;
  std::unique_ptr<Control>            control_;
  std::unique_ptr<Elements>           elements_;
  std::unique_ptr<Derivative>         derivative_;
  std::shared_ptr<Connectivity>       connectivity_;
  std::shared_ptr<BMMap>              buffers_managers_;
  std::unique_ptr<BEMap>              boundary_exchanges_;
  std::unique_ptr<SimulationParams>   simulation_params_;
  std::unique_ptr<TimeLevel>          time_level_;
  std::unique_ptr<VerticalRemapManager> vertical_remap_mgr_;

  // Clear the objects Context manages.
  void clear();

public:
  Context();
  virtual ~Context();

  // Getters for each managed object.
  Comm& get_comm();
  Control& get_control();
  Elements& get_elements();
  Derivative& get_derivative();
  SimulationParams& get_simulation_params();
  TimeLevel& get_time_level();
  VerticalRemapManager& get_vertical_remap_manager();
  std::shared_ptr<Connectivity> get_connectivity();
  BMMap& get_buffers_managers();
  std::shared_ptr<BuffersManager> get_buffers_manager(short int exchange_type);
  BEMap& get_boundary_exchanges();
  std::shared_ptr<BoundaryExchange> get_boundary_exchange(const std::string& name);

  // Exactly one singleton.
  static Context& singleton();

  static void finalize_singleton();
};

}

#endif // HOMMEXX_CONTEXT_HPP
