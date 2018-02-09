#ifndef HOMMEXX_VERTICAL_REMAP_MANAGER_HPP
#define HOMMEXX_VERTICAL_REMAP_MANAGER_HPP

#include <memory>

namespace Homme {

struct VerticalRemapManager {
  VerticalRemapManager();

  void run_remap(int np1, int n0_qdp, double dt) const;

private:
  struct Impl;
  std::shared_ptr<Impl> p_;
};

}

#endif
