#include "Types.hpp"

namespace Homme {

class AdvanceRK {
 public:
  using SharedView =
      Kokkos::View<Scalar[NP][NP], Kokkos::LayoutLeft,
                   SharedSpace, Kokkos::MemoryUnmanaged>;

  // TODO set up views
  AdvanceRK(Dinv arg_elem_dinv, Deriv arg_elem_deriv,
            P arg_elem_p, PS arg_elem_ps,
            GradPS arg_elem_gradps, V arg_elem_v,
            Det arg_elem_metdet)
      : elem_dinv{arg_elem_dinv},
        elem_deriv{arg_elem_deriv},
        elem_p{arg_elem_p},
        elem_ps{arg_elem_ps},
        elem_gradps{arg_elem_gradps},
        elem_v{arg_elem_v},
        elem_metdet{arg_elem_metdet} {}

  size_t team_shmem_size(int team_size) const {
    return 4u * SharedView::shmem_size();
  }

  KOKKOS_FUNCTION
  void operator()(const TeamMember &thread) const;

  KOKKOS_FUNCTION
  void divergence_sphere(const TeamMember &thread,
                         const int ie, const int level,
                         SharedView div) const;

 private:
  // RK     my_rk;
  Dinv elem_dinv;
  Deriv elem_deriv;
  P elem_p;
  PS elem_ps;
  GradPS elem_gradps;
  V elem_v;
  Det elem_metdet;
};

}  // namespace Homme
