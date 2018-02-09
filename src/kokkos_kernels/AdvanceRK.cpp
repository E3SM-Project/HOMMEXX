#include "AdvanceRK.hpp"

namespace Homme {

KOKKOS_FUNCTION
void AdvanceRK::operator()(const TeamMember &thread) const {
  const int ie = thread.league_rank() / NLEV;
  const int level = thread.league_rank() % NLEV;

  // code before ...

  SharedView div(thread.team_shmem());

  divergence_sphere(thread, ie, level, div);

  // code after ...
}

KOKKOS_FUNCTION
void AdvanceRK::divergence_sphere(const TeamMember &thread,
                                  const int ie,
                                  const int level,
                                  SharedView div) const {
  // SharedView gu( thread.team_shmem() +
  // SharedView::shmem_size() );
  // SharedView gv( thread.team_shmem() + 2u *
  // SharedView::shmem_size() );
  // SharedView vvtemp( thread.team_shmem() + 3u *
  // SharedView::shmem_size() );

  // TODO offset correctly
  SharedView gu(thread.team_shmem());
  SharedView gv(thread.team_shmem());
  SharedView vvtemp(thread.team_shmem());

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, NP * NP),
      [&](const int x) {
        const int i = x % NP;
        const int j = x / NP;
        gu(i, j) = elem_metdet(i, j, ie) *
                   (elem_dinv(i, j, 0, 0, ie) *
                        elem_v(i, j, 0, level, ie) +
                    elem_dinv(i, j, 0, 1, ie) *
                        elem_v(i, j, 1, level, ie));
        gv(i, j) = elem_metdet(i, j, ie) *
                   (elem_dinv(i, j, 1, 0, ie) *
                        elem_v(i, j, 0, level, ie) +
                    elem_dinv(i, j, 1, 1, ie) *
                        elem_v(i, j, 1, level, ie));
      });

  // TODO construct view in scratch memory
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, NP * NP * N),
      [&](const int x) {
        const int j = x % NP;
        const int l = x / NP;
        // TODO vector parallel_reduce
        double dudx = 0.0;
        double dvdy = 0.0;
        for(int i = 0; i < NP; ++i) {
          dudx += elem_deriv(i, l, ie) * gu(i, j);
          dvdy += elem_deriv(i, l, ie) * gv(j, i);
        }
        div(l, j) = dudx;
        vvtemp(j, l) = dvdy;
      });

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(thread, NP * NP),
      [&](const int x) {
        const int i = x % NP;
        const int j = x / NP;
        div(i, j) = RREARTH * (div(i, j) + vvtemp(i, j)) /
                    elem_metdet(i, j);
      });
}

}  // namespace Homme
