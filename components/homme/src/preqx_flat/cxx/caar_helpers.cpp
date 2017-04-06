#include "caar_helpers.hpp"
#include <iomanip>
#include "Utility.hpp"
#include "Dimensions.hpp"

namespace Homme {

extern "C" {

void caar_compute_pressure_c (const int& nets, const int& nete,
                              const int& nelemd, const int& n0, const Real& hyai_ps0,
                              CRPtr& p_ptr, CRPtr& dp_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointer into an unmanaged view (all runtime dimensions, since the
  // type double[n]* is not allowed in C++)
  HostViewF90<Real****>  p_host  (p_ptr,  NP, NP, NUM_LEV, nelemd);
  HostViewF90<Real*****> dp_host (dp_ptr, NP, NP, NUM_LEV, NUM_TIME_LEVELS, nelemd);

  // Forward the views (just the input, actually) to the execution space, still with f90 ordering
  ExecViewF90<Real****>  p_exec_f90  (p_ptr,  NP, NP, NUM_LEV, nelemd);
  ExecViewF90<Real*****> dp_exec_f90 (dp_ptr, NP, NP, NUM_LEV, NUM_TIME_LEVELS, nelemd);
  Kokkos::deep_copy(dp_exec_f90, dp_host);

  // Flip the views (again, only need to process the inputs) to get compile time dimensions
  ExecViewManaged<Real*[NUM_LEV][NP][NP]> p_exec ("p", nelemd);
  ExecViewManaged<Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]> dp_exec ("dp", nelemd);
  flip_view_f90_to_cxx (dp_exec_f90, dp_exec);

  const int nets_c = nets - 1;
  const int n0_c   = n0 - 1;

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      Kokkos::single(
        Kokkos::PerTeam(team_member),
        KOKKOS_LAMBDA ()
        {
          const int ie = nets_c + team_member.league_rank();
          ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp = subview(dp_exec,ie,n0_c,ALL(),ALL(),ALL());
          ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p  = subview(p_exec,ie,ALL(),ALL(),ALL());

          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, NP * NP),
            KOKKOS_LAMBDA(const int idx)
            {
              const int igp = idx / NP;
              const int jgp = idx % NP;
              p(0,igp,jgp) = hyai_ps0 + 0.5 * dp(0,igp,jgp);
            }
          );

          for (int ilev=1; ilev<NUM_LEV; ++ilev)
          {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team_member, NP * NP),
              KOKKOS_LAMBDA(const int idx)
              {
                int igp = idx / NP;
                int jgp = idx % NP;
                p(ilev,igp,jgp) = p(ilev-1,igp,jgp) + 0.5*dp(ilev-1,igp,jgp) + 0.5*dp(ilev,igp,jgp);
              }
            );
          }
        }
      );
    }
  );

  // Flip the output views on the execution space
  flip_view_cxx_to_f90 (p_exec, p_exec_f90);

  // Copy the output views back onto the host space
  Kokkos::deep_copy(p_host, p_exec_f90);
}

} // extern "C"

} // namespace Homme
