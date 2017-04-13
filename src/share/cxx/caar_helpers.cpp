#include "Types.hpp"
#include "Utility.hpp"
#include "Dimensions.hpp"
#include "SphereOperators.hpp"
#include "PhysicalConstants.hpp"
#include "KokkosExp_MDRangePolicy.hpp"

namespace Homme {

KOKKOS_FORCEINLINE_FUNCTION
Real Virtual_Temperature (const Real Tin, const Real rin)
{
  return Tin*(1.0 + (PhysicalConstants::Rwater_vapor/PhysicalConstants::Rgas - 1.0)*rin);
}

KOKKOS_FORCEINLINE_FUNCTION
Real Virtual_Specific_Heat(const Real rin)
{
  return PhysicalConstants::cp*(1.0 + (PhysicalConstants::Cpwater_vapor/PhysicalConstants::cp - 1.0)*rin);
}

extern "C" {

static constexpr ExecSpace exec_space;

void caar_compute_pressure_c (const int& nets, const int& nete,
                              const int& nelemd, const int& n0, const Real& hyai_ps0,
                              RCPtr& p_ptr, CRCPtr& dp_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointer into an unmanaged view
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]> p_host (p_ptr, nelemd);
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]> dp_host (dp_ptr, nelemd);

  // Create the execution space views. If ExecSpace can access HostMemSpace,
  // then the exexecution space views.c view is storing the host view pointer, otherwise is managed
  auto p_exec  = Kokkos::create_mirror_view (exec_space,p_host);
  auto dp_exec = Kokkos::create_mirror_view (exec_space,dp_host);

  // Forward the views (just the input, actually) to the execution space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view(dp_exec, dp_host);

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
          ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> dp = subview(dp_exec,ie,n0_c,ALL(),ALL(),ALL());
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

  // Copy the output views back onto the host space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view(p_host, p_exec);
}

void caar_compute_vort_and_div_c (const int& nets, const int& nete, const int& nelemd, const int& n0,
                                  const Real& eta_ave_w, CRCPtr& dvv_ptr, CRCPtr& D_ptr, CRCPtr& Dinv_ptr,
                                  CRCPtr& metdet_ptr, CRCPtr& rmetdet_ptr, CRCPtr& p_ptr, CRCPtr& dp_ptr,
                                  RCPtr& grad_p_ptr, RCPtr& vgrad_p_ptr, CRCPtr& v_ptr, RCPtr& vn0_ptr,
                                  RCPtr& vdp_ptr, RCPtr& div_vdp_ptr, RCPtr& vort_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointers into a unmanaged views
  HostViewUnmanaged<const Real[NP][NP]>                               dvv_host     (dvv_ptr);
  HostViewUnmanaged<const Real*[2][2][NP][NP]>                        D_host       (D_ptr,       nelemd);
  HostViewUnmanaged<const Real*[2][2][NP][NP]>                        Dinv_host    (Dinv_ptr,    nelemd);
  HostViewUnmanaged<const Real*[NP][NP]>                              metdet_host  (metdet_ptr,  nelemd);
  HostViewUnmanaged<const Real*[NP][NP]>                              rmetdet_host (rmetdet_ptr,  nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>                     p_host       (p_ptr,       nelemd);
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]>    dp_host      (dp_ptr,      nelemd);
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NUM_LEV][2][NP][NP]> v_host       (v_ptr,       nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][2][NP][NP]>                        vn0_host     (vn0_ptr,     nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][2][NP][NP]>                        grad_p_host  (grad_p_ptr,  nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]>                           vgrad_p_host (vgrad_p_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][2][NP][NP]>                        vdp_host     (vdp_ptr,     nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]>                           div_vdp_host (div_vdp_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]>                           vort_host    (vort_ptr,    nelemd);

  // Create the execution space views. If ExecSpace can access HostMemSpace,
  // then the exec view is storing the host view pointer, otherwise is managed
  auto dvv_exec     = Kokkos::create_mirror_view (exec_space, dvv_host     );
  auto D_exec       = Kokkos::create_mirror_view (exec_space, D_host       );
  auto Dinv_exec    = Kokkos::create_mirror_view (exec_space, Dinv_host    );
  auto metdet_exec  = Kokkos::create_mirror_view (exec_space, metdet_host  );
  auto rmetdet_exec = Kokkos::create_mirror_view (exec_space, rmetdet_host );
  auto p_exec       = Kokkos::create_mirror_view (exec_space, p_host       );
  auto dp_exec      = Kokkos::create_mirror_view (exec_space, dp_host      );
  auto v_exec       = Kokkos::create_mirror_view (exec_space, v_host       );
  auto vn0_exec     = Kokkos::create_mirror_view (exec_space, vn0_host     );
  auto grad_p_exec  = Kokkos::create_mirror_view (exec_space, grad_p_host  );
  auto vgrad_p_exec = Kokkos::create_mirror_view (exec_space, vgrad_p_host );
  auto vdp_exec     = Kokkos::create_mirror_view (exec_space, vdp_host     );
  auto div_vdp_exec = Kokkos::create_mirror_view (exec_space, div_vdp_host );
  auto vort_exec    = Kokkos::create_mirror_view (exec_space, vort_host    );

  // Forward the views (just the inputs, actually) to the execution space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view(dvv_exec,     dvv_host);
  deep_copy_mirror_view(D_exec,       D_host);
  deep_copy_mirror_view(Dinv_exec,    Dinv_host);
  deep_copy_mirror_view(metdet_exec,  metdet_host);
  deep_copy_mirror_view(rmetdet_exec, rmetdet_host);
  deep_copy_mirror_view(p_exec,       p_host);
  deep_copy_mirror_view(dp_exec,      dp_host);
  deep_copy_mirror_view(v_exec,       v_host);
  deep_copy_mirror_view(vn0_exec,     vn0_host);

  // Fix indices: they are coming from fortran where they are 1-based.
  const int nets_c = nets - 1;
  const int n0_c   = n0 - 1;

  // Parallel loop

  Kokkos::parallel_for(
#ifdef USE_KOKKOS_ON_ELEMENTS // Manually add this to the config.h.c file before compiling (but AFTER configuring)
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, Kokkos::AUTO),
#else
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, ExecSpace::thread_pool_size()),
#endif
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ie = nets_c + team_member.league_rank();

      // Create subviews to explicitly have static dimensions
      ExecViewUnmanaged<const Real[2][2][NP][NP]>       D_ie       = subview(D_exec,       ie,ALL(),ALL(),ALL(),ALL());
      ExecViewUnmanaged<const Real[2][2][NP][NP]>       Dinv_ie    = subview(Dinv_exec,    ie,ALL(),ALL(),ALL(),ALL());
      ExecViewUnmanaged<const Real[NP][NP]>             metdet_ie  = subview(metdet_exec,  ie,ALL(),ALL());
      ExecViewUnmanaged<const Real[NP][NP]>             rmetdet_ie  = subview(rmetdet_exec,  ie,ALL(),ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    p_ie       = subview(p_exec,       ie,ALL(),ALL(),ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    dp_ie      = subview(dp_exec,      ie,n0_c,ALL(),ALL(),ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][2][NP][NP]> v_ie       = subview(v_exec,       ie,n0_c,ALL(),ALL(),ALL(),ALL());
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]>       vn0_ie     = subview(vn0_exec,     ie,ALL(),ALL(),ALL(),ALL());
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]>       grad_p_ie  = subview(grad_p_exec,  ie,ALL(),ALL(),ALL(),ALL());
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>          vgrad_p_ie = subview(vgrad_p_exec, ie,ALL(),ALL(),ALL());
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]>       vdp_ie     = subview(vdp_exec,     ie,ALL(),ALL(),ALL(),ALL());
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>          div_vdp_ie = subview(div_vdp_exec, ie,ALL(),ALL(),ALL());
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>          vort_ie    = subview(vort_exec,    ie,ALL(),ALL(),ALL());

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, NUM_LEV),
        KOKKOS_LAMBDA (const int ilev)
        {
          // Extract level slice since gradient_sphere acts on a single level
          ExecViewUnmanaged<const Real[NP][NP]> p_ilev      = subview(p_ie, ilev, ALL(), ALL());
          ExecViewUnmanaged<Real[2][NP][NP]>    grad_p_ilev = subview(grad_p_ie, ilev, ALL(), ALL(), ALL());
          gradient_sphere(team_member, p_ilev, dvv_exec, Dinv_ie, grad_p_ilev);

          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, NP * NP),
            KOKKOS_LAMBDA (const int idx)
            {
              const int igp = idx / NP;
              const int jgp = idx % NP;

              const Real v1 = v_ie(ilev,0,igp,jgp);
              const Real v2 = v_ie(ilev,1,igp,jgp);

              vgrad_p_ie(ilev, igp, jgp) = v1*grad_p_ie(ilev, 0, igp, jgp) + v2*grad_p_ie(ilev, 1, igp, jgp);

              vdp_ie(ilev, 0, igp, jgp) = v1 * dp_ie(ilev, igp, jgp);
              vdp_ie(ilev, 1, igp, jgp) = v2 * dp_ie(ilev, igp, jgp);

              vn0_ie(ilev, 0, igp, jgp) += eta_ave_w * vdp_ie(ilev, 0, igp, jgp);
              vn0_ie(ilev, 1, igp, jgp) += eta_ave_w * vdp_ie(ilev, 1, igp, jgp);
            }
          );

          // Extract level slice since divergence_sphere acts on a single level
          ExecViewUnmanaged<Real[NP][NP]>    div_vdp_ilev = subview(div_vdp_ie, ilev, ALL(), ALL());
          ExecViewUnmanaged<Real[2][NP][NP]> vdp_ilev     = subview(vdp_ie, ilev, ALL(), ALL(), ALL());
          divergence_sphere(team_member, vdp_ilev, dvv_exec, metdet_ie, rmetdet_ie, Dinv_ie, div_vdp_ilev);

          // Extract level slice since vorticity_sphere acts on a single level (and (u,v) separated)
          ExecViewUnmanaged<const Real[NP][NP]> U_ilev    = subview(v_ie, ilev, 0, ALL(), ALL());
          ExecViewUnmanaged<const Real[NP][NP]> V_ilev    = subview(v_ie, ilev, 1, ALL(), ALL());
          ExecViewUnmanaged<Real[NP][NP]>       vort_ilev = subview(vort_ie, ilev, ALL(), ALL());
          vorticity_sphere(team_member, U_ilev, V_ilev, dvv_exec, rmetdet_ie, D_ie, vort_ilev);
        }
      );
    }
  );

  // Copy the output views back onto the host space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view(grad_p_host,  grad_p_exec);
  deep_copy_mirror_view(vgrad_p_host, vgrad_p_exec);
  deep_copy_mirror_view(vdp_host,     vdp_exec);
  deep_copy_mirror_view(div_vdp_host, div_vdp_exec);
  deep_copy_mirror_view(vort_host,    vort_exec);
  deep_copy_mirror_view(vn0_host,     vn0_exec);
}

void caar_compute_t_v_c (const int& nets, const int& nete, const int& nelemd,
                         const int& n0,   const int& qn0,  const int& use_cpstar,
                         RCPtr& T_v_ptr,  RCPtr& kappa_star_ptr,
                         CRCPtr& dp_ptr,  CRCPtr& Temp_ptr, CRCPtr& Qdp_ptr)
{
  // Store the f90 pointers into a unmanaged views
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]> Temp_host (Temp_ptr, nelemd);

  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]> T_v_host        (T_v_ptr,        nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]> kappa_star_host (kappa_star_ptr, nelemd);

  // Create the execution space views. If ExecSpace can access HostMemSpace,
  // then the exec view is storing the host view pointer, otherwise is managed
  auto T_v_exec        = Kokkos::create_mirror_view (exec_space, T_v_host       );
  auto kappa_star_exec = Kokkos::create_mirror_view (exec_space, kappa_star_host);
  auto Temp_exec       = Kokkos::create_mirror_view (exec_space, Temp_host      );

  // Forward the views (just the inputs, actually) to the execution space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (Temp_exec, Temp_host);

  // Fix indices: they are coming from fortran where they are 1-based.
  const int nets_c = nets - 1;
  const int n0_c   = n0 - 1;
  const int qn0_c  = qn0 - 1;

  if (qn0_c == -1)
  {
#ifdef USE_MD_RANGE_POLICY
    constexpr Kokkos::Experimental::Iterate Right = Kokkos::Experimental::Iterate::Right;
    using Rank = Kokkos::Experimental::Rank<3,Right,Right>;
    using RangePolicy = Kokkos::Experimental::MDRangePolicy<Rank,Kokkos::IndexType<int>>;

    Kokkos::Experimental::md_parallel_for(
      RangePolicy({nets_c,0,0},{nete,NUM_LEV,NP},{1,1,1}),
      KOKKOS_LAMBDA (const int ie, const int ilev, const int igp)
      {
        // MDRangePolicy supports up to rank 3 so far... We need to add the inner layer by hand
        for (int jgp=0; jgp<NP; ++jgp)
        {
          T_v_exec (ie,ilev,igp,jgp) = Temp_exec(ie,n0,ilev,igp,jgp);
          kappa_star_exec(ie,ilev,igp,jgp) = PhysicalConstants::kappa;
        }
      }
    );
#else
    Kokkos::parallel_for(
#ifdef USE_KOKKOS_ON_ELEMENTS // Manually add this to the config.h.c file before compiling (but AFTER configuring)
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, Kokkos::AUTO),
#else
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, ExecSpace::thread_pool_size()),
#endif
      KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
      {
        const int ie = nets_c + team_member.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, NUM_LEV),
          KOKKOS_LAMBDA (const int ilev)
          {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team_member, NP * NP),
              KOKKOS_LAMBDA (const int idx)
              {
                const int igp = idx / NP;
                const int jgp = idx % NP;

                T_v_exec (ie,ilev,igp,jgp) = Temp_exec(ie,n0,ilev,igp,jgp);
                kappa_star_exec(ie,ilev,igp,jgp) = PhysicalConstants::kappa;
              }
            );
          }
        );
      }
    );
#endif
  }
  else
  {
    // Two more views needed
    HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]>            dp_host  (dp_ptr,  nelemd);
    HostViewUnmanaged<const Real*[QSIZE_D][Q_NUM_TIME_LEVELS][NUM_LEV][NP][NP]> Qdp_host (Qdp_ptr, nelemd);

    auto dp_exec         = Kokkos::create_mirror_view (exec_space, dp_host );
    auto Qdp_exec        = Kokkos::create_mirror_view (exec_space, Qdp_host);

    deep_copy_mirror_view (dp_exec,  dp_host);
    deep_copy_mirror_view (Qdp_exec, Qdp_host);
#ifdef USE_MD_RANGE_POLICY
    constexpr Kokkos::Experimental::Iterate Right = Kokkos::Experimental::Iterate::Right;
    using Rank = Kokkos::Experimental::Rank<3,Right,Right>;
    using RangePolicy = Kokkos::Experimental::MDRangePolicy<Rank,Kokkos::IndexType<int>>;

    Kokkos::Experimental::md_parallel_for(
      RangePolicy({nets_c,0,0},{nete,NUM_LEV,NP},{1,1,1}),
      KOKKOS_LAMBDA (const int ie, const int ilev, const int igp)
      {
        // MDRangePolicy supports up to rank 3 so far... We need to add the inner layer by hand
        for (int jgp=0; jgp<NP; ++jgp)
        {
          const Real Qt  = Qdp_exec(ie,0,qn0_c,ilev,igp,jgp) / dp_exec(ie,n0_c,ilev,igp,jgp);
          const Real Tin = Temp_exec(ie,n0_c,ilev,igp,jgp);

          T_v_exec (ie,ilev,igp,jgp) = Virtual_Temperature(Tin,Qt);
          if (use_cpstar==1)
          {
            kappa_star_exec(ie,ilev,igp,jgp) = PhysicalConstants::Rgas/Virtual_Specific_Heat(Qt);
          }
          else
          {
            kappa_star_exec(ie,ilev,igp,jgp) = PhysicalConstants::kappa;
          }
        }
      }
    );
#else
    Kokkos::parallel_for(
#ifdef USE_KOKKOS_ON_ELEMENTS // Manually add this to the config.h.c file before compiling (but AFTER configuring)
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, Kokkos::AUTO),
#else
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, ExecSpace::thread_pool_size()),
#endif
      KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
      {
        const int ie = nets_c + team_member.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, NUM_LEV),
          KOKKOS_LAMBDA (const int ilev)
          {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team_member, NP * NP),
              KOKKOS_LAMBDA (const int idx)
              {
                const int igp = idx / NP;
                const int jgp = idx % NP;

                const Real Qt  = Qdp_exec(ie,0,qn0_c,ilev,igp,jgp) / dp_exec(ie,n0_c,ilev,igp,jgp);
                const Real Tin = Temp_exec(ie,n0_c,ilev,igp,jgp);

                T_v_exec (ie,ilev,igp,jgp) = Virtual_Temperature(Tin,Qt);

                if (use_cpstar==1)
                {
                  kappa_star_exec(ie,ilev,igp,jgp) = PhysicalConstants::Rgas/Virtual_Specific_Heat(Qt);
                }
                else
                {
                  kappa_star_exec(ie,ilev,igp,jgp) = PhysicalConstants::kappa;
                }
              }
            );
          }
        );
      }
    );
#endif
  }
}

} // extern "C"

} // namespace Homme
