#include "Types.hpp"
#include "Utility.hpp"
#include "Dimensions.hpp"
#include "Derivative.hpp"
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

static constexpr ExecSpace exec_space;

extern "C" {

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
                                  const Real& eta_ave_w, CRCPtr& D_ptr, CRCPtr& Dinv_ptr,
                                  CRCPtr& metdet_ptr, CRCPtr& rmetdet_ptr, CRCPtr& p_ptr, CRCPtr& dp_ptr,
                                  RCPtr& grad_p_ptr, RCPtr& vgrad_p_ptr, CRCPtr& v_ptr, RCPtr& vn0_ptr,
                                  RCPtr& vdp_ptr, RCPtr& div_vdp_ptr, RCPtr& vort_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointers into a unmanaged views
  HostViewUnmanaged<const Real*[2][2][NP][NP]>                        D_host       (D_ptr,       nelemd);
  HostViewUnmanaged<const Real*[2][2][NP][NP]>                        Dinv_host    (Dinv_ptr,    nelemd);
  HostViewUnmanaged<const Real*[NP][NP]>                              metdet_host  (metdet_ptr,  nelemd);
  HostViewUnmanaged<const Real*[NP][NP]>                              rmetdet_host (rmetdet_ptr, nelemd);
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
  auto dvv_exec     = get_derivative().get_dvv();
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
// Manually add this definition to the config.h.c file before compiling
// (but AFTER configuring) if you want Kokkos to parallelize on elements
// as well. We are using this to compare apples to apples with the fortran
// version, where we manually specify the number of threads for horizontal
// and vertical openmp. Once comparisons are done, we will remove this
// ifdef, and let Kokkos decide teams sizes
#ifdef USE_KOKKOS_ON_ELEMENTS
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
  const int qn0_c  = qn0==-1 ? -1 : qn0 - 1;

  if (qn0_c == -1)
  {
#ifdef USE_MD_RANGE_POLICY
    constexpr Kokkos::Experimental::Iterate Right = Kokkos::Experimental::Iterate::Right;
    using Rank = Kokkos::Experimental::Rank<3,Right,Right>;
    using RangePolicy = Kokkos::Experimental::MDRangePolicy<Rank,Kokkos::IndexType<int>>;

    Kokkos::Experimental::md_parallel_for(
      RangePolicy({0,0,0},{NUM_LEV,NP,NP},{1,1,1}),
      KOKKOS_LAMBDA (const int ilev, const int igp, const int jgp)
      {
        // MDRangePolicy supports up to rank 3 so far... We need to add the outer layer by hand
        for (int ie=nets_c; ie<nete; ++ie)
        {
          T_v_exec (ie,ilev,igp,jgp) = Temp_exec(ie,n0_c,ilev,igp,jgp);
          kappa_star_exec(ie,ilev,igp,jgp) = PhysicalConstants::kappa;
        }
      }
    );
#else
    Kokkos::parallel_for(
// Manually add this definition to the config.h.c file before compiling
// (but AFTER configuring) if you want Kokkos to parallelize on elements
// as well. We are using this to compare apples to apples with the fortran
// version, where we manually specify the number of threads for horizontal
// and vertical openmp. Once comparisons are done, we will remove this
// ifdef, and let Kokkos decide teams sizes
#ifdef USE_KOKKOS_ON_ELEMENTS
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

                T_v_exec (ie,ilev,igp,jgp) = Temp_exec(ie,n0_c,ilev,igp,jgp);
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
    HostViewUnmanaged<const Real*[Q_NUM_TIME_LEVELS][QSIZE_D][NUM_LEV][NP][NP]> Qdp_host (Qdp_ptr, nelemd);

    auto dp_exec         = Kokkos::create_mirror_view (exec_space, dp_host );
    auto Qdp_exec        = Kokkos::create_mirror_view (exec_space, Qdp_host);

    deep_copy_mirror_view (dp_exec,  dp_host);
    deep_copy_mirror_view (Qdp_exec, Qdp_host);
#ifdef USE_MD_RANGE_POLICY
    constexpr Kokkos::Experimental::Iterate Right = Kokkos::Experimental::Iterate::Right;
    using Rank = Kokkos::Experimental::Rank<3,Right,Right>;
    using RangePolicy = Kokkos::Experimental::MDRangePolicy<Rank,Kokkos::IndexType<int>>;

    Kokkos::Experimental::md_parallel_for(
      RangePolicy({0,0,0},{NUM_LEV,NP,NP},{1,1,1}),
      KOKKOS_LAMBDA (const int ilev, const int igp, const int jgp)
      {
        // MDRangePolicy supports up to rank 3 so far... We need to add the outer layer by hand
        for (int ie=nets_c; ie<nete; ++ie)
        {
          const Real Qt  = Qdp_exec(ie,qn0_c,0,ilev,igp,jgp) / dp_exec(ie,n0_c,ilev,igp,jgp);
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
// Manually add this definition to the config.h.c file before compiling
// (but AFTER configuring) if you want Kokkos to parallelize on elements
// as well. We are using this to compare apples to apples with the fortran
// version, where we manually specify the number of threads for horizontal
// and vertical openmp. Once comparisons are done, we will remove this
// ifdef, and let Kokkos decide teams sizes
#ifdef USE_KOKKOS_ON_ELEMENTS
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

                const Real Qt  = Qdp_exec(ie,qn0_c,0,ilev,igp,jgp) / dp_exec(ie,n0_c,ilev,igp,jgp);
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

  // Copy the output views back onto the host space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view(T_v_host,        T_v_exec);
  deep_copy_mirror_view(kappa_star_host, kappa_star_exec);
}

void caar_preq_hydrostatic_c (const int& nets, const int& nete, const int& nelemd,
                              const int& n0,   RCPtr& phi_ptr,  CRCPtr& phis_ptr,
                              CRCPtr& T_v_ptr, CRCPtr& p_ptr,   CRCPtr& dp_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointers into a unmanaged views
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]> dp_host (dp_ptr, nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]> p_host (p_ptr, nelemd);
  HostViewUnmanaged<const Real*[NP][NP]> phis_host (phis_ptr, nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]> T_v_host (T_v_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]> phi_host (phi_ptr, nelemd);

  // Create the execution space views. If ExecSpace can access HostMemSpace,
  // then the exec view is storing the host view pointer, otherwise is managed
  auto dp_exec   = Kokkos::create_mirror_view (exec_space, dp_host  );
  auto p_exec    = Kokkos::create_mirror_view (exec_space, p_host   );
  auto phis_exec = Kokkos::create_mirror_view (exec_space, phis_host);
  auto T_v_exec  = Kokkos::create_mirror_view (exec_space, T_v_host );
  auto phi_exec  = Kokkos::create_mirror_view (exec_space, phi_host );

  // Temporary view
  ExecViewManaged<Real[NUM_LEV][NP][NP]> phii ("phii");

  // Forward the views (just the inputs, actually) to the execution space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (dp_exec,   dp_host);
  deep_copy_mirror_view (p_exec,    p_host);
  deep_copy_mirror_view (phis_exec, phis_host);
  deep_copy_mirror_view (T_v_exec,  T_v_host);

  // Fix indices: they are coming from fortran where they are 1-based.
  const int nets_c = nets - 1;
  const int n0_c   = n0 - 1;

  Kokkos::parallel_for(
// Manually add this definition to the config.h.c file before compiling
// (but AFTER configuring) if you want Kokkos to parallelize on elements
// as well. We are using this to compare apples to apples with the fortran
// version, where we manually specify the number of threads for horizontal
// and vertical openmp. Once comparisons are done, we will remove this
// ifdef, and let Kokkos decide teams sizes
#ifdef USE_KOKKOS_ON_ELEMENTS
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, Kokkos::AUTO),
#else
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, ExecSpace::thread_pool_size()),
#endif
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      Kokkos::single(
        Kokkos::PerTeam(team_member),
        KOKKOS_LAMBDA ()
        {
          const int ie = nets_c + team_member.league_rank();
          ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> dp  = subview(dp_exec,ie,n0_c,ALL(),ALL(),ALL());
          ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> p   = subview(p_exec,ie,ALL(),ALL(),ALL());
          ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> T_v = subview(T_v_exec,ie,ALL(),ALL(),ALL());
          ExecViewUnmanaged<const Real[NP][NP]> phis = subview(phis_exec,ie,ALL(),ALL());

          ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi = subview(phi_exec,ie,ALL(),ALL(),ALL());

          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, NP * NP),
            KOKKOS_LAMBDA(const int idx)
            {
              const int igp = idx / NP;
              const int jgp = idx % NP;

              Real hkk = dp(NUM_LEV-1,igp,jgp)*0.5/p(NUM_LEV-1,igp,jgp);
              phii(NUM_LEV-1,igp,jgp) = PhysicalConstants::Rgas*T_v(NUM_LEV-1,igp,jgp)*(2.0*hkk);
              phi (NUM_LEV-1,igp,jgp) = phis(igp,jgp) + PhysicalConstants::Rgas*T_v(NUM_LEV-1,igp,jgp)*hkk;
            }
          );

          for (int ilev=NUM_LEV-2; ilev>=0; --ilev)
          {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team_member, NP * NP),
              KOKKOS_LAMBDA(const int idx)
              {
                int igp = idx / NP;
                int jgp = idx % NP;

                Real hkk = dp(ilev,igp,jgp)*0.5/p(ilev,igp,jgp);
                // phii(0,*,*) will never be used. But I loop till ilev=0 anyways, to save one parallel_for dispatch
                phii(ilev,igp,jgp) = phii(ilev+1,igp,jgp) + PhysicalConstants::Rgas*T_v(ilev,igp,jgp)*(2.0*hkk);
                phi (ilev,igp,jgp) = phis(igp,jgp) + phii(ilev+1,igp,jgp) + PhysicalConstants::Rgas*T_v(ilev,igp,jgp)*hkk;
              }
            );
          }
        }
      );
    }
  );

  // Copy the output view(s) back onto the host space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view(phi_host, phi_exec);
}

void caar_preq_omega_ps_c (const int& nets, const int& nete, const int& nelemd,
                           CRCPtr& div_vdp_ptr, CRCPtr& vgrad_p_ptr,
                           CRCPtr& p_ptr,       RCPtr& omega_p_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointers into a unmanaged views
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]> div_vdp_host (div_vdp_ptr, nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]> vgrad_p_host (vgrad_p_ptr, nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]> p_host       (p_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]>       omega_p_host (omega_p_ptr, nelemd);

  // Create the execution space views. If ExecSpace can access HostMemSpace,
  // then the exec view is storing the host view pointer, otherwise is managed
  auto div_vdp_exec = Kokkos::create_mirror_view (exec_space, div_vdp_host);
  auto vgrad_p_exec = Kokkos::create_mirror_view (exec_space, vgrad_p_host );
  auto p_exec       = Kokkos::create_mirror_view (exec_space, p_host   );
  auto omega_p_exec = Kokkos::create_mirror_view (exec_space, omega_p_host );

  // Temporary view
  ExecViewManaged<Real[NP][NP]> suml ("suml");

  // Forward the views (just the inputs, actually) to the execution space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (div_vdp_exec, div_vdp_host);
  deep_copy_mirror_view (vgrad_p_exec, vgrad_p_host);
  deep_copy_mirror_view (p_exec,       p_host);

  // Fix indices: they are coming from fortran where they are 1-based.
  const int nets_c = nets - 1;

  Kokkos::parallel_for(
// Manually add this definition to the config.h.c file before compiling
// (but AFTER configuring) if you want Kokkos to parallelize on elements
// as well. We are using this to compare apples to apples with the fortran
// version, where we manually specify the number of threads for horizontal
// and vertical openmp. Once comparisons are done, we will remove this
// ifdef, and let Kokkos decide teams sizes
#ifdef USE_KOKKOS_ON_ELEMENTS
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, Kokkos::AUTO),
#else
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, ExecSpace::thread_pool_size()),
#endif
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      Kokkos::single(
        Kokkos::PerTeam(team_member),
        KOKKOS_LAMBDA ()
        {
          const int ie = nets_c + team_member.league_rank();
          ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> div_vdp = subview(div_vdp_exec,ie,ALL(),ALL(),ALL());
          ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> vgrad_p = subview(vgrad_p_exec,ie,ALL(),ALL(),ALL());
          ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> p       = subview(p_exec,ie,ALL(),ALL(),ALL());

          ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> omega_p = subview(omega_p_exec,ie,ALL(),ALL(),ALL());

          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, NP * NP),
            KOKKOS_LAMBDA(const int idx)
            {
              const int igp = idx / NP;
              const int jgp = idx % NP;

              Real ckk = 0.5/p(0,igp,jgp);
              omega_p(0,igp,jgp) = vgrad_p(0,igp,jgp)/p(0,igp,jgp) - ckk*div_vdp(0,igp,jgp);
              suml(igp,jgp) = div_vdp(0,igp,jgp);
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

                Real ckk = 0.5/p(ilev,igp,jgp);
                omega_p(ilev,igp,jgp) = vgrad_p(ilev,igp,jgp)/p(ilev,igp,jgp) -2*ckk*suml(igp,jgp) - ckk*div_vdp(ilev,igp,jgp);
                suml(igp,jgp) += div_vdp(ilev,igp,jgp);
              }
            );
          }
        }
      );
    }
  );

  // Copy the output view(s) back onto the host space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view(omega_p_host, omega_p_exec);
}

void caar_compute_eta_dot_dpdn_c (const int& nets, const int& nete,
                                  const int& nelemd, const Real& eta_ave_w,
                                  CRCPtr& omega_p_ptr, RCPtr& eta_dot_dpdn_ptr,
                                  RCPtr& T_vadv_ptr, RCPtr& v_vadv_ptr,
                                  RCPtr& elem_derived_eta_dot_dpdn_ptr,
                                  RCPtr& elem_derived_omega_p_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointers into a unmanaged views
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]> omega_p_host      (omega_p_ptr,      nelemd);
  HostViewUnmanaged<Real*[NUM_LEV_P][NP][NP]>     eta_dot_dpdn_host (eta_dot_dpdn_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]> T_vadv_host (T_vadv_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][2][NP][NP]> v_vadv_host (v_vadv_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_LEV_P][NP][NP]> elem_derived_eta_dot_dpdn_host (elem_derived_eta_dot_dpdn_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]> elem_derived_omega_p_host      (elem_derived_omega_p_ptr,      nelemd);

  // Create the execution space views. If ExecSpace can access HostMemSpace,
  // then the exec view is storing the host view pointer, otherwise is managed
  auto omega_p_exec                   = Kokkos::create_mirror_view (exec_space, omega_p_host );
  auto eta_dot_dpdn_exec              = Kokkos::create_mirror_view (exec_space, eta_dot_dpdn_host );
  auto T_vadv_exec                    = Kokkos::create_mirror_view (exec_space, T_vadv_host );
  auto v_vadv_exec                    = Kokkos::create_mirror_view (exec_space, v_vadv_host );
  auto elem_derived_eta_dot_dpdn_exec = Kokkos::create_mirror_view (exec_space, elem_derived_eta_dot_dpdn_host );
  auto elem_derived_omega_p_exec      = Kokkos::create_mirror_view (exec_space, elem_derived_omega_p_host );

  // Forward the views (just the inputs, actually) to the execution space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (omega_p_exec, omega_p_host);
  deep_copy_mirror_view (elem_derived_eta_dot_dpdn_exec, elem_derived_eta_dot_dpdn_host);
  deep_copy_mirror_view (elem_derived_omega_p_exec, elem_derived_omega_p_host);

  // This code is specifically for rsplit>0. If rsplit==0, then
  // loop on levels (see f90 code)
  deep_copy (eta_dot_dpdn_exec, 0.0);
  deep_copy (T_vadv_exec, 0.0);
  deep_copy (v_vadv_exec, 0.0);

  // Fix indices: they are coming from fortran where they are 1-based.
  const int nets_c = nets - 1;

  Kokkos::parallel_for (
// Manually add this definition to the config.h.c file before compiling
// (but AFTER configuring) if you want Kokkos to parallelize on elements
// as well. We are using this to compare apples to apples with the fortran
// version, where we manually specify the number of threads for horizontal
// and vertical openmp. Once comparisons are done, we will remove this
// ifdef, and let Kokkos decide teams sizes
#ifdef USE_KOKKOS_ON_ELEMENTS
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

              elem_derived_eta_dot_dpdn_exec(ie,ilev,igp,jgp) += eta_ave_w * eta_dot_dpdn_exec(ie,ilev,igp,jgp);
              elem_derived_omega_p_exec(ie,ilev,igp,jgp) += eta_ave_w * omega_p_exec (ie,ilev,igp,jgp);
            }
          );
        }
      );

      Kokkos::single(
        Kokkos::PerTeam(team_member),
        KOKKOS_LAMBDA ()
        {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, NP * NP),
            KOKKOS_LAMBDA (const int idx)
            {
              const int igp = idx / NP;
              const int jgp = idx % NP;

              elem_derived_eta_dot_dpdn_exec(ie,NUM_LEV,igp,jgp) += eta_ave_w * eta_dot_dpdn_exec(ie,NUM_LEV,igp,jgp);
            }
          );
        }
      );
    }
  );

  // Copy the output view(s) back onto the host space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (elem_derived_omega_p_host, elem_derived_omega_p_exec);
  deep_copy_mirror_view (elem_derived_eta_dot_dpdn_host, elem_derived_eta_dot_dpdn_exec);
  deep_copy_mirror_view (eta_dot_dpdn_host, eta_dot_dpdn_exec);
  deep_copy_mirror_view (T_vadv_host, T_vadv_exec);
  deep_copy_mirror_view (v_vadv_host, v_vadv_exec);
}

void caar_compute_phi_kinetic_energy_c (const int& nets, const int& nete,
                                        const int& nelemd, const int& n0,
                                        CRCPtr& p_ptr, CRCPtr& grad_p_ptr, CRCPtr& vort_ptr,
                                        CRCPtr& v_vadv_ptr, CRCPtr& T_vadv_ptr,
                                        CRCPtr& elem_Dinv_ptr, CRCPtr& elem_fcor_ptr,
                                        CRCPtr& elem_state_T_ptr, CRCPtr& elem_state_v_ptr,
                                        CRCPtr& elem_derived_phi_ptr, CRCPtr& elem_derived_pecnd_ptr,
                                        CRCPtr& kappa_star_ptr, CRCPtr& T_v_ptr, CRCPtr& omega_p_ptr,
                                        RCPtr& ttens_ptr, RCPtr& vtens1_ptr, RCPtr& vtens2_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointers into a unmanaged views
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>     p_host          (p_ptr,          nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>     vort_host       (vort_ptr,       nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][2][NP][NP]>  grad_p_host     (grad_p_ptr,     nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>     omega_p_host    (omega_p_ptr,    nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>     T_vadv_host     (T_vadv_ptr,     nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][2][NP][NP]>  v_vadv_host     (v_vadv_ptr,     nelemd);
  HostViewUnmanaged<const Real*[2][2][NP][NP]>        Dinv_host       (elem_Dinv_ptr,  nelemd);
  HostViewUnmanaged<const Real*[NP][NP]>              fcor_host       (elem_fcor_ptr,  nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>     kappa_star_host (kappa_star_ptr, nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>     T_v_host        (T_v_ptr,        nelemd);

  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]>    elem_state_T_host  (elem_state_T_ptr,  nelemd);
  HostViewUnmanaged<const Real*[NUM_TIME_LEVELS][NUM_LEV][2][NP][NP]> elem_state_v_host  (elem_state_v_ptr,  nelemd);

  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]> elem_derived_phi_host   (elem_derived_phi_ptr,  nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]> elem_derived_pecnd_host (elem_derived_pecnd_ptr,  nelemd);

  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]>           ttens_host   (ttens_ptr,   nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]>           vtens1_host  (vtens1_ptr,  nelemd);
  HostViewUnmanaged<Real*[NUM_LEV][NP][NP]>           vtens2_host  (vtens2_ptr,  nelemd);

  // Create the execution space views. If ExecSpace can access HostMemSpace,
  // then the exec view is storing the host view pointer, otherwise is managed
  auto dvv_exec                = get_derivative().get_dvv();
  auto Dinv_exec               = Kokkos::create_mirror_view (exec_space, Dinv_host);
  auto fcor_exec               = Kokkos::create_mirror_view (exec_space, fcor_host);
  auto omega_p_exec            = Kokkos::create_mirror_view (exec_space, omega_p_host);
  auto kappa_star_exec         = Kokkos::create_mirror_view (exec_space, kappa_star_host);
  auto p_exec                  = Kokkos::create_mirror_view (exec_space, p_host);
  auto grad_p_exec             = Kokkos::create_mirror_view (exec_space, grad_p_host);
  auto T_vadv_exec             = Kokkos::create_mirror_view (exec_space, T_vadv_host);
  auto v_vadv_exec             = Kokkos::create_mirror_view (exec_space, v_vadv_host);
  auto vort_exec               = Kokkos::create_mirror_view (exec_space, vort_host);
  auto T_v_exec                = Kokkos::create_mirror_view (exec_space, T_v_host);
  auto elem_state_T_exec       = Kokkos::create_mirror_view (exec_space, elem_state_T_host);
  auto elem_state_v_exec       = Kokkos::create_mirror_view (exec_space, elem_state_v_host);
  auto elem_derived_phi_exec   = Kokkos::create_mirror_view (exec_space, elem_derived_phi_host);
  auto elem_derived_pecnd_exec = Kokkos::create_mirror_view (exec_space, elem_derived_pecnd_host);
  auto ttens_exec              = Kokkos::create_mirror_view (exec_space, ttens_host);
  auto vtens1_exec             = Kokkos::create_mirror_view (exec_space, vtens1_host);
  auto vtens2_exec             = Kokkos::create_mirror_view (exec_space, vtens2_host);

  // Temporaries
  ExecViewManaged<Real[NP][NP]>     Ephi   ("ephi");
  ExecViewManaged<Real[2][NP][NP]>  grad   ("grad_tmp");
  ExecViewManaged<Real[NP][NP]>     vgradT ("vgradT");

  // Forward the views (just the inputs, actually) to the execution space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (Dinv_exec,               Dinv_host);
  deep_copy_mirror_view (fcor_exec,               fcor_host);
  deep_copy_mirror_view (omega_p_exec,            omega_p_host);
  deep_copy_mirror_view (kappa_star_exec,         kappa_star_host);
  deep_copy_mirror_view (p_exec,                  p_host);
  deep_copy_mirror_view (grad_p_exec,             grad_p_host);
  deep_copy_mirror_view (T_vadv_exec,             T_vadv_host);
  deep_copy_mirror_view (v_vadv_exec,             v_vadv_host);
  deep_copy_mirror_view (vort_exec,               vort_host);
  deep_copy_mirror_view (T_v_exec,                T_v_host);
  deep_copy_mirror_view (elem_state_T_exec,       elem_state_T_host);
  deep_copy_mirror_view (elem_state_v_exec,       elem_state_v_host);
  deep_copy_mirror_view (elem_derived_phi_exec,   elem_derived_phi_host);
  deep_copy_mirror_view (elem_derived_pecnd_exec, elem_derived_pecnd_host);

  // Fix indices: they are coming from fortran where they are 1-based.
  const int nets_c = nets - 1;
  const int n0_c   = n0 - 1;

  Kokkos::parallel_for (
// Manually add this definition to the config.h.c file before compiling
// (but AFTER configuring) if you want Kokkos to parallelize on elements
// as well. We are using this to compare apples to apples with the fortran
// version, where we manually specify the number of threads for horizontal
// and vertical openmp. Once comparisons are done, we will remove this
// ifdef, and let Kokkos decide teams sizes
#ifdef USE_KOKKOS_ON_ELEMENTS
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, Kokkos::AUTO),
#else
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, ExecSpace::thread_pool_size()),
#endif
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ie = nets_c + team_member.league_rank();

      // Extracting this element (and time level) slices
      ExecViewUnmanaged<const Real[2][2][NP][NP]>       Dinv       = subview (Dinv_exec, ie, ALL(), ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][2][NP][NP]> v          = subview (elem_state_v_exec, ie, n0_c, ALL(), ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    phi        = subview (elem_derived_phi_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    pecnd      = subview (elem_derived_pecnd_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][2][NP][NP]> v_vadv     = subview (v_vadv_exec, ie, ALL(), ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    T_vadv     = subview (T_vadv_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    kappa_star = subview (kappa_star_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    omega_p    = subview (omega_p_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NP][NP]>             fcor       = subview (fcor_exec, ie, ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    vort       = subview (vort_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    T_v        = subview (T_v_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    p          = subview (p_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][2][NP][NP]> grad_p     = subview (grad_p_exec, ie, ALL(), ALL(), ALL(), ALL());

      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> ttens  = subview (ttens_exec,  ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> vtens1 = subview (vtens1_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> vtens2 = subview (vtens2_exec, ie, ALL(), ALL(), ALL());

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

              Real v1 = v(ilev,0,igp,jgp);
              Real v2 = v(ilev,1,igp,jgp);

              Ephi(igp,jgp) = 0.5*(v1*v1 + v2*v2) + phi(ilev,igp,jgp) + pecnd(ilev,igp,jgp);
            }
          );

          gradient_sphere (team_member, subview (elem_state_T_exec, ie, n0_c, ilev, ALL(), ALL()), dvv_exec, Dinv, grad);

          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, NP * NP),
            KOKKOS_LAMBDA (const int idx)
            {
              const int igp = idx / NP;
              const int jgp = idx % NP;

              Real v1 = v(ilev,0,igp,jgp);
              Real v2 = v(ilev,1,igp,jgp);

              vgradT(igp,jgp) = v1*grad(0,igp,jgp) + v2*grad(1,igp,jgp);
            }
          );

          gradient_sphere (team_member, Ephi, dvv_exec, Dinv, grad);

          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team_member, NP * NP),
            KOKKOS_LAMBDA (const int idx)
            {
              const int igp = idx / NP;
              const int jgp = idx % NP;

              Real gpterm = T_v(ilev,igp,jgp)/p(ilev,igp,jgp);
              Real glnps1 = PhysicalConstants::Rgas*gpterm*grad_p(ilev,0,igp,jgp);
              Real glnps2 = PhysicalConstants::Rgas*gpterm*grad_p(ilev,1,igp,jgp);

              Real v1 = v(ilev,0,igp,jgp);
              Real v2 = v(ilev,1,igp,jgp);

              vtens1(ilev,igp,jgp) = -v_vadv(ilev,0,igp,jgp) + v2*(fcor(igp,jgp) + vort(ilev,igp,jgp)) - grad(0,igp,jgp) - glnps1;
              vtens2(ilev,igp,jgp) = -v_vadv(ilev,1,igp,jgp) - v1*(fcor(igp,jgp) + vort(ilev,igp,jgp)) - grad(1,igp,jgp) - glnps2;

              ttens(ilev,igp,jgp)  = -T_vadv(ilev,igp,jgp) - vgradT(igp,jgp) + kappa_star(ilev,igp,jgp)*T_v(ilev,igp,jgp)*omega_p(ilev,igp,jgp);
            }
          );
        }
      );
    }
  );

  // Copy the output view(s) back onto the host space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (ttens_host, ttens_exec);
  deep_copy_mirror_view (vtens1_host, vtens1_exec);
  deep_copy_mirror_view (vtens2_host, vtens2_exec);
}

void caar_update_states_c (const int& nets, const int& nete, const int& nelemd,
                           const int& nm1, const int& np1, const Real& dt2,
                           const int& rsplit, const int& ntrac, const Real& eta_ave_w,
                           CRCPtr& vdp_ptr, CRCPtr& div_vdp_ptr,
                           CRCPtr& eta_dot_dpdn_ptr, CRCPtr& sdot_sum_ptr,
                           CRCPtr& ttens_ptr, CRCPtr& vtens1_ptr, CRCPtr& vtens2_ptr,
                           CRCPtr& elem_Dinv_ptr, CRCPtr& elem_metdet_ptr, CRCPtr& elem_spheremp_ptr,
                           RCPtr& elem_state_v_ptr, RCPtr& elem_state_T_ptr, RCPtr& elem_state_dp3d_ptr,
                           RCPtr& elem_sub_elem_mass_flux_ptr, RCPtr& elem_state_ps_v_ptr)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Store the f90 pointers into a unmanaged views
  HostViewUnmanaged<const Real*[2][2][NP][NP]>       Dinv_host         (elem_Dinv_ptr,     nelemd);
  HostViewUnmanaged<const Real*[NP][NP]>             metdet_host       (elem_metdet_ptr,   nelemd);
  HostViewUnmanaged<const Real*[NP][NP]>             spheremp_host     (elem_spheremp_ptr, nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][2][NP][NP]> vdp_host          (vdp_ptr,           nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>    div_vdp_host      (div_vdp_ptr,       nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV_P][NP][NP]>  eta_dot_dpdn_host (eta_dot_dpdn_ptr,  nelemd);
  HostViewUnmanaged<const Real*[NP][NP]>             sdot_sum_host     (sdot_sum_ptr,      nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>    ttens_host        (ttens_ptr,         nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>    vtens1_host       (vtens1_ptr,        nelemd);
  HostViewUnmanaged<const Real*[NUM_LEV][NP][NP]>    vtens2_host       (vtens2_ptr,        nelemd);

  HostViewUnmanaged<Real*[NUM_LEV][4][NC][NC]> elem_sub_elem_mass_flux_host (elem_sub_elem_mass_flux_ptr, nelemd);

  HostViewUnmanaged<Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]>    elem_state_T_host    (elem_state_T_ptr,    nelemd);
  HostViewUnmanaged<Real*[NUM_TIME_LEVELS][NUM_LEV][2][NP][NP]> elem_state_v_host    (elem_state_v_ptr,    nelemd);
  HostViewUnmanaged<Real*[NUM_TIME_LEVELS][NUM_LEV][NP][NP]>    elem_state_dp3d_host (elem_state_dp3d_ptr, nelemd);
  HostViewUnmanaged<Real*[NUM_TIME_LEVELS][NP][NP]>             elem_state_ps_v_host (elem_state_ps_v_ptr, nelemd);

  // Create the execution space views. If ExecSpace can access HostMemSpace,
  // then the exec view is storing the host view pointer, otherwise is managed
  auto Dinv_exec            = Kokkos::create_mirror_view (exec_space, Dinv_host);
  auto metdet_exec          = Kokkos::create_mirror_view (exec_space, metdet_host);
  auto spheremp_exec        = Kokkos::create_mirror_view (exec_space, spheremp_host);
  auto vdp_exec             = Kokkos::create_mirror_view (exec_space, vdp_host);
  auto div_vdp_exec         = Kokkos::create_mirror_view (exec_space, div_vdp_host);
  auto ttens_exec           = Kokkos::create_mirror_view (exec_space, ttens_host);
  auto vtens1_exec          = Kokkos::create_mirror_view (exec_space, vtens1_host);
  auto vtens2_exec          = Kokkos::create_mirror_view (exec_space, vtens2_host);
  auto eta_dot_dpdn_exec    = Kokkos::create_mirror_view (exec_space, eta_dot_dpdn_host);
  auto sdot_sum_exec        = Kokkos::create_mirror_view (exec_space, sdot_sum_host);
  auto elem_state_T_exec    = Kokkos::create_mirror_view (exec_space, elem_state_T_host);
  auto elem_state_v_exec    = Kokkos::create_mirror_view (exec_space, elem_state_v_host);
  auto elem_state_dp3d_exec = Kokkos::create_mirror_view (exec_space, elem_state_dp3d_host);
  auto elem_state_ps_v_exec = Kokkos::create_mirror_view (exec_space, elem_state_ps_v_host);
  auto elem_sub_elem_mass_flux_exec = Kokkos::create_mirror_view (exec_space, elem_sub_elem_mass_flux_host);

  // Forward the views (just the inputs, actually) to the execution space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (Dinv_exec,            Dinv_host);
  deep_copy_mirror_view (metdet_exec,          metdet_host);
  deep_copy_mirror_view (spheremp_exec,        spheremp_host);
  deep_copy_mirror_view (elem_state_T_exec,    elem_state_T_host);
  deep_copy_mirror_view (elem_state_v_exec,    elem_state_v_host);
  deep_copy_mirror_view (elem_state_dp3d_exec, elem_state_dp3d_host);
  deep_copy_mirror_view (elem_state_ps_v_exec, elem_state_ps_v_host);

  // Temporaries
  ExecViewManaged<Real[4][NC][NC]>  temp_flux ("temp fluxes");
  ExecViewManaged<Real[2][NP][NP]>  temp_vdp  ("temp vdp");

  // Fix indices: they are coming from fortran where they are 1-based.
  const int nets_c = nets - 1;
  const int nm1_c  = nm1 - 1;
  const int np1_c  = np1 - 1;
  Kokkos::parallel_for (
// Manually add this definition to the config.h.c file before compiling
// (but AFTER configuring) if you want Kokkos to parallelize on elements
// as well. We are using this to compare apples to apples with the fortran
// version, where we manually specify the number of threads for horizontal
// and vertical openmp. Once comparisons are done, we will remove this
// ifdef, and let Kokkos decide teams sizes
#ifdef USE_KOKKOS_ON_ELEMENTS
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, Kokkos::AUTO),
#else
    Kokkos::TeamPolicy<ExecSpace>(nete-nets_c, ExecSpace::thread_pool_size()),
#endif
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team_member)
    {
      const int ie = nets_c + team_member.league_rank();

      // Extracting this element (and time level) slices
      ExecViewUnmanaged<const Real[2][2][NP][NP]>       Dinv         = subview (Dinv_exec, ie, ALL(), ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NP][NP]>             metdet       = subview (metdet_exec, ie, ALL(), ALL());
      ExecViewUnmanaged<const Real[NP][NP]>             spheremp     = subview (spheremp_exec, ie, ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][2][NP][NP]> vdp          = subview (vdp_exec, ie, ALL(), ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    div_vdp      = subview (div_vdp_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV_P][NP][NP]>  eta_dot_dpdn = subview (eta_dot_dpdn_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    ttens        = subview (ttens_exec,  ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    vtens1       = subview (vtens1_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>    vtens2       = subview (vtens2_exec, ie, ALL(), ALL(), ALL());
      ExecViewUnmanaged<const Real[NP][NP]>             sdot_sum     = subview (sdot_sum_exec, ie, ALL(), ALL());


      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>    T_nm1    = subview (elem_state_T_exec, ie, nm1_c, ALL(), ALL(), ALL());
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]> v_nm1    = subview (elem_state_v_exec, ie, nm1_c, ALL(), ALL(), ALL(), ALL());
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>    dp3d_nm1 = subview (elem_state_dp3d_exec, ie, nm1_c, ALL(), ALL(), ALL());
      ExecViewUnmanaged<Real[NP][NP]>             ps_v_nm1 = subview (elem_state_ps_v_exec, ie, nm1_c, ALL(), ALL());

      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>    T_np1    = subview (elem_state_T_exec, ie, np1_c, ALL(), ALL(), ALL());
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]> v_np1    = subview (elem_state_v_exec, ie, np1_c, ALL(), ALL(), ALL(), ALL());
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>    dp3d_np1 = subview (elem_state_dp3d_exec, ie, np1_c, ALL(), ALL(), ALL());
      ExecViewUnmanaged<Real[NP][NP]>             ps_v_np1 = subview (elem_state_ps_v_exec, ie, np1_c, ALL(), ALL());

      ExecViewUnmanaged<Real[NUM_LEV][4][NC][NC]> sub_elem_mass_flux = subview (elem_sub_elem_mass_flux_exec, ie, ALL(), ALL(), ALL(), ALL());

      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(team_member, NP*NP),
        KOKKOS_LAMBDA (const int idx)
        {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          ps_v_np1(igp,jgp) = spheremp (igp,jgp)* (ps_v_nm1(igp,jgp) - dt2*sdot_sum(igp,jgp));
        }
      );

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

              v_np1(ilev,0,igp,jgp) = spheremp(igp,jgp) * (v_nm1(ilev,0,igp,jgp) + dt2*vtens1(ilev,igp,jgp));
              v_np1(ilev,1,igp,jgp) = spheremp(igp,jgp) * (v_nm1(ilev,1,igp,jgp) + dt2*vtens2(ilev,igp,jgp));

              T_np1(ilev,igp,jgp) = spheremp(igp,jgp) * (T_nm1(ilev,igp,jgp) + dt2*ttens(ilev,igp,jgp));

              dp3d_np1(ilev,igp,jgp) = spheremp(igp,jgp) * (dp3d_nm1(ilev,igp,jgp) - dt2*(div_vdp(ilev,igp,jgp)+eta_dot_dpdn(ilev+1,igp,jgp)-eta_dot_dpdn(ilev,igp,jgp)) );
            }
          );

          if (rsplit>0 && ntrac>0 && eta_ave_w>0.0)
          {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team_member, NP * NP),
              KOKKOS_LAMBDA (const int idx)
              {
                const int igp = idx / NP;
                const int jgp = idx % NP;

                temp_vdp(0,igp,jgp) = Dinv(0,0,igp,jgp)*vdp(ilev,0,igp,jgp) + Dinv(0,1,igp,jgp)*vdp(ilev,1,igp,jgp);
                temp_vdp(1,igp,jgp) = Dinv(1,0,igp,jgp)*vdp(ilev,0,igp,jgp) + Dinv(1,1,igp,jgp)*vdp(ilev,1,igp,jgp);
              }
            );
            Kokkos::deep_copy(temp_flux,0.);
            subcell_div_fluxes (team_member, temp_vdp, metdet, temp_flux);
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team_member, 4 * NC * NC),
              KOKKOS_LAMBDA (const int idx)
              {
                const int kcp =  idx / NP;
                const int jgp = (idx / NP) / NP;
                const int igp = (idx / NP) % 4;

                sub_elem_mass_flux (ilev, kcp, igp, jgp) -= temp_flux(kcp,igp,jgp)*eta_ave_w;
              }
            );
          }
        }
      );
    }
  );

  // Copy the output view(s) back onto the host space
  // Note: this is a no-op if ExecSpace can access HostMemSpace
  deep_copy_mirror_view (ttens_host, ttens_exec);
  deep_copy_mirror_view (vtens1_host, vtens1_exec);
  deep_copy_mirror_view (vtens2_host, vtens2_exec);
}

} // extern "C"

} // namespace Homme
