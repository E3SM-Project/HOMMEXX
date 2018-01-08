#include "Context.hpp"
#include "Connectivity.hpp"
#include "BoundaryExchange.hpp"

#include <map>

namespace Homme
{

extern "C"
{

void init_connectivity (const int& num_local_elems)
{
  Connectivity& connectivity = Context::singleton().get_connectivity();
  connectivity.set_num_elements(num_local_elems);
}

void add_connection (const int& first_elem_lid,  const int& first_elem_gid,  const int& first_elem_pos,  const int& first_elem_pid,
                     const int& second_elem_lid, const int& second_elem_gid, const int& second_elem_pos, const int& second_elem_pid)
{
  // Check that F90 is in base 1
  // Note: an elem lid=0 means not on this process, so we accept it. Obviously, that is acceptable
  //       only for the remote, but here we do not know which one is the remote and which one is the local, yet
  if (first_elem_lid<0  || first_elem_gid<=0  || first_elem_pos<=0  || first_elem_pid<=0 ||
      second_elem_lid<0 || second_elem_gid<=0 || second_elem_pos<=0 || second_elem_pid<=0)
  {
    std::cout << "ERROR! We were assuming F90 indices started at 1, but it appears there is an exception.\n";
    std::abort();
  }

  Connectivity& connectivity = Context::singleton().get_connectivity();
  connectivity.add_connection(first_elem_lid-1, first_elem_gid-1, first_elem_pos-1, first_elem_pid-1,
                              second_elem_lid-1,second_elem_gid-1,second_elem_pos-1,second_elem_pid-1);
}

void finalize_connectivity ()
{
  Connectivity& connectivity = Context::singleton().get_connectivity();

  connectivity.finalize();
}

void cleanup_mpi_structures ()
{
  std::map<std::string,BoundaryExchange>& be = Context::singleton().get_boundary_exchanges ();
  for (auto& it : be) {
    it.second.clean_up();
  }
}

} // extern "C"

} // namespace Homme
