#include "Connectivity.hpp"
#include "BoundaryExchange.hpp"

#include <map>

namespace Homme
{

extern "C"
{

void init_connectivity (const int& num_local_elems, const int& num_local_connections, const int& num_shared_connections)
{
  Connectivity& connectivity = get_connectivity();
  connectivity.set_num_my_elems(num_local_elems);
  connectivity.set_num_connections(num_local_connections,num_shared_connections);
}

void add_connection (const int& first_elem_lid,  const int& first_elem_pos,  const int& first_elem_pid,
                     const int& second_elem_lid, const int& second_elem_pos, const int& second_elem_pid)
{
  /// check that F90 is in base 1
  if (first_elem_lid==0  || first_elem_pos==0  || first_elem_pid==0 ||
      second_elem_lid==0 || second_elem_pos==0 || second_elem_pid==0)
  {
    std::cout << "ERROR! We were assuming F90 indices started at 1, but it appears there is an exception.\n";
    std::abort();
  }

  Connectivity& connectivity = get_connectivity();
  connectivity.add_connection(first_elem_lid-1, first_elem_pos-1, first_elem_pid-1,
                   second_elem_lid-1,second_elem_pos-1,second_elem_pid-1);
}

void finalize_connectivity ()
{
  Connectivity& connectivity = get_connectivity();

  connectivity.finalize();
}

void cleanup_mpi_structures ()
{
  std::map<std::string,BoundaryExchange> be = get_all_boundary_exchange ();
  for (auto it : be) {
    it.second.clean_up();
  }
}

} // extern "C"

} // namespace Homme
