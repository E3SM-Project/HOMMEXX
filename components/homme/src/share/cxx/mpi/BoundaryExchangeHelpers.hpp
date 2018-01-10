#ifndef HOMMEXX_BOUNDARY_EXCHANGE_HELPERS_HPP
#define HOMMEXX_BOUNDARY_EXCHANGE_HELPERS_HPP

#include <memory>

namespace Homme
{

// Forward declarations
class BoundaryExchange;
class BuffersManager;

// This free function is the only function that can add the BM to the BE and register the BE as
// customer of the BM. This is to make sure that all and only the customers of BM contain a
// valid ptr to BM. Otherwise, one would need to REMEMBER to set both of them into each other.
void create_buffers_provider_customer_relationship (std::shared_ptr<BuffersManager>   bm,
                                                    std::shared_ptr<BoundaryExchange> be);

} // namespace Homme

#endif // HOMMEXX_BOUNDARY_EXCHANGE_HELPERS_HPP
