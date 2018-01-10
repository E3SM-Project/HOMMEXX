#include "BoundaryExchangeHelpers.hpp"

#include "BoundaryExchange.hpp"
#include "BuffersManager.hpp"

namespace Homme
{

void create_buffers_provider_customer_relationship (std::shared_ptr<BuffersManager>   bm,
                                                    std::shared_ptr<BoundaryExchange> be)
{
  be->set_buffers_manager(bm);
  bm->add_customer(be);
}

} // namespace Homme
