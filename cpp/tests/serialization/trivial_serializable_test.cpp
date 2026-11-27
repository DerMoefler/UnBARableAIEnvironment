#include "serialization/serialize_information.h"
#include "serialization/trivial_serializable.hpp"

namespace {

using namespace UnBARableAINS;
using namespace serialization;

static_assert(!TrivialSerializable<std::vector<int>>,
              "std::vector<int> must not be a TrivialSerializable.");

}  // namespace
