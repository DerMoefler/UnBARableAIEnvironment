#include "UnBARableAI/engine_information.hpp"
#include "serialization/engine_information.tpp"
#include "serialization/trivial_serializable.hpp"

namespace {

using namespace UnBARableAINS;
using namespace serialization;

static_assert(Serializable<EngineInformation>, "EngineInformation cannot be properly serialized");
constexpr EngineInformation engineInformation{EngineStatus::RUNNING, GrpcConnection{12345}, true};
constexpr auto serializedEngineInformation = serialize<EngineInformation>(engineInformation);
constexpr auto deserializedEngineInformation =
    deserialize<EngineInformation>(serializedEngineInformation);
static_assert(engineInformation == deserializedEngineInformation,
              "(De-)Serialization of EngineInformation not working");

}  // namespace
