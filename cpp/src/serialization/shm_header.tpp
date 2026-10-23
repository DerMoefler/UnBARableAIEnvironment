#ifndef SHM_HEADER_TPP_
#define SHM_HEADER_TPP_

#include <stdexcept>
#include <type_traits>

#include "memory/shm_header.hpp"
#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<memory::ShmHeader> {
    using Type = memory::ShmHeader;

    static constexpr bool c_serializable = true;

    struct F_Magic : detail::Field_t {
        using Type = uint32_t;
    };

    struct F_Version : detail::Field_t {
        using Type = uint32_t;
    };

    using Fields = detail::Fields<F_Magic, F_Version>;

    static constexpr uint32_t get(std::type_identity<F_Magic>, const Type value) {
        return Type::c_UnBARableAI_magic;
    }

    static constexpr uint32_t get(std::type_identity<F_Version>, const Type value) {
        return Type::c_version;
    }

    static constexpr Type constructFromFields(const uint32_t magic, const uint32_t version) {
        // TODO check if this is the correct way to do this (the whole thing isnt that pretty but I
        // like it still :D )
        if (magic != Type::c_UnBARableAI_magic) {
            throw std::invalid_argument("UnBARableAI magic is wrong");
        }
        if (version != Type::c_version) {
            throw std::invalid_argument("Shared memory version's differ");
        }
        return Type{};
    }
};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
