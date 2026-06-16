#pragma once
#include <array>
#include <vector>
#include <concepts>

namespace UnBARableAINS {

namespace serialization {

/**
 * \brief A traits struct to hold information about how to serialize a type.
 * \tparam T Type to get serialization information for.
 * This is the base declaration. If no partial specialization exists, this is used as a default fallback,
 * meaning no way to serialize \p T exists (yet).
 */
template<typename T>
struct SerializeInformation {
    /**
     * \brief Specifies if \p T is serializable.
     * Should be true for every partial specialization. If you depend on some condition, make sure to static_assert(c_serializable, "Condition xy not met")
     */
    inline static constexpr bool c_serializable = false;
};

/**
 * \brief A concept for types that are serializable.
 * \tparam T Type to check.
 */
template<typename T>
concept IsSerializable = SerializeInformation<T>::c_serializables;

/**
 * \brief Partial specialization for integral types.
 * \todo Document
 */
template<std::integral T>
struct SerializeInformation<T> {
    inline static constexpr bool c_serializable = true;
    inline static constexpr bool c_inlineable   = true;
    inline static constexpr std::array<std::byte, sizeof(T)> serialize(const T value) {
        std::array<std::byte, sizeof(T)> result{};
        for (size_t i = 0; i < sizeof(T); ++i) {
            result[i] = std::byte((value >> ((sizeof(T) - 1 - i) * 8)) & 0xFF);
        }
        return result;
    };
};

/**
 * \brief Partial specialization for vector.
 * \todo Document
 */
template<typename T, typename Alloc>
struct SerializeInformation<std::vector<T, Alloc>> {
    inline static constexpr bool c_serializable = SerializeInformation<T>::c_serializable;
    static_assert(c_serializable, "Vector's type is not serializable.");
    inline static constexpr bool c_inlineable   = false;
    inline static std::vector<std::byte> serialize(const std::vector<T, Alloc>& vector) {
        const size_t numElements = vector.size();
        std::vector<std::byte> result{sizeof(size_t) + numElements * sizeof(T)};
        const auto lengthSerialized = SerializeInformation<size_t>::serialize(numElements);
        for (int i = 0; i < lengthSerialized.size(); i++) {
            result[i] = lengthSerialized[i];
        }
        for (int i = 0; i < numElements; i++) {
            const auto elementSerialized = SerializeInformation<T>::serialize(vector[i]);
            for (int j = 0; j < elementSerialized.size(); j++) {
                result[sizeof(size_t) + i * sizeof(T) + j] = elementSerialized[j];
            }
        }
        return result;
    };
};

}; // namespace unit

}; // namespace UnBARableAI