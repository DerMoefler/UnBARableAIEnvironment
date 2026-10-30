#ifndef TRIVIAL_SERIALIZABLE_H_
#define TRIVIAL_SERIALIZABLE_H_

#include <algorithm>
#include <type_traits>
#include <tuple>
#include <utility>

#include "serialization/serialize_information.h"
#include "utility/always_false.h"
#include "utility/typelist.h"

namespace UnBARableAINS {

namespace serialization {

namespace detail {

/**
 * \brief Metafunction to check that a field has constant size.
 * \param F Fieldlike to check.
 *
 * There are essentially three cases:
 * - \p F is a MultiField -> Field does not have constant size
 * - \p F is a Field and its \ref UnderlyingType satisfies ConstSize -> Field has constant size
 * - \p The Fields of the \ref UnderlyingType have to be checked recursively.
 */
template <detail::Fieldlike F>
struct IsFieldConstantSize_MF {
    /// \brief Underlying type of \p F.
    using UnderlyingType = typename GetFieldlikeValueType_MF<F>::Type;
    /// \brief Fields of the \ref UnderlyingType.
    using Fields = typename SerializeInformation<UnderlyingType>::Fields;

    /// \brief Implemenation.
    static consteval bool impl(void) {
        if constexpr (MultiField<F>) {
            return true;
        }
        else if constexpr (ConstSize<UnderlyingType>) {
            return true;
        }
        else {
            return Typelist::AllOf_v<Fields, IsFieldConstantSize_MF>;
        }
    }

    /// \brief Computed result.
    static constexpr bool value = impl();
};

/// \brief Metafunction wrapping \ref isFieldInlined and isTypeInlined. Allows use as Predicate.
template <detail::Fieldlike F>
struct IsFieldFullyInlined_MF
    : std::bool_constant<isFieldInlined<F>() &&
                         isTypeInlined<typename GetFieldlikeValueType_MF<F>::Type>()> {};

/// \brief Checks that all fields are fully inlined and have constant size.
template <Serializable S>
struct TrivialSerializableImpl_MF {
    using Fields = typename SerializeInformation<S>::Fields;
    static constexpr bool value = Typelist::AllOf_v<Fields, IsFieldFullyInlined_MF> &&
                                  Typelist::AllOf_v<Fields, IsFieldConstantSize_MF>;
};

/// \brief Value template for \ref TrivialSerializableImpl_MF.
template <Serializable S>
static constexpr bool TrivialSerializableImpl_v = TrivialSerializableImpl_MF<S>::value;

}  // namespace detail

/**
 * \brief Concept to check that a Serializable is trivially serializable.
 * \tparam S Serializable to check.
 *
 * Trivially serializable means the size of \p S can be computed at compile time and it can be
 * serialized as one, meaning all its fields are fully inlined.
 */
template <typename S>
concept TrivialSerializable =
    detail::TrivialSerializableImpl_v<S> || detail::FundamentalSerializable<S>;

// Forward declaration
template <TrivialSerializable S>
consteval std::size_t getSerializableSize(std::type_identity<S>);

namespace detail {

/**
 * \brief Metafunction to sum the sizes of Fields.
 * \tparam Fields Typelist of Fieldlikes to sum.
 */
template <typename Fields>
struct SumFieldSizes_MF;

/**
 * \brief Implemenation to sum field sizes.
 * \tparam Fs The actual fields.
 */
template <detail::Fieldlike... Fs>
struct SumFieldSizes_MF<detail::Fields<Fs...>>
    : std::integral_constant<
          std::size_t,
          (getSerializableSize(std::type_identity<typename GetFieldlikeValueType_MF<Fs>::Type>{}) +
           ... + std::size_t{0})> {};

}  // namespace detail

/**
 * \brief Compute the size of a TrivialSerializable.
 * \tparam S A TrivialSerializable.
 * \returns Constant serialized size of \p S.
 */
template <TrivialSerializable S>
consteval std::size_t getSerializableSize(std::type_identity<S>) {
    // Base case
    if constexpr (detail::ConstSize<S>) {
        return SerializeInformation<S>::c_serialized_size;
    }
    // Recursive case
    else {
        using Fields = typename SerializeInformation<S>::Fields;
        return detail::SumFieldSizes_MF<Fields>::value;
    }
}

/**
 * \brief Compute the offset of \p F in \p S.
 * \tparam S A TrivialSerializable.
 * \tparam F Field inside TrivialSerializable.
 * \returns The offset.
 */
template <TrivialSerializable S, detail::Fieldlike F>
consteval std::size_t getFieldOffset(std::type_identity<S>, std::type_identity<F>) {
    using Fields = typename SerializeInformation<S>::Fields;
    constexpr std::size_t fieldIndex = Typelist::FindType_MF<Fields, F>::value;
    if constexpr (!fieldIndex) {
        return 0;
    }
    else {
        using FieldsBefore = Typelist::Sublist_T<Fields, 0, fieldIndex>;
        return detail::SumFieldSizes_MF<FieldsBefore>::value;
    }
}

/**
 * \brief Metafunction that computes the View (std::span with fixed size) for a TrivialSerializable.
 * \tparam S A TrivialSerializable.
 */
template <TrivialSerializable S>
struct GetSerializableView_MF {
    using Type = std::span<const std::byte, getSerializableSize(std::type_identity<S>{})>;
};

/// \brief Type alias for \ref TrivialSerializable::Type.
template <TrivialSerializable S>
using GetSerializableView_T = typename GetSerializableView_MF<S>::Type;

/**
 * \brief Serialize a TrivialSerializable,
 * \tparam S A TrivialSerializable.
 * \param value The value of type \p S to serialize.
 * \returns The serialized representation of \p value.
 */
template <TrivialSerializable S>
constexpr auto serialize(const S& value)
    -> std::array<std::byte, getSerializableSize(std::type_identity<S>{})> {
    // Base case
    if constexpr (detail::SerializeMethodAvailable<S>) {
        return SerializeInformation<S>::serialize(value);
    }
    // Recursive case
    else {
        static_assert(!detail::SerializeMethodAvailable<S>,
                      "Serialize method should not be available");
        using Fields = typename SerializeInformation<S>::Fields;
        using ValueTypes = Typelist::Transform_T<Fields, detail::GetFieldlikeValueType_MF>;

        std::array<std::byte, getSerializableSize(std::type_identity<S>{})> result{};

        // Nested immediately invoked lambdaes: Serialize each field and copy it into result.
        [&]<std::size_t... Is>(std::index_sequence<Is...>) -> void {
            (
                [&]() -> void {
                    using Field = Typelist::TypeAtIndex_T<Fields, Is>;
                    using ValueType = Typelist::TypeAtIndex_T<ValueTypes, Is>;

                    constexpr std::size_t offset =
                        getFieldOffset(std::type_identity<S>{}, std::type_identity<Field>{});
                    constexpr std::size_t size =
                        getSerializableSize(std::type_identity<ValueType>{});
                    const std::array<std::byte, size> serialized =
                        serialize(SerializeInformation<S>::get(std::type_identity<Field>{}, value));

                    std::ranges::copy(serialized, result.begin() + offset);
                }(),
                ...);
        }(std::make_index_sequence<Typelist::Size_MF<Fields>::value>{});

        return result;
    }
}

/**
 * \brief Deserialize a TrivialSerializable.
 * \tparam S A TrivialSerializable.
 * \param serialized Serialized representation (obtainable through \ref serialize).
 * \returns The original value.
 */
template <TrivialSerializable S>
constexpr S deserialize(const GetSerializableView_T<S> serialized) {
    // Base case
    if constexpr (detail::DeserializeMethodAvailable<S>) {
        return SerializeInformation<S>::deserialize(serialized);
    }
    // Recursive case
    else if constexpr (detail::ConstructibleFromFields<S>) {
        using Fields = typename SerializeInformation<S>::Fields;
        using ValueTypes = Typelist::Transform_T<Fields, detail::GetFieldlikeValueType_MF>;
        using ValueTypesTuple = Typelist::Rebind_T<std::tuple, ValueTypes>;

        // Build the tuple.
        const auto values = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> ValueTypesTuple {
            return ValueTypesTuple{[&]() {
                using Field = Typelist::TypeAtIndex_T<Fields, Is>;
                using ValueType = Typelist::TypeAtIndex_T<ValueTypes, Is>;

                constexpr std::size_t offset =
                    getFieldOffset(std::type_identity<S>{}, std::type_identity<Field>{});
                constexpr std::size_t size = getSerializableSize(std::type_identity<ValueType>{});
                auto fieldView = serialized.template subspan<offset, size>();
                return deserialize<ValueType>(fieldView);
            }()...};
        }(std::make_index_sequence<Typelist::Size_MF<Fields>::value>{});

        // Unpack the tuple and call constructFromFields.
        return std::apply(
            []<typename... Values>(Values&&... values) -> S {
                return SerializeInformation<S>::constructFromFields(
                    std::forward<Values>(values)...);
            },
            std::move(values));
    }
    // Error
    else {
        static_assert(AlwaysFalse_MF<S>::value,
                      "Trivial serializable must either provide a deserialize method or be "
                      "constructible from its fields");
    }
}

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // TRIVIAL_SERIALIZABLE_H_
