#ifndef TRIVIAL_SERIALIZABLE_H_
#define TRIVIAL_SERIALIZABLE_H_

#include <type_traits>

#include "serialization/serialize_information.h"
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
 * \brief Computer the size of a TrivialSerializable.
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

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // TRIVIAL_SERIALIZABLE_H_
