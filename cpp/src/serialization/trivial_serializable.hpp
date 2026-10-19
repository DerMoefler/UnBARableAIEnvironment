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

}  // namespace serialization

}  // namespace UnBARableAINS

#endif  // TRIVIAL_SERIALIZABLE_H_
