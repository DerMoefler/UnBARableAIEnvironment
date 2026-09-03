#pragma once
#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "byte_container.h"
#include "utility/always_false.h"
#include "utility/typelist.h"

namespace UnBARableAINS {

namespace serialization {

/**
 * \brief A traits struct to hold information about how to serialize a type.
 * \tparam T Type to get serialization information for.
 * This is the base declaration. If no partial specialization exists, this is used as a default
 * fallback, meaning no way to serialize \p T exists (yet).
 */
template <typename T>
struct SerializeInformation {
    /**
     * \brief Specifies if \p T is serializable.
     * Should be true for every partial specialization. If you depend on some condition, make sure
     * to static_assert(c_serializable, "Condition xy not met")
     */
    inline static constexpr bool c_serializable = false;
};

/**
 * \brief A concept for types that are serializable.
 * \tparam T Type to check.
 */
template <typename T>
concept Serializable = SerializeInformation<T>::c_serializable;

namespace detail {
/// \brief Tag Type for a simple Field.
struct Field_t {};

/// \brief Tag Type for a Field holding multiple other Fields.
struct MultiField_t {};

/**
 * \brief Metafunction to check whether a type qualifies as a simple field.
 * \tparam T Type to check.
 * Checks whether \p T derives from \ref Field_t.
 */
template <typename T>
struct IsField_MF : std::bool_constant<std::derived_from<T, detail::Field_t>> {};

/**
 * \brief Metafunction to check whether a type qualifies as a MultiField.
 * \tparam T Type to check.
 * \todo fix declaration and partial specializations documentation
 */
template <typename T>
struct IsMultifield_MF : std::false_type {};

/**
 * \brief Metafunction to check whether a type qualifies as any Field.
 * \tparam T Type to check.
 * Checks whether \p T is a simple field or MultiField.
 */
template <typename T>
struct IsFieldlike_MF : std::bool_constant<IsField_MF<T>::value || IsMultifield_MF<T>::value> {};

/**
 * \brief Metafunction to check whether a type qualifies as a MultiField.
 * \tparam T Type to check.
 * Actual implementation of the metafunction. Requires T::Field to name a Fieldlike and \p T to
 * derive from \ref MultiField_t.
 */
template <typename T>
    requires requires { typename T::Field; }
struct IsMultifield_MF<T> : std::bool_constant<std::derived_from<T, MultiField_t> &&
                                               IsFieldlike_MF<typename T::Field>::value> {};

/**
 * \brief Concept for a simple field.
 * \tparam T Type to check.
 * Conceptually, a (simple or multi) Field is some serializable Type in a datastructure. This could
 * for example be a float "health" in a datastructure describing a unit. A Field is identified by
 * some Type satisfying this concept. Such a "Field Type" is just a tag type, deriving from the
 * relevant tag (either \ref Field_t or \ref MultiField_t).
 */
template <typename T>
concept Field = detail::IsField_MF<T>::value;

/**
 * \brief Concept for a field of fields.
 * \tparam T Type to check.
 */
template <typename T>
concept MultiField = detail::IsMultifield_MF<T>::value;

/**
 * \brief Concept for a Fieldlike, meaning either a simple Field or MultiField.
 * \tparam T Type to check.
 * \anchor FieldlikeConcept
 */
template <typename T>
concept Fieldlike = detail::IsFieldlike_MF<T>::value;

/**
 * \brief Typelist of \ref Fieldlike%s.
 * \tparam F... Fieldlikes.
 * Struct to hold a parameter pack so you can define some operations on them.
 */
template <Fieldlike... F>
struct Fields : Typelist::Typelist<F...> {};

/**
 * \brief Concept to check whether a \ref FieldlikeConcept "Fieldlike" specifies how to inline it.
 * \tparam F \ref FieldlikeConcept "Fieldlike" to check.
 */
template <typename F>
concept HasInlineOverride = Fieldlike<F> && requires {
    { F::c_inline } -> std::convertible_to<bool>;
};

/**
 * \brief Function to check whether to inline a \ref Fieldlike.
 * \tparam F Fieldlike to check.
 * If not specified explicitly (see \ref HasInlineOverride), defaults to:
 *
 * - true for \ref Field
 * - false for \ref MultiField
 */
template <Fieldlike F>
consteval bool inlineField(void) {
    if constexpr (HasInlineOverride<F>) {
        return F::c_inline;
    }
    else if constexpr (Field<F>) {
        return true;
    }
    else if constexpr (MultiField<F>) {
        return false;
    }
    else {
        static_assert(AlwaysFalse_MF<F>::value, "Implementation error");
    };
}

/**
 * \brief Concept to check whether the inlineability of Type is defined in its \ref
 * SerializeInformation.
 * \tparam P Type to check.
 * \see \ref ConstSizeConcept "ConstSize": Inlineability is ignored for Types satisfying ConstSize.
 */
template <typename T>
concept SpecifiesInlineability = requires {
    { SerializeInformation<T>::c_inline } -> std::convertible_to<bool>;
};

/**
 * \brief Function to check whether to inline a Type.
 * \tparam T Type to check.
 * If not specified explicitly (see \ref SpecifiesInlineability), defaults to true.
 */
template <typename T>
consteval bool inlineType(void) {
    if constexpr (SpecifiesInlineability<T>) {
        return static_cast<bool>(SerializeInformation<T>::c_inline);
    }
    else {
        return true;
    }
}

/**
TODO move doc
There are two categories of types:
- those with constant size, defined by the constant c_serialized_size
- those with variable size, requiring a member function getSize(const T& value);
*/

/**
 * \brief Concept to check whether a fixed serialized size is available or not.
 * \tparam T Type to check.
 * \note A Type with a constant size will always be inlined. Specifying c_inline for a field
 * which has a Type with constant size will just be ignored.
 * \anchor ConstSizeConcept
 */
template <typename T>
concept ConstSize = requires {
    { SerializeInformation<T>::c_serialized_size } -> std::convertible_to<size_t>;
};

/**
 * \brief Metafunction to get the ValueType for a \ref FieldlikeConcept "Fieldlike".
 * \tparam F The \ref FieldlikeConcept "Fieldlike".
 */
template <typename F>
struct GetFieldlikeValueType_MF;

/**
 * \brief Implementation for GetFieldlikeValueType_MF for a simple Field.
 * \tparam F The Field.
 */
template <Field F>
struct GetFieldlikeValueType_MF<F> {
    using Type = F::Type;
};

/**
 * \brief Implementation for GetFieldlikeValueType_MF for a MultiField.
 * \tparam F The MultiField.
 */
template <MultiField F>
struct GetFieldlikeValueType_MF<F> {
    using Type = F::Field::Type;
};

template <typename S>
concept SerializeMethodAvailable = Serializable<S> && requires(const S value) {
    { SerializeInformation<S>::serialize(value) } -> ByteContainer;
};

}  // namespace detail

// Forward declaration for the shared_ptr in FieldNode.
template <Serializable T>
class Layout;

/**
 * \brief Common information about memory offset and sizes for \ref FieldlikeConcept "Fieldlike".
 * \tparam F The \ref FieldlikeConcept "Fieldlike".
 */
template <typename F>
struct FieldNodeCommon {
    /// \brief The offset within the parent's data.
    size_t offset = 0;
    /// \brief The size occupied in the parent's data.
    size_t inlineSize = 0;
    /// \brief The entire size occupied in memory by this field.
    size_t deepSize = 0;

    /// \brief Whether the field is inlined or not.
    inline static constexpr bool isInlined = detail::inlineField<F>();
};

/**
 * \brief A struct to store information about the memory layout of a Field.
 * \tparam F A \ref FieldlikeConcept "Fieldlike".
 */
template <typename F>
struct FieldNode;

/**
 * \brief Implementation of a FieldNode for a simple Field.
 * \tparam F The simple Field.
 */
template <detail::Field F>
struct FieldNode<F> : FieldNodeCommon<F> {
    /// \brief Type alias for the Field's Tagtype.
    using Tag = F;
    /// \brief Type alias for the underlying Type of the value for the Field \p F.
    using ValueType = F::Type;

    /// \brief Pointer to the Layout of the ValueType if either the field itself or the type isn't
    /// inlined.
    std::shared_ptr<Layout<ValueType>> child = nullptr;
};

/**
 * \brief Implementation of a FieldNode for a MultiField.
 * \tparam F The MultiField.
 */
template <detail::MultiField F>
struct FieldNode<F> : FieldNodeCommon<F> {
    /// \brief Type alias for the Field's Tagtype.
    using Tag = F;
    /// \brief Type alias for the underlying Type of the value for the Field \p F.
    using ValueType = F::Field::Type;

    /// \brief The amount of elements contained in the field.
    size_t count = 0;
    /// \brief \ref count "Count times" pointers to the Layout of the ValueType if either the field
    /// itself or the type isn't inlined.
    std::vector<std::shared_ptr<Layout<ValueType>>> children;
};

template <typename N>
consteval bool isNodeInlined(void) {
    using Tag = typename std::remove_cvref_t<N>::Tag;
    using ValueType = typename std::remove_cvref_t<N>::ValueType;
    return detail::inlineField<Tag> && detail::inlineType<ValueType>();
}

// TODO could just as well be implemented as a member function
template <typename N, typename F>
void visitNodeChildLayouts(N& node, F&& func) {
    using Tag = typename std::remove_cvref_t<decltype(node)>::Tag;
    if constexpr (!detail::Fieldlike<Tag>) {
        static_assert(AlwaysFalse_MF<Tag>::value, "Tag is not a Fieldlike.");
    }
    if constexpr (serialization::detail::MultiField<Tag>) {
        const auto& children = node.children;
        for (const auto& child : children) {
            if (child) {
                func(child);
            }
        }
    }
    else if constexpr (serialization::detail::Field<Tag>) {
        if (node.child) {
            func(node.child);
        }
    }
    else {
        static_assert(AlwaysFalse_MF<Tag>::value,
                      "The Fieltype of 'Tag' is not currently implemented for this function.");
    }
}

/**
 * \brief Partial specialization for integral types.
 * \todo Document
 */
template <std::integral T>
struct SerializeInformation<T> {
    using Type = T;
    // TODO fix (or maybe this is nice, I dunno)
    using Fields = detail::Fields<>;

    inline static constexpr bool c_serializable = true;
    inline static constexpr size_t c_serialized_size = sizeof(Type);

    inline static constexpr std::array<std::byte, sizeof(Type)> serialize(const Type value) {
        std::array<std::byte, sizeof(Type)> result{};
        for (size_t i = 0; i < sizeof(Type); ++i) {
            result[i] = std::byte((value >> ((sizeof(Type) - 1 - i) * 8)) & 0xFF);
        }
        return result;
    };
};

/**
 * \brief Partial specialization for vector.
 * \todo Document
 */
template <typename T, typename Alloc>
struct SerializeInformation<std::vector<T, Alloc>> {
    inline static constexpr bool c_serializable = SerializeInformation<T>::c_serializable;
    inline static constexpr bool c_inline = false;

    using Type = std::vector<T, Alloc>;

    struct F_Length : public detail::Field_t {
        using Type = size_t;
    };

    struct F_Element : public detail::Field_t {
        using Type = T;
    };

    struct F_Elements : public detail::MultiField_t {
        using Field = F_Element;
        // TODO check if needed
        inline static constexpr bool c_inline = true;
    };

    using Fields = detail::Fields<F_Length, F_Elements>;

    inline static constexpr size_t get(std::type_identity<F_Length>, const Type& v) noexcept {
        return v.size();
    }

    inline static constexpr decltype(auto) get(std::type_identity<F_Elements>, const Type& v,
                                               const size_t index) noexcept {
        return v[index];
    }

    inline static constexpr size_t getSize(std::type_identity<F_Elements>, const Type& v) noexcept {
        return v.size();
    }
};

static_assert(detail::SerializeMethodAvailable<int>, "Cannot serialize integers!");

};  // namespace serialization

};  // namespace UnBARableAINS
