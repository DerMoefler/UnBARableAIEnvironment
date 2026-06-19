#pragma once
#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../memory/shared_memory_types.h"

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
concept Serializable = SerializeInformation<T>::c_serializable;

namespace detail {

    /**
    * \brief Metafunction to use for triggering a static_assert in portions of code, that should never be compiled.
    * \tparam T An arbitrary type.
    * Do not partially specialized this Metafunction ever since its entire purpose is always evaluate to false.
    */
    template<typename T>
    struct AlwaysFalse_MF : std::false_type {};

    /// \brief Tag Type for a simple Field.
    struct Field_t {};

    /// \brief Tag Type for a Field holding multiple other Fields.
    struct MultiField_t {};

    /**
    * \brief Metafunction to check whether a type qualifies as a simple field.
    * \tparam T Type to check.
    * Checks whether \p T derives from \ref Field_t.
    */
    template<typename T>
    struct IsField_MF
        : std::bool_constant<std::derived_from<T, detail::Field_t>> 
    {};

    /**
    * \brief Metafunction to check whether a type qualifies as a MultiField.
    * \tparam T Type to check.
    * \todo fix declaration and partial specializations documentation
    */
    template<typename T, typename = void>
    struct IsMultifield_MF : std::false_type {};

    /**
    * \brief Metafunction to check whether a type qualifies as any Field.
    * \tparam T Type to check.
    * Checks whether \p T is a simple field or MultiField.
    */
    template<typename T>
    struct IsFieldlike_MF
        : std::bool_constant<IsField_MF<T>::value || IsMultifield_MF<T>::value>
    {};

    /**
    * \brief Metafunction to check whether a type qualifies as a MultiField.
    * \tparam T Type to check.
    * Actual implementation of the metafunction. Requires T::Field to name a Fieldlike and \p T to derive from \ref MultiField_t.
    */
    template<typename T>
        requires requires { typename T::Field; }
    struct IsMultifield_MF<T>
        : std::bool_constant<
            std::derived_from<T, MultiField_t> &&
            IsFieldlike_MF<typename T::Field>::value
        >
    {};

    /**
    * \brief Concept for a simple field.
    * \tparam T Type to check.
    * Conceptually, a (simple or multi) Field is some serializable Type in a datastructure. This could for example be a float "health" in
    * a datastructure describing a unit. A Field is identified by some Type satisfying this concept. Such a "Field Type" is just a tag type,
    * deriving from the relevant tag (either \ref Field_t or \ref MultiField_t).
    */
    template<typename T>
    concept Field           = detail::IsField_MF<T>::value;

    /**
    * \brief Concept for a field of fields.
    * \tparam T Type to check.
    */
    template<typename T>
    concept MultiField      = detail::IsMultifield_MF<T>::value;

    /**
    * \brief Concept for a Fieldlike, meaning either a \ref Field or \ref MultiField.
    * \tparam T Type to check.
    * \anchor FieldlikeConcept
    */
    template<typename T>
    concept Fieldlike       = detail::IsFieldlike_MF<T>::value; 

    /**
    * \brief Struct to hold a parameter pack of \ref Fieldlike%s.
    * \tparam F... Fieldlikes.
    */
    template<Fieldlike... F>
    struct Fields;

    /**
    * \brief Concept to check whether a \ref FieldlikeConcept "Fieldlike" specifies how to inline it.
    * \tparam F \ref FieldlikeConcept "Fieldlike" to check.
    */
    template<typename F>
    concept HasInlineOverride = Fieldlike<F> && requires { { F::c_inline } -> std::convertible_to<bool>; };

    /**
    * \brief Function to check whether to inline a \ref Fieldlike.
    * \tparam F Fieldlike to check.
    * If not specified explicitly (see \ref HasInlineOverride), defaults to:
    *
    * - true for \ref Field
    * - false for \ref MultiField
    */
    template<Fieldlike F>
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
            static_assert(AlwaysFalse_MF<F>::value, "Implmentation error");
        };
    }

    /**
    * \brief Concept to check whether the inlineability of Type is defined in its \ref SerializeInformation.
    * \tparam P Type to check.
    */
    template<typename T>
    concept SpecifiesInlineability = requires { { SerializeInformation<T>::c_inline } -> std::convertible_to<bool>; };

    /**
    * \brief Function to check whether to inline a Type.
    * \tparam T Type to check.
    * If not specified explicitly (see \ref SpecifiesInlineability), defaults to true.
    */
    template<typename T>
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
    */
    template<typename T>
    concept ConstSize = requires {
        { SerializeInformation<T>::c_serialized_size } -> std::convertible_to<size_t>;
    };

    template<Serializable T>
    size_t getSize(const T& serializable);


    template<Field F>
    size_t getFieldSize(const typename F::Type& value) {
        using Type = typename F::Type;
        // End of recursion
        if constexpr (detail::ConstSize<Type>) {
            return SerializeInformation<Type>::c_serialized_size;
        }
        // Continue recusion through possible subfields if the type is to be inlined
        else if constexpr (inlineType<Type>()){
            return getSize<Type>(value);
        }
        // End recursion because the Type is not inlined and will be referenced using a link_t
        else {
            return sizeof(memory::link_t);
        }
    }

    template<MultiField F>
    size_t getFieldSize(const typename F::Field::Type& value) {
        using Type = typename F::Field::Type;
        // End of recursion
        if constexpr (detail::ConstSize<Type>) {
            return SerializeInformation<Type>::c_serialized_size;
        }
        // Continue recusion through possible subfields if the type is to be inlined
        else if constexpr (inlineType<Type>()){
            return getSize<Type>(value);
        }
        // End recursion because the Type is not inlined and will be referenced using a link_t
        else {
            return sizeof(memory::link_t);
        }
    }

    template<Serializable T, typename U>
    struct GetSizeImpl_MF;

    template<Serializable T, Fieldlike... Fs>
    struct GetSizeImpl_MF<T, Fields<Fs...>> {
    private:
        template<Fieldlike F>
        inline static constexpr size_t getOne(const T& serializable) {
            using SI = SerializeInformation<T>;
            constexpr std::type_identity<F> fieldKey{};
            if constexpr(inlineField<F>()) {
                if constexpr (IsField_MF<F>::value) {
                    decltype(auto) fieldData = SI::get(fieldKey, serializable);
                    return getFieldSize<F>(fieldData);
                }
                else {
                    size_t totalSize {0};
                    size_t numElements = SI::getSize(fieldKey, serializable);
                    for (size_t i = 0; i < numElements; i++) {
                        auto fieldData = SI::get(fieldKey, serializable, i);
                        totalSize += getFieldSize<F>(fieldData);
                    }
                    return totalSize;
                }
            }
            else {
                return sizeof(memory::link_t);
            }
        }
    public:
        inline static constexpr size_t get(const T& serializable) {
            return (getOne<Fs>(serializable) + ... + 0);
        }
    };

    template<Serializable T>
    size_t getSize(const T& serializable) {
        using Fields = typename SerializeInformation<T>::Fields;
        return detail::GetSizeImpl_MF<T, Fields>::get(serializable);
    }

}; // namespace detail

template<Serializable T>
struct FieldSizes {
    static size_t get(const T& serializable) {
        return detail::getSize(serializable);
    }
};

/**
 * \brief Partial specialization for integral types.
 * \todo Document
 */
template<std::integral T>
struct SerializeInformation<T> {
    using Type = T;

    inline static constexpr bool    c_serializable = true;
    inline static constexpr size_t  c_serialized_size = sizeof(Type);

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
template<typename T, typename Alloc>
struct SerializeInformation<std::vector<T, Alloc>> {
    inline static constexpr bool c_serializable = SerializeInformation<T>::c_serializable;
    inline static constexpr bool c_inline       = false;

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

    inline static constexpr decltype(auto) get(std::type_identity<F_Elements>, const Type& v, const size_t index) noexcept {
        return v[index];
    }

    inline static constexpr size_t getSize(std::type_identity<F_Elements>, const Type& v) noexcept {
        return v.size();
    }
};

}; // namespace unit

}; // namespace UnBARableAI