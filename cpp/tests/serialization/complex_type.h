#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

namespace test {

/**
 * \brief A dummy test object containing some data members to test different Layouts on.
 * This class is intended to be derived from so different SerializeInformation specializations for
 * essentially the same data can be constructed and tested/verified against each other.
 */
struct ComplexBase {
    /// \brief A tensor of third order (storing ints) to test nesting multiple vector's in each other.
    std::vector<std::vector<std::vector<int>>> tensor3;
};

/// \brief Object specifying to inline \ref ComplexBase::tensor3
struct ComplexA : ComplexBase {};

/// \brief Object specifying to not inline \ref ComplexBase::tensor3
struct ComplexB : ComplexBase {};

} // namespace test

template<>
struct SerializeInformation<test::ComplexA> {
    inline static constexpr bool c_serializable = true;

    using Type = test::ComplexA;

    struct F_Tensor3 : public detail::Field_t {
        using Type = std::vector<std::vector<std::vector<int>>>;
    };

    using Fields = detail::Fields<F_Tensor3>;
    
    inline static constexpr decltype(auto) get(std::type_identity<F_Tensor3>, const Type& v) noexcept {
        return v.tensor3;
    }
};

template<>
struct SerializeInformation<test::ComplexB> {
    inline static constexpr bool c_serializable = true;

    using Type = test::ComplexB;

    struct F_Tensor3 : public detail::Field_t {
        using Type = std::vector<std::vector<std::vector<int>>>;
        inline static constexpr bool c_inline = false;
    };

    using Fields = detail::Fields<F_Tensor3>;
    
    inline static constexpr decltype(auto) get(std::type_identity<F_Tensor3>, const Type& v) noexcept {
        return v.tensor3;
    }
};

} // namespace serialization

} // namespace UnBARableAI