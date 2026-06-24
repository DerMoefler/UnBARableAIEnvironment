#pragma once
#include <type_traits>

/**
* \brief Metafunction to use for triggering a static_assert in portions of code, that should never be compiled.
* \tparam T An arbitrary type.
* Do not partially specialized this Metafunction ever since its entire purpose is always evaluate to false.
*/
template<typename T>
struct AlwaysFalse_MF : std::false_type {};