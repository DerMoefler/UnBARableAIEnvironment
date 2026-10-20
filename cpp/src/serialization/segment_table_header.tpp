#ifndef SEGMENT_TABLE_HEADER_TPP_
#define SEGMENT_TABLE_HEADER_TPP_

#include "serialization/serialize_information.h"

namespace UnBARableAINS {

namespace serialization {

template <>
struct SerializeInformation<memory::SegmentTableHeader> {};

}  // namespace serialization

}  // namespace UnBARableAINS

#endif
