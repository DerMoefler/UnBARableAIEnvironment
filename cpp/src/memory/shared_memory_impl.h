#pragma once

namespace UnBARableAI {

namespace memory {

/**
 * \brief Requirements for a shared memory backend.
 *
 * A shared memory implementation is used to write identifiable segments into memory.
 * This means, an implementation should at least provide a version at the start of the shared
 * memory followed by segments starting with a length and an id:
 \verbatim
 vers.maj | vers.min | vers.patch
 length0          
 id0
 \p data0
 length1
 id1
 \p data1
 ...
 lengthN
 idn
 \p dataN
 \endverbatim
 */
template<class T>
concept SharedMemoryImpl = true;

}; // namespace memory

}; // namespace UnBARableAI

