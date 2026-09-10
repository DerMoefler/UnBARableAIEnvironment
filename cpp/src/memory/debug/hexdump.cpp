#include "hexdump.hpp"

#include <array>
#include <cstddef>
#include <iomanip>

namespace UnBARableAINS {

namespace memory {

namespace debug {

void hexdump(std::ostream& out, std::istream& in) {
    constexpr std::size_t bytesPerLine = 16;

    std::array<unsigned char, bytesPerLine> lineBuffer{};
    std::size_t offset = 0;

    // Preserve caller's output stream formatting.
    const auto flags = out.flags();
    const auto fill = out.fill();

    while (in) {
        in.read(reinterpret_cast<char*>(lineBuffer.data()),
                static_cast<std::streamsize>(lineBuffer.size()));
        const auto bytesRead = static_cast<std::size_t>(in.gcount());

        if (bytesRead == 0) {
            break;
        }

        out << std::hex << std::setw(8) << std::setfill('0') << offset << "  ";

        for (std::size_t i = 0; i < bytesPerLine; i++) {
            if (i < bytesRead) {
                out << std::setw(2) << static_cast<unsigned>(lineBuffer[i]) << ' ';
            }
            else {
                out << "  ";
            }
            if (i == 7) {
                out << ' ';
            }
        }

        out << " |";

        for (std::size_t i = 0; i < bytesRead; i++) {
            const unsigned char byte = lineBuffer[i];
            out << (std::isprint(byte) ? static_cast<char>(byte) : '.');
        }
        out << "|\n";

        offset += bytesRead;
    }
    // Restore output stream formatting
    out.flags(flags);
    out.fill(fill);
}

}  // namespace debug

}  // namespace memory

}  // namespace UnBARableAINS
