#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

auto main(int argc, char* argv[]) -> int {
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] [%n::%!] %v");

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
