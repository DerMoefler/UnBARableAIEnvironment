from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeDeps, CMakeToolchain


class UnBARableAIRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"

    def requirements(self):
        self.requires("grpc/1.78.1")
        self.requires("protobuf/6.33.5", override=True)
        self.requires("gtest/1.17.0")
        self.requires("pybind11/2.13.6")

    def build_requirements(self):
        self.tool_requires("protobuf/6.33.5")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)

        # Verhindert das Schreiben von CMakeUserPresets.json
        # ins read-only Source-Root (/build/cpp)
        tc.user_presets_path = False

        tc.generate()