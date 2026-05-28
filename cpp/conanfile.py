from conan import ConanFile
from conan.tools.cmake import cmake_layout


class UnBARableAIRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("grpc/1.78.1")
        self.requires("protobuf/6.33.5", override = True)
        self.requires("gtest/1.17.0")

    def build_requirements(self):
        self.tool_requires("protobuf/6.33.5")

    def layout(self):
        cmake_layout(self)