find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

pybind11_add_module(ane ane_module.cpp)
target_link_libraries(ane PRIVATE libane_static)
target_include_directories(ane PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
if(APPLE)
    target_link_libraries(ane PRIVATE
        "-framework Foundation"
        "-framework IOSurface"
        "-framework CoreFoundation"
        "-framework Accelerate"
    )
endif()

install(TARGETS ane DESTINATION . COMPONENT python)
install(FILES ane.pyi DESTINATION . COMPONENT python)
