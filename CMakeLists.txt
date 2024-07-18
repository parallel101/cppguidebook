cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)

project(main)

if (MSVC)
    add_compile_options(/Zc:preprocessor /utf-8 /DNOMINMAX /D_USE_MATH_DEFINES /EHsc /bigobj)
else()
    if (WIN32)
        add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
    endif()
    add_compile_options(-Wall -Wextra -Werror=return-type)
endif()


find_package(fmt REQUIRED)
find_package(Boost REQUIRED COMPONENTS locale)

file(GLOB sources CONFIGURE_DEPENDS "examples/*.cpp")
foreach(source IN LISTS sources)
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE fmt::fmt Boost::locale)
    target_include_directories(${name} PRIVATE include)
endforeach()
