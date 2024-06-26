cmake_minimum_required(VERSION 3.10)
project(FlexFlow)

include(ExternalProject) 

# Set policy CMP0074 to eliminate cmake warnings
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(FLEXFLOW_ROOT ${CMAKE_CURRENT_LIST_DIR})

# Set a default build type if none was specified
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
endif()

# do not disable assertions even if in release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

# only used for pypi
option(FF_BUILD_FROM_PYPI "Build from pypi" OFF)

# build shared or static flexflow lib
option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" ON)

# option for using Python
option(FF_USE_PYTHON "Enable Python" ON)  

# option for using Python
option(FF_USE_GASNET "Run FlexFlow with GASNet" OFF)
set(FF_GASNET_CONDUITS aries udp mpi ibv ucx)
set(FF_GASNET_CONDUIT "mpi" CACHE STRING "Select Gasnet conduit ${FF_GASNET_CONDUITS}")
set_property(CACHE FF_GASNET_CONDUIT PROPERTY STRINGS ${FF_GASNET_CONDUITS})  

# option for cuda arch
set(FF_CUDA_ARCH "" CACHE STRING "Target CUDA Arch")

# option for nccl
option(FF_USE_NCCL "Run FlexFlow with NCCL" OFF)

# option for avx2
option(FF_USE_AVX2 "Run FlexFlow with AVX2" OFF)

# option for max dim
set(FF_MAX_DIM "4" CACHE STRING "Maximum dimention of tensors")

set(FLEXFLOW_EXT_LIBRARIES "")
set(FLEXFLOW_INCLUDE_DIRS "")

# get FLAGS from ENV
set(CC_FLAGS $ENV{CC_FLAGS})
set(NVCC_FLAGS $ENV{NVCC_FLAGS})
set(LD_FLAGS $ENV{LD_FLAGS})

# Set global FLAGS
list(APPEND CC_FLAGS
  -std=c++11)

list(APPEND NVCC_FLAGS
  -std=c++11)
	
add_compile_options(${CC_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS})
link_libraries(${LD_FLAGS})

# ZLIB
include(zlib)

# CUDA
include(cuda)
 
# CUDNN
include(cudnn)

# NCCL
if(FF_USE_NCCL)
  include(nccl)
  list(APPEND FF_CC_FLAGS
    -DFF_USE_NCCL)
  list(APPEND FF_NVCC_FLAGS
    -DFF_USE_NCCL)
endif()

# Legion
include(legion)

if(FF_USE_PYTHON)
  list(APPEND FF_CC_FLAGS
    -DBINDINGS_AUGMENT_PYTHONPATH)
  list(APPEND FF_NVCC_FLAGS
    -DBINDINGS_AUGMENT_PYTHONPATH)
endif()

# Start build FlexFlow
if (CMAKE_BUILD_TYPE EQUAL "Debug")
  list(APPEND FF_CC_FLAGS
    -DFF_DEBUG)
  list(APPEND FF_NVCC_FLAGS
    -DFF_DEBUG)
endif()

message(STATUS "FlexFlow MAX_DIM: ${FF_MAX_DIM}")

list(APPEND FF_CC_FLAGS
  -DMAX_TENSOR_DIM=${FF_MAX_DIM})

if(FF_USE_AVX2)
  list(APPEND FF_CC_FLAGS
    -DFF_USE_AVX2
    -mavx2)
endif()
  
list(APPEND FF_NVCC_FLAGS
  -Wno-deprecated-gpu-targets
  -DMAX_TENSOR_DIM=${FF_MAX_DIM})
  
list(APPEND FF_LD_FLAGS
  -lrt
  -ldl
  -rdynamic)

# Set FF FLAGS
add_compile_options(${FF_CC_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${FF_NVCC_FLAGS})
link_libraries(${FF_LD_FLAGS})

list(APPEND FLEXFLOW_INCLUDE_DIRS
  ${FLEXFLOW_ROOT}/include
  ${FLEXFLOW_ROOT}/fbuf2/include
  ${FLEXFLOW_ROOT})

file(GLOB_RECURSE FLEXFLOW_HDR 
  LIST_DIRECTORIES False 
  ${FLEXFLOW_ROOT}/include/*.h
  ${FLEXFLOW_ROOT}/fbuf2/include/*.h)

file(GLOB_RECURSE FLEXFLOW_SRC 
  LIST_DIRECTORIES False 
  ${FLEXFLOW_ROOT}/src/*.cc)
list(REMOVE_ITEM FLEXFLOW_SRC "${FLEXFLOW_ROOT}/src/runtime/cpp_driver.cc")

file(GLOB_RECURSE FLEXFLOW_GPU_SRC 
  LIST_DIRECTORIES False 
  ${FLEXFLOW_ROOT}/src/*.cu)

set(FLEXFLOW_CPP_DRV_SRC
  ${FLEXFLOW_ROOT}/src/runtime/cpp_driver.cc)  

#message("FLEXFLOW_INCLUDE_DIRS: ${FLEXFLOW_INCLUDE_DIRS}")

# compile flexflow lib
if(BUILD_SHARED_LIBS)
  cuda_add_library(flexflow SHARED ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC} OPTIONS ${CUDA_GENCODE})
else()
  cuda_add_library(flexflow STATIC ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC} OPTIONS ${CUDA_GENCODE})
endif()

target_include_directories(flexflow PUBLIC ${FLEXFLOW_INCLUDE_DIRS})

if(FF_USE_NCCL)
  add_dependencies(flexflow ${NCCL_NAME})
endif()

# build binary
option(FF_BUILD_RESNET "build resnet example" OFF)
option(FF_BUILD_ALEXNET "build alexnet example" OFF)
option(FF_BUILD_DLRM "build DLRM example" OFF)
option(FF_BUILD_INCEPTION "build inception example" OFF)
option(FF_BUILD_CANDLE_UNO "build candle uno example" OFF)
option(FF_BUILD_TRANSFORMER "build transformer example" OFF)
option(FF_BUILD_ALL_EXAMPLES "build all examples. Overrides others" OFF)
option(FF_BUILD_UNIT_TESTS "build non-operator unit tests" OFF)

if(FF_BUILD_TESTS)
  set(BUILD_GMOCK OFF)
  add_subdirectory(deps/googletest)
  enable_testing()
  add_subdirectory(tests/unit)
endif()

# Python
if(FF_USE_PYTHON)
  add_subdirectory(deps/pybind11)
  add_subdirectory(python)
endif()

if(FF_BUILD_RESNET OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/ResNet)
  add_subdirectory(examples/cpp/ResNetsim)
endif()

if(FF_BUILD_ALEXNET OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/AlexNet)
  add_subdirectory(examples/cpp/alexnetsim)
endif()

if(FF_BUILD_INCEPTION OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/InceptionV3)
  add_subdirectory(examples/cpp/InceptionV3sim)
endif()

#TODO: Once functional add to BUILD_ALL_EXAMPLES
if(FF_BUILD_CANDLE_UNO OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/candle_uno)
  add_subdirectory(examples/cpp/candle_unosim)
endif()

if(FF_BUILD_DLRM OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/DLRM)
  add_subdirectory(examples/cpp/DLRMsim)

  #add_executable(generate_dlrm_hetero_strategy src/runtime/dlrm_strategy_hetero.cc)
  #target_include_directories(generate_dlrm_hetero_strategy PUBLIC ${FLEXFLOW_INCLUDE_DIRS})

  #add_executable(generate_dlrm_strategy src/runtime/dlrm_strategy.cc)
  #target_include_directories(generate_dlrm_strategy PUBLIC ${FLEXFLOW_INCLUDE_DIRS})
endif()

if(FF_BUILD_TRANSFORMER OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/Transformer)
  add_subdirectory(examples/cpp/Transformersim)
endif()

if(FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/mixture_of_experts)
  add_subdirectory(examples/cpp/mixture_of_expertssim)
endif()

if(FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/mixture_of_experts_bak)
  add_subdirectory(examples/cpp/mixture_of_expertssim_bak)
endif()

if(FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/vgg16sim)
endif()

if(FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/ncfsim)
endif()

# installation
set(INCLUDE_DEST "include")
set(LIB_DEST "lib")
install(FILES ${FLEXFLOW_HDR} DESTINATION ${INCLUDE_DEST})
install(TARGETS flexflow DESTINATION ${LIB_DEST})
