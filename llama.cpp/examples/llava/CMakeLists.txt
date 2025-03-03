add_library(llava OBJECT
            llava.cpp
            llava.h
            clip.cpp
            clip.h
            )

target_link_libraries(llava PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(llava PUBLIC .)
target_include_directories(llava PUBLIC ../..)
target_include_directories(llava PUBLIC ../../common)

target_compile_features(llava PRIVATE cxx_std_17)

add_library(llava_static STATIC $<TARGET_OBJECTS:llava>)
if (BUILD_SHARED_LIBS)
    set_target_properties(llava PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llava PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(llava_shared SHARED $<TARGET_OBJECTS:llava>)
    target_link_libraries(llava_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS llava_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(llava PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(llava BUILD_INFO)
endif()

set(TARGET llama-llava-cli)
add_executable(${TARGET} llava-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-llava-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-minicpmv-cli)
add_executable(${TARGET} minicpmv-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-minicpmv-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-qwen2vl-cli)
add_executable(${TARGET} qwen2vl-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-qwen2vl-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-llava-clip-quantize-cli)
add_executable(${TARGET} clip-quantize-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-llava-clip-quantize-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)



# TinyXML2 추가
add_library(TinyXML2 STATIC tinyxml2.cpp)
target_include_directories(TinyXML2 PUBLIC .)

# TinyEXIF 추가 (TinyXML2 의존)
add_library(TinyEXIF STATIC TinyEXIF.cpp)
target_include_directories(TinyEXIF PUBLIC .)
target_link_libraries(TinyEXIF PRIVATE TinyXML2)

# ONNX Runtime 및 OpenCV 라이브러리 경로 설정 (절대 경로 변환)
get_filename_component(ONNX_RUNTIME_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../onnxruntime" ABSOLUTE)
get_filename_component(OPENCV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../opencv" ABSOLUTE)

set(ONNX_RUNTIME_LIB "${ONNX_RUNTIME_PATH}/build/Windows/Release/Release/onnxruntime.lib")
set(OPENCV_LIB "${OPENCV_PATH}/install/x64/vc17/lib/opencv_world4120.lib")

# ONNX Runtime 및 OpenCV include 경로 추가
target_include_directories(llama-qwen2vl-cli PRIVATE 
    "${ONNX_RUNTIME_PATH}/include/onnxruntime/core/session" 
    "${OPENCV_PATH}/install/include"
)

# ONNX Runtime 및 OpenCV 라이브러리 추가
target_link_libraries(llama-qwen2vl-cli PRIVATE 
    common llava ${CMAKE_THREAD_LIBS_INIT} 
    ${ONNX_RUNTIME_LIB} 
    ${OPENCV_LIB}
    TinyEXIF TinyXML2
)

# 경로 디버깅 메시지 출력
message(STATUS "Computed ONNX Runtime Path: ${ONNX_RUNTIME_LIB}")
message(STATUS "Computed OpenCV Path: ${OPENCV_LIB}")

# 라이브러리 존재 여부 확인 후 오류 처리
if(NOT EXISTS ${ONNX_RUNTIME_LIB})
    message(FATAL_ERROR "ONNX Runtime library not found: ${ONNX_RUNTIME_LIB}")
endif()

if(NOT EXISTS ${OPENCV_LIB})
    message(FATAL_ERROR "OpenCV library not found: ${OPENCV_LIB}")
endif()
