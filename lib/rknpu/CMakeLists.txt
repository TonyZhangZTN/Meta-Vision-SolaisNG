add_library(
        libRGA
        SHARED
        IMPORTED
       	GLOBAL
	)
set_target_properties(
	libRGA PROPERTIES
	IMPORTED_LOCATION
	${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/RK3588/lib/Linux/aarch64/librga.so
	)

add_library(
        libRKNNRT
        SHARED
        IMPORTED
	GLOBAL
	)	
set_target_properties(
	libRKNNRT PROPERTIES
	IMPORTED_LOCATION
	${CMAKE_CURRENT_SOURCE_DIR}/RK3588/Linux/librknn_api/aarch64/librknnrt.so
	)
