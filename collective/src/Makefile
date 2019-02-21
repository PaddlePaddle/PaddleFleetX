CC      = g++
CFLAGS  = -std=c++11 -fPIC -O3 -DNDEBUG

CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

INCLUDEFLAGS = -isystem $(CUDA_HOME)/include
LDFLAGS = -rdynamic $(CUDA_HOME)/lib64/libcudart.so  $(CUDA_HOME)/lib64/libnvrtc.so -lpthread -Wl,-rpath,$(CUDA_HOME)/lib64 -ldl

ifneq ($(NCCL_INCLUDE), "")
INCLUDEFLAGS += -I$(NCCL_INCLUDE)
endif

CUDA_VERSION = $(strip $(shell $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)

CUDA8_GENCODE = -gencode=arch=compute_30,code=sm_30 \
		-gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_52,code=sm_52 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61
CUDA9_GENCODE = -gencode=arch=compute_70,code=sm_70

CUDA8_PTX     = -gencode=arch=compute_61,code=compute_61
CUDA9_PTX     = -gencode=arch=compute_70,code=compute_70

# Include Volta support if we're using CUDA9 or above
ifeq ($(shell test "$(CUDA_MAJOR)" -gt 8; echo $$?),0)
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA9_PTX)
else
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA8_PTX)
endif

NVCUFLAGS := -ccbin $(CC) $(NVCC_GENCODE) $(INCLUDEFLAGS) -std=c++11 --compiler-options '-fPIC'

ifeq ($(DEBUG), 1)
NVCUFLAGS += -DDEBUG_PRINT
CC  += -DDEBUG_PRINT
endif

#BUILD_PREFIX ?= ../build
BIN_DIR := $(BUILD_PREFIX)/bin
OBJ_DIR := $(BUILD_PREFIX)/obj
INC_DIR := $(BUILD_PREFIX)/include
LIB_DIR := $(BUILD_PREFIX)/lib

INCEXPORT = dgc.h
INCTARGET = $(patsubst %, $(INC_DIR)/%, $(INCEXPORT))
LIBNAME = libdgc.so
LIBTARGET = $(patsubst %, $(LIB_DIR)/%, $(LIBNAME))
STATICLIBNAME = libdgc.a
STATICLIBTARGET = $(patsubst %, $(LIB_DIR)/%, $(STATICLIBNAME))

DIRS = $(shell find . -maxdepth 3 -type d)
SRCS := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cc))
OBJS := $(SRCS:%.cc=$(OBJ_DIR)/%.cc.o)

CU_SRCS := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cu))
CU_OBJS := $(CU_SRCS:%.cu=$(OBJ_DIR)/%.cu.o)

.PHONY : staticlib lib clean

$(OBJ_DIR)/%.cc.o : %.cc
	@printf "debug: $@ <-- $<"
	mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDEFLAGS) -o $@ -c $<

$(OBJ_DIR)/%.cu.o : %.cu
	@printf "debug: $@ <-- $<"
	mkdir -p $(OBJ_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

$(LIB_DIR)/libdgc.so : $(OBJS) $(CU_OBJS)
	@printf "Linking   %-35s\n" $@
	mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,libdgc.so -o $@ $(LDFLAGS) $(OBJS) $(CU_OBJS)

$(LIB_DIR)/%.a : $(OBJS) $(CU_OBJS)
	@printf "Archiving %-35s\n" $@
	mkdir -p $(LIB_DIR)
	ar cr $@ $(OBJS) $(CU_OBJS)

$(INC_DIR)/%.h : %.h
	@printf "Grabbing %-35s > %s\n" $< $@
	mkdir -p $(INC_DIR)
	cp -f $< $@

lib : $(INCTARGET) $(LIBTARGET)

staticlib : $(INCTARGET) $(STATICLIBTARGET)

install:
	mkdir -p $(INSTALL_PREFIX)/lib
	mkdir -p $(INSTALL_PREFIX)/include
	cp -P -v $(BUILD_PREFIX)/lib/* $(INSTALL_PREFIX)/lib/
	cp -v $(BUILD_PREFIX)/include/* $(INSTALL_PREFIX)/include/

clean:
	rm -rf $(BUILD_PREFIX)
