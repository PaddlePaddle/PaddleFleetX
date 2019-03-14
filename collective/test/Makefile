CC      = g++
CFLAGS  = -std=c++11 -fPIC -O3 -DNDEBUG

CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

DGC_INCLUDE_DIR = $(PREFIX)/include
DGC_LIBRARY = $(PREFIX)/lib/libdgc.a

INCLUDEFLAGS = -I$(DGC_INCLUDE_DIR)
LDFLAGS = $(DGC_LIBRARY)

ifneq ($(NCCL_HOME), "")
INCLUDEFLAGS += -I$(NCCL_HOME)/include/
LDFLAGS += -L$(NCCL_HOME)/lib
endif
LDFLAGS += -lnccl

ifneq ($(MPI_HOME), "")
INCLUDEFLAGS += -I$(MPI_HOME)/include/
LDFLAGS += -L$(MPI_HOME)/lib
endif
LDFLAGS += -lmpi_cxx -lopen-rte -lopen-pal -lmpi

INCLUDEFLAGS += -isystem $(CUDA_HOME)/include
LDFLAGS += -rdynamic $(CUDA_HOME)/lib64/libcudart.so $(CUDA_HOME)/lib64/libcurand.so $(CUDA_HOME)/lib64/libnvrtc.so -lpthread -Wl,-rpath,$(CUDA_HOME)/lib64 -ldl

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

NVCUFLAGS := -ccbin $(CC) $(NVCC_GENCODE) $(INCLUDEFLAGS) -std=c++11  --compiler-options '-fPIC'

ifeq ($(DEBUG), 1)
NVCUFLAGS += -DDEBUG_PRINT
CC  += -DDEBUG_PRINT
endif

#PREFIX := ../build
BIN_DIR := $(PREFIX)/bin
OBJ_DIR := $(PREFIX)/obj
DIRS = $(shell find . -maxdepth 3 -type d)
SRCS := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cc))
OBJS := $(SRCS:%.cc=$(OBJ_DIR)/%.cc.o)
TARGETS := $(SRCS:%.cc=$(BIN_DIR)/%)

CU_SRCS := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cu))
CU_OBJS := $(CU_SRCS:%.cu=$(OBJ_DIR)/%.cu.o)

all : test

test : $(TARGETS)

$(BIN_DIR)/% : $(OBJ_DIR)/%.cc.o $(CU_OBJS) 
	@printf "Linking: $@ <-- $^"
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.cc.o : %.cc
	@printf "Compiling: $@ <-- $<"
	mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDEFLAGS) -o $@ -c $<

$(OBJ_DIR)/%.cu.o : %.cu
	@printf "Compiling: $@ <-- $<"
	mkdir -p $(OBJ_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

clean:
	rm -rf $(PREFIX)
