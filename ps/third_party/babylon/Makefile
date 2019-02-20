CFLAGS = -Wall -Wno-format -Wno-sign-compare -std=c++11 -pthread -O3 -mfpmath=sse -fPIC -g -DHAVE_PROTOTYPES
export PADDLE_ROOT=/home/dongdaxiang/lego_trainer_lib
LIBFLAGS = -lprotobuf -lpthread -lrt -ldl -L./third_party/protobuf/lib \

CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"
BUILD_TIME := $(shell date +'%Y%m%d%H%M%S')
BUILD_COMMIT := $(shell git log|head -1|cut -d' ' -f2)
CFLAGS += -DBUILD_TIME=$(BUILD_TIME)\
	-DBUILD_COMMIT=$(BUILD_COMMIT)
CXX = /opt/compiler/gcc-4.8.2/bin/g++

CC = $(CXX) $(CFLAGS)
INCLUDES = -I./include -I../gflags/include -I../glog/include 
LIBNAME = babylon
SRC_DIR = src
LIB_DIR = lib
OUTPUT_DIR = output

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst %.cpp,%.o,$(SOURCES))
DEPS := ${OBJS:.o=.d}

.PHONY: all clean lib

all: lib
	if [ ! -d lib ]; then mkdir -p lib; fi
	if [ ! -d $(OUTPUT_DIR)/lib ]; then mkdir -p $(OUTPUT_DIR)/lib; fi
	cp -r lib/* $(OUTPUT_DIR)/lib
	if [ ! -d $(OUTPUT_DIR)/include ]; then mkdir -p $(OUTPUT_DIR)/include; fi
	cp include/*.h $(OUTPUT_DIR)/include

lib: $(OBJS)
	if [ ! -d $(LIB_DIR) ]; then mkdir $(LIB_DIR); fi
	ar -ruv $(LIB_DIR)/lib$(LIBNAME).a $(OBJS)

$(OBJS): %.o: %.cpp
	$(CC) -c $< -o $@ $(INCLUDES)


clean:
	rm -f $(OBJS)  $(LIB_DIR)/*.a
	rm -f ./src/*.o  ./src/*.d ./*~ src/*~ include/*~ 
	if [ -d $(OUTPUT_DIR) ]; then rm -r $(OUTPUT_DIR); fi
	if [ -d $(LIB_DIR)/so ]; then rm -rf $(LIB_DIR)/so/*.so; fi
