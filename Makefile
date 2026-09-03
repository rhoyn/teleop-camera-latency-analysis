CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pthread
OPENCV         := opencv5
OPENCV_PREFIX  ?= /opt/opencv5
PKGC           := PKG_CONFIG_PATH=$(OPENCV_PREFIX)/lib/pkgconfig pkg-config

# pkg-config --libs pulls in every module; link only what we use.
CXXFLAGS += $(shell $(PKGC) --cflags $(OPENCV))
LDLIBS   := $(shell $(PKGC) --libs-only-L $(OPENCV)) \
            -Wl,-rpath,$(shell $(PKGC) --variable=libdir $(OPENCV)) \
            -pthread -lopencv_freetype -lopencv_objdetect -lopencv_geometry \
            -lopencv_videoio -lopencv_imgproc -lopencv_core

BIN := build/teleop-camera-latency-analysis
SRC := main.cpp

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC) | build
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

build:
	@mkdir -p $@

clean:
	rm -rf build
