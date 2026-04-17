CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter

test_layer: test_layer.cpp conv_scalar.hpp
	$(CXX) $(CXXFLAGS) -o $@ test_layer.cpp

clean:
	rm -f test_layer

.PHONY: clean
