CXXFLAGS = -std=c++17 -O0 -Wextra -fsanitize=undefined
# CXXFLAGS = -std=c++17 -O3 -Wextra

HEADERS = ndarray.hpp

default: test

test.o: $(HEADERS)

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test
