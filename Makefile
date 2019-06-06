CXXFLAGS = -std=c++17 -O0 -Wextra

HEADERS = ndarray.hpp sequence.hpp

default: test

test.o: $(HEADERS)

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test
