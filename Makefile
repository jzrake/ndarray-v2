CXXFLAGS = -std=c++17 -O0 -Wextra -fsanitize=undefined
# CXXFLAGS = -std=c++17 -O3 -Wextra

HEADERS = ndarray.hpp ndarray-refactor.hpp sequence.hpp

default: test main

main.o: $(HEADERS)

test.o: $(HEADERS)

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

main: main.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test main
