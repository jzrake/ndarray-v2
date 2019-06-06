# CXXFLAGS = -std=c++17 -O0 -Wextra -fsanitize=undefined
CXXFLAGS = -std=c++17 -O0 -Wextra

HEADERS = ndarray-refactor.hpp sequence.hpp

default: test

test.o: $(HEADERS)

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^

main: main.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	$(RM) *.o test main
