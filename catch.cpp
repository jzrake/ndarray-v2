// work-around for g++ bug on MacOS, include something from C++ standard lib
// before anything else:
#include <tuple>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
