#include "ndarray.hpp"
#include "catch.hpp"




//=============================================================================
TEST_CASE("shapes can be constructed", "[shape]")
{
    auto shape1 = nd::make_shape(10, 10, 10);
    auto shape2 = nd::make_shape(10, 10, 5);

    REQUIRE(shape1 != shape2);
    REQUIRE_FALSE(shape1 == shape2);
    REQUIRE(shape1.size() == 3);
    REQUIRE(shape1.volume() == 1000);
    REQUIRE(shape1.contains(0, 0, 0));
    REQUIRE(shape1.contains(9, 9, 9));
    REQUIRE_FALSE(shape1.contains(10, 9, 9));
}

TEST_CASE("shapes support insertion and removal of elements")
{
    auto shape = nd::make_shape(0, 1, 2);
    REQUIRE(shape.remove_elements(nd::make_index(0, 1)) == nd::make_shape(2));
    REQUIRE(shape.remove_elements(nd::make_index(1, 2)) == nd::make_shape(0));
    REQUIRE(shape.remove_elements(nd::make_index(0, 2)) == nd::make_shape(1));
    REQUIRE(shape.insert_elements(nd::make_index(0, 1), nd::make_shape(8, 9)) == nd::make_shape(8, 9, 0, 1, 2));
    REQUIRE(shape.insert_elements(nd::make_index(1, 2), nd::make_shape(8, 9)) == nd::make_shape(0, 8, 9, 1, 2));
    REQUIRE(shape.insert_elements(nd::make_index(2, 3), nd::make_shape(8, 9)) == nd::make_shape(0, 1, 8, 9, 2));
    REQUIRE(shape.insert_elements(nd::make_index(3, 4), nd::make_shape(8, 9)) == nd::make_shape(0, 1, 2, 8, 9));
}

TEST_CASE("can zip, transform, enumerate a range", "[range] [transform] [zip] [divvy]")
{
    auto n = 0;

    for (auto a : nd::range(10) | nd::transform([] (auto a) { return 2 * a; }))
    {
        REQUIRE(a == 2 * n);
        ++n;
    }
    for (auto&& [m, n] : nd::zip(nd::range(10), nd::range(10)))
    {
        REQUIRE(m == n);
    }
    for (auto&& [m, n] : enumerate(nd::range(10)))
    {
        REQUIRE(m == n);
    }

    REQUIRE(nd::divvy(10)(nd::range(10)).size() == 10);
    REQUIRE(nd::divvy(4)(nd::range(100)).size() == 4);
    REQUIRE(nd::divvy(3)(nd::range(100)).size() == 3);

    n = 0;

    for (auto group : nd::range(20) | nd::divvy(3))
    {
        for (auto item : group)
        {
            REQUIRE(item == n);
            ++n;
        }
    }

    n = 0;

    for (auto group : nd::range(20) | nd::divvy(5))
    {
        for (auto item : group)
        {
            REQUIRE(item == n);
            ++n;
        }
    }

    n = 0;

    for (auto group : nd::range(20) | nd::divvy(22))
    {
        for (auto item : group)
        {
            REQUIRE(item == n);
            ++n;
        }
    }
}

TEST_CASE("range can be constructed", "[distance] [enumerate] [range]")
{
    REQUIRE(nd::distance(nd::enumerate(nd::range(10))) == 10);
}

TEST_CASE("buffer works as expected", "[buffer]")
{
    SECTION("can instantiate an empty buffer")
    {
        nd::buffer_t<double> B;
        REQUIRE(B.size() == 0);
        REQUIRE(B.data() == nullptr);
    }

    SECTION("can instantiate a constant buffer")
    {
        nd::buffer_t<double> B(100, 1.5);
        REQUIRE(B.size() == 100);
        REQUIRE(B.data() != nullptr);
        REQUIRE(B[0] == 1.5);
        REQUIRE(B[99] == 1.5);
    }

    SECTION("can instantiate a buffer from input iterator")
    {
        std::vector<int> A{0, 1, 2, 3};
        nd::buffer_t<double> B(A.begin(), A.end());
        REQUIRE(B.size() == 4);
        REQUIRE(B[0] == 0);
        REQUIRE(B[1] == 1);
        REQUIRE(B[2] == 2);
        REQUIRE(B[3] == 3);
    }

    SECTION("can move-construct and move-assign a buffer")
    {
        nd::buffer_t<double> A(100, 1.5);
        nd::buffer_t<double> B(200, 2.0);

        B = std::move(A);

        REQUIRE(A.size() == 0);
        REQUIRE(A.data() == nullptr);

        REQUIRE(B.size() == 100);
        REQUIRE(B[0] == 1.5);
        REQUIRE(B[99] == 1.5);

        auto C = std::move(B);

        REQUIRE(B.size() == 0);
        REQUIRE(B.data() == nullptr);
        REQUIRE(C.size() == 100);
        REQUIRE(C[0] == 1.5);
        REQUIRE(C[99] == 1.5);
    }

    SECTION("equality operators between buffers work correctly")
    {
        nd::buffer_t<double> A(100, 1.5);   
        nd::buffer_t<double> B(100, 1.5);
        nd::buffer_t<double> C(200, 1.5);
        nd::buffer_t<double> D(100, 2.0);

        REQUIRE(A == A);
        REQUIRE(A == B);
        REQUIRE(A != C);
        REQUIRE(A != D);

        REQUIRE(B == A);
        REQUIRE(B == B);
        REQUIRE(B != C);
        REQUIRE(B != D);

        REQUIRE(C != A);
        REQUIRE(C != B);
        REQUIRE(C == C);
        REQUIRE(C != D);

        REQUIRE(D != A);
        REQUIRE(D != B);
        REQUIRE(D != C);
        REQUIRE(D == D);
    }
}

TEST_CASE("access patterns work OK", "[access_pattern]")
{
    SECTION("can be constructed with factory")
    {
        REQUIRE(nd::make_access_pattern(10, 10, 10).size() == 1000);
        REQUIRE(nd::make_access_pattern(10, 10, 10).with_jumps(2, 2, 2).size() == 125);
    }
    SECTION("can be iterated over")
    {
        auto pat = nd::make_access_pattern(5, 5);
        REQUIRE(nd::distance(pat) == long(pat.size()));
        REQUIRE(pat.contains(0, 0));
        REQUIRE_FALSE(pat.contains(0, 5));
        REQUIRE_FALSE(pat.contains(5, 0));
    }
    SECTION("contains indexes as expected")
    {
        auto pat = nd::make_access_pattern(10).with_start(4).with_jumps(2);
        REQUIRE(pat.contains(0));
        REQUIRE(pat.contains(2));
        REQUIRE_FALSE(pat.contains(3));
    }
    SECTION("generates indexes as expected")
    {
        auto pat = nd::make_access_pattern(10).with_start(4).with_jumps(2);
        REQUIRE(pat.generates(4));
        REQUIRE(pat.generates(6));
        REQUIRE(pat.generates(8));
        REQUIRE_FALSE(pat.generates(0));
        REQUIRE_FALSE(pat.generates(5));
    }
    SECTION("can map and un-map indexes")
    {
        auto pat = nd::make_access_pattern(10).with_start(4).with_jumps(2);
        REQUIRE(pat.inverse_map_index(pat.map_index(nd::make_index(6))) == nd::make_index(6));
    }
}

TEST_CASE("can create strides", "[memory_strides]")
{
    auto strides = nd::make_strides_row_major(nd::make_shape(20, 10, 5));
    REQUIRE(strides[0] == 50);
    REQUIRE(strides[1] == 5);
    REQUIRE(strides[2] == 1);
    REQUIRE(strides.compute_offset(1, 1, 1) == 56);
}

TEST_CASE("array can be constructed with an index provider", "[array] [index_provider]")
{
    auto A = nd::index_array(10);
    REQUIRE(A(5) == nd::make_index(5));
}

TEST_CASE("ones, zeros array factories work as expected", "[ones] [zeros]")
{
    auto A = nd::ones(10, 20);
    auto B = nd::zeros<double>(10, 20);
    static_assert(std::is_same<decltype(A)::value_type, int>::value);
    static_assert(std::is_same<decltype(B)::value_type, double>::value);
    REQUIRE(A(5, 5) == 1);
    REQUIRE(B(5, 5) == 0.0);
}

TEST_CASE("uniform provider can be constructed", "[uniform_provider]")
{
    auto p = nd::make_uniform_provider(1.0, 10, 20, 40);
    auto q = p.reshape(nd::make_shape(5, 2, 10, 2, 20, 2));
    REQUIRE(p(nd::make_index(0, 0, 0)) == 1.0);
    REQUIRE(p(nd::make_index(9, 19, 39)) == 1.0);
    REQUIRE(q(nd::make_index(0, 0, 0, 0, 0, 0)) == 1.0);
    REQUIRE(q(nd::make_index(4, 1, 9, 1, 19, 1)) == 1.0);
    REQUIRE(p.size() == q.size());
}

TEST_CASE("shared buffer provider can be constructed", "[array] [shared_provider] [unique_provider]")
{
    auto provider = nd::make_unique_provider<double>(20, 10, 5);
    auto data = provider.data();

    provider(1, 0, 0) = 1;
    provider(0, 2, 0) = 2;
    provider(0, 0, 3) = 3;

    REQUIRE(provider(1, 0, 0) == 1);
    REQUIRE(provider(0, 2, 0) == 2);
    REQUIRE(provider(0, 0, 3) == 3);

    SECTION("can move the provider into a mutable array and get the same data")
    {
        auto A = nd::make_array(std::move(provider));
        A(1, 2, 3) = 123;
        REQUIRE(provider.data() == nullptr);
        REQUIRE(A(1, 0, 0) == 1);
        REQUIRE(A(0, 2, 0) == 2);
        REQUIRE(A(0, 0, 3) == 3);
        REQUIRE(A(1, 2, 3) == 123);
        REQUIRE(A.data() == data);

        static_assert(std::is_same<decltype(A)::provider_type, nd::unique_provider_t<3, double>>::value);
    }
    SECTION("can copy a mutable version of the provider into an array and get different data")
    {
        auto A = nd::make_array(provider.shared());
        REQUIRE(provider.data() != nullptr);
        REQUIRE(A(1, 0, 0) == 1);
        REQUIRE(A(0, 2, 0) == 2);
        REQUIRE(A(0, 0, 3) == 3);
        REQUIRE(A.get_provider().data() != data);
    }
    SECTION("can move a mutable version of the provider into an array and get the same data")
    {
        auto A = nd::make_array(std::move(provider).shared());
        REQUIRE(provider.data() == nullptr);
        REQUIRE(A(1, 0, 0) == 1);
        REQUIRE(A(0, 2, 0) == 2);
        REQUIRE(A(0, 0, 3) == 3);
        REQUIRE(A.get_provider().data() == data);
    }
    SECTION("can create a transient array from an immutable one")
    {
        auto A = nd::make_array(std::move(provider).shared());
        auto a = A;
        auto B = A.unique();
        // auto b = B; // cannot copy-construct B
        auto C = B.shared(); // cannot assign to C
        B(1, 2, 3) = 123;
        REQUIRE(A(1, 2, 3) != 123);
        REQUIRE(B(1, 2, 3) == 123);
        REQUIRE(a.get_provider().data() == A.get_provider().data());
        REQUIRE(A.get_provider().data() != B.get_provider().data());
    }
}

TEST_CASE("can zip arrays together", "[zip_arrays]")
{
    auto A = nd::make_shared_array<double>(10, 10);
    auto B = nd::make_shared_array<int>(10, 10);
    auto AB = nd::zip_arrays(A, B);
    REQUIRE(AB(0, 0) == std::make_tuple(0.0, 0));
}

TEST_CASE("bounds checking operator works as expected", "[bounds_check]")
{
    auto A1 = nd::index_array(10, 10);
    auto A2 = nd::index_array(10, 10) | nd::bounds_check();
    REQUIRE_NOTHROW(A1(0, 0));
    REQUIRE_NOTHROW(A2(0, 0));
    REQUIRE_NOTHROW(A1(10, 10));
    REQUIRE_THROWS (A2(10, 10));
}

TEST_CASE("providers can be reshaped", "[unique_provider] [shared_provider] [reshape]")
{
    SECTION("unique")
    {
        auto provider = nd::make_unique_provider<double>(10, 10);
        REQUIRE_NOTHROW(provider.reshape(nd::make_shape(10, 10)));
        REQUIRE_NOTHROW(provider.reshape(nd::make_shape(5, 20)));
        REQUIRE_NOTHROW(provider.reshape(nd::make_shape(5, 5, 4)));
        REQUIRE_THROWS(provider.reshape(nd::make_shape(10, 10, 10)));
    }
    SECTION("shared")
    {
        auto provider = nd::make_shared_provider<double>(10, 10);
        REQUIRE_NOTHROW(provider.reshape(nd::make_shape(10, 10)));
        REQUIRE_NOTHROW(provider.reshape(nd::make_shape(5, 20)));
        REQUIRE_NOTHROW(provider.reshape(nd::make_shape(5, 5, 4)));
        REQUIRE_THROWS(provider.reshape(nd::make_shape(10, 10, 10)));
        REQUIRE(provider.reshape(nd::make_shape(5, 5, 4)).data() == provider.data());
    }
}

TEST_CASE("arrays can be reshaped given a reshapable provider", "[unique_provider] [reshape]")
{
    auto A = nd::make_array(nd::make_unique_provider<double>(10, 10));
    REQUIRE_NOTHROW(A | nd::reshape(2, 50));
    REQUIRE_THROWS(A | nd::reshape(2, 51));
}

TEST_CASE("replace operator works as expected", "[replace]")
{
    SECTION("trying to replace a region with an array of the wrong size throws")
    {
        auto A1 = nd::index_array(10);
        auto A2 = nd::index_array(5);
        auto patch1 = nd::make_access_pattern(10).with_start(5);
        auto patch2 = nd::make_access_pattern(10).with_start(6);
        REQUIRE_NOTHROW(nd::replace(patch1, A2));
        REQUIRE_THROWS(A1 | nd::replace(patch2, A2));
    }
    SECTION("replacing all of an array works")
    {
        auto A1 = nd::make_array(nd::make_uniform_provider(1.0, 10));
        auto A2 = nd::make_array(nd::make_uniform_provider(2.0, 10));
        auto patch = nd::make_access_pattern(10);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.indexes())
        {
            REQUIRE(A3(index) == 2.0);
        }
    }
    SECTION("replacing the first half of an array with constant values works")
    {
        auto A1 = nd::make_array(nd::make_uniform_provider(1.0, 10));
        auto A2 = nd::make_array(nd::make_uniform_provider(2.0, 5));
        auto patch = nd::make_access_pattern(5);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.indexes())
        {
            REQUIRE(A3(index) == (index[0] < 5 ? 2.0 : 1.0));
        }
    }
    SECTION("replacing the second half of an array with constant values works")
    {
        auto A1 = nd::make_array(nd::make_uniform_provider(1.0, 10));
        auto A2 = nd::make_array(nd::make_uniform_provider(2.0, 5));
        auto patch = nd::make_access_pattern(10).with_start(5);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.indexes())
        {
            REQUIRE(A3(index) == (index[0] < 5 ? 1.0 : 2.0));
        }
    }
    SECTION("replacing the second half of an array with linear values works")
    {
        auto A1 = nd::index_array(10);
        auto A2 = nd::index_array(5);
        auto patch = nd::make_access_pattern(10).with_start(5);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.indexes())
        {
            REQUIRE(A3(index)[0] == (index[0] < 5 ? index[0] : index[0] - 5));
        }
    }
    SECTION("replacing every other value works")
    {
        auto A1 = nd::index_array(10);
        auto A2 = nd::index_array(5);
        auto patch = nd::make_access_pattern(10).with_start(0).with_jumps(2);
        auto A3 = A1 | nd::replace(patch, A2);

        for (auto index : A3.indexes())
        {
            REQUIRE(A3(index)[0] == (index[0] % 2 == 0 ? index[0] / 2 : index[0]));
        }
    }
    SECTION("replace_from operator works", "[replace_from]")
    {
        auto A = nd::zeros(10, 10);
        REQUIRE_NOTHROW(A | nd::replace_from(0, 0).to(10, 5).with(nd::ones(10, 5)));
        REQUIRE_THROWS(A | nd::replace_from(0, 0).to(10, 5).with(nd::ones(10, 6)));
    }
}

TEST_CASE("map operator works as expected", "[map]")
{
    SECTION("with index provider")
    {
        auto A1 = nd::index_array(10);
        auto A2 = A1 | nd::map([] (auto i) { return i[0] * 2.0; });

        for (auto index : A2.indexes())
        {
            REQUIRE(A2(index) == index[0] * 2.0);
        }
    }
    SECTION("with shared provider")
    {
        auto B1 = nd::make_shared_array<double>(10);
        auto B2 = B1 | nd::map([] (auto) { return 2.0; });

        for (auto index : B2.indexes())
        {
            REQUIRE(B2(index) == 2.0);
        }
    }
    SECTION("with unique provider")
    {
        auto C1 = nd::make_unique_array<double>(10);
        auto C2 = C1.shared() | nd::map([] (auto) { return 2.0; });

        for (auto index : C2.indexes())
        {
            REQUIRE(C2(index) == 2.0);
        }
    }
}

TEST_CASE("select operator works as expected", "[select]")
{
    SECTION("with index array")
    {
        auto A1 = nd::index_array(10);
        auto A2 = A1 | nd::select(nd::make_access_pattern(5));
        auto A3 = A1 | nd::select(nd::make_access_pattern(10).with_start(5));
        REQUIRE(A2.shape() == nd::make_shape(5));
        REQUIRE(A3.shape() == nd::make_shape(5));
        REQUIRE(A2(0) == nd::make_index(0));
        REQUIRE(A3(0) == nd::make_index(5));
        REQUIRE_NOTHROW(A1 | nd::select(nd::make_access_pattern(10)));
        REQUIRE_THROWS(A1 | nd::select(nd::make_access_pattern(11)));
    }
    SECTION("with shared array")
    {
        auto A1 = nd::make_unique_array<double>(10, 10);
        auto A2 = A1.shared() | nd::select(nd::make_access_pattern(5, 5));
        A1(0, 0) = 1.0;
        REQUIRE(A1(0, 0) == 1.0);
        REQUIRE(A2(0, 0) == 0.0);
        REQUIRE(A2.shape() == nd::make_shape(5, 5));
    }
}

TEST_CASE("select_axis operator works as expected", "[select_axis]")
{
    auto A = nd::index_array(10, 10);

    REQUIRE((A | nd::select_axis(0).from(2).to(8)).shape() == nd::make_shape(6, 10));
    REQUIRE((A | nd::select_axis(1).from(2).to(8)).shape() == nd::make_shape(10, 6));
    REQUIRE((A | nd::select_axis(0).from(2).to(2).from_the_end()).shape() == nd::make_shape(6, 10));
    REQUIRE((A | nd::select_axis(1).from(2).to(2).from_the_end()).shape() == nd::make_shape(10, 6));

    REQUIRE((A | nd::select_axis(0).from(2).to(2).from_the_end() | nd::read_index(0, 0)) == nd::make_index(2, 0));
    REQUIRE((A | nd::select_axis(1).from(2).to(2).from_the_end() | nd::read_index(0, 0)) == nd::make_index(0, 2));
}

TEST_CASE("freeze_axis operator works as expected", "[freeze_axis]")
{
    auto A = nd::index_array(10, 10);
    REQUIRE((A | nd::freeze_axis(0).at_index(5)).shape() == nd::make_shape(10));
    REQUIRE((A | nd::freeze_axis(0).at_index(5))(0) == nd::make_index(5, 0));
    REQUIRE((A | nd::freeze_axis(0).at_index(5))(5) == nd::make_index(5, 5));
    REQUIRE((A | nd::freeze_axis(1).at_index(5))(0) == nd::make_index(0, 5));
    REQUIRE((A | nd::freeze_axis(1).at_index(5))(5) == nd::make_index(5, 5));
}

TEST_CASE("binary operation works as expected")
{
    auto F = nd::binary_op(std::plus<>());
    auto A = nd::ones(10, 10);
    auto B = nd::ones<double>(10, 10);
    auto b = nd::ones<double>(10, 11);
    auto C = F(A, B);
    static_assert(std::is_same<decltype(C(0, 0)), double>::value);

    REQUIRE(C(0, 0) == 2.0);
    REQUIRE_THROWS(F(A, b));
    REQUIRE_THROWS((A + b)(0, 0));

    REQUIRE((A + B)(0, 0) == 2.0);
    REQUIRE((C + 2.0)(0, 0) == 4.0);
    REQUIRE((C - 2.0)(0, 0) == 0.0);
    REQUIRE((C * 2.0)(0, 0) == 4.0);
    REQUIRE((C / 2.0)(0, 0) == 1.0);
}

TEST_CASE("can read an index from an array", "[read_index]")
{
    REQUIRE((nd::ones(10, 20, 40) | nd::read_index(2, 3, 4)) == 1);
}

TEST_CASE("can sum an array", "[sum]")
{
    using namespace nd;
    REQUIRE((index_array(3) | map([] (auto i) { return i[0]; }) | sum()) == 3);
    REQUIRE((ones(10, 10) | sum()) == 100);
}

TEST_CASE("can test for equality", "[sum] [any] [all]")
{
    REQUIRE(((nd::ones(10, 10) == nd::ones(10, 10)) | nd::sum()) == 100);
    REQUIRE(bool((nd::ones(10, 10) == nd::ones(10, 10)) | nd::all()));
    REQUIRE(bool((nd::ones(10, 10) == nd::ones<double>(10, 10)) | nd::all()));
    REQUIRE(bool((nd::ones(10, 10) != nd::zeros<double>(10, 10)) | nd::all()));
    REQUIRE_FALSE(bool((nd::ones(10, 10) == nd::zeros<double>(10, 10)) | nd::any()));
}

TEST_CASE("can get an index array using where, and pass that to read_indexes", "[where] [read_indexes]")
{
    auto A = nd::index_array(10) | nd::map([] (auto i) { return i[0]; });
    REQUIRE(nd::where(A < 5).size() == 5);
    REQUIRE(bool(((A | nd::read_indexes(nd::where(A < 5))) < 5) | nd::all()));
}

TEST_CASE("can get the sum of a 3D array on each axis", "[collect]")
{
    auto A = nd::ones(10, 20, 30);

    REQUIRE((A | nd::collect(nd::sum()).along_axis(0) | nd::read_index(0, 0)) == 10);
    REQUIRE((A | nd::collect(nd::sum()).along_axis(1) | nd::read_index(0, 0)) == 20);
    REQUIRE((A | nd::collect(nd::sum()).along_axis(2) | nd::read_index(0, 0)) == 30);
}

TEST_CASE("can concat two 3d arrays on compatible axes", "[collect]")
{
    using namespace nd;
    REQUIRE((ones(10, 10, 20) | concat(zeros(10, 10, 30)).on_axis(2) | read_index(0, 0, 19)) == 1);
    REQUIRE((ones(10, 10, 20) | concat(zeros(10, 10, 30)).on_axis(2) | read_index(0, 0, 20)) == 0);
    REQUIRE_THROWS(ones(10, 10, 20) | concat(zeros(10, 11, 30)).on_axis(2));
}

TEST_CASE("can create the cartesian product of arrays", "[cartesian_product]")
{
    auto A = nd::cartesian_product(nd::ones(10), nd::zeros(20));
    REQUIRE(A.shape() == nd::make_shape(10, 20));
    REQUIRE(A(0, 0) == std::make_tuple(1, 0));
}

TEST_CASE("can shift an array", "[shift]")
{
    auto A = nd::index_array(10, 10);
    REQUIRE((A | nd::shift_by(2).along_axis(0)).shape() == nd::make_shape(8, 10));
    REQUIRE((A | nd::shift_by(2).along_axis(1)).shape() == nd::make_shape(10, 8));
    REQUIRE((A | nd::shift_by(-2).along_axis(0)).shape() == nd::make_shape(8, 10));
    REQUIRE((A | nd::shift_by(-2).along_axis(1)).shape() == nd::make_shape(10, 8));
    REQUIRE((A | nd::shift_by(-2).along_axis(0) | nd::read_index(0, 0)) == nd::make_index(2, 0));
    REQUIRE((A | nd::shift_by(-2).along_axis(1) | nd::read_index(0, 0)) == nd::make_index(0, 2));
    REQUIRE((A | nd::shift_by(+2).along_axis(0) | nd::read_index(2, 0)) == nd::make_index(0, 0));
    REQUIRE((A | nd::shift_by(+2).along_axis(1) | nd::read_index(0, 2)) == nd::make_index(0, 0));
}
