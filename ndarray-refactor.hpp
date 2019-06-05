/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
*/




#pragma once
#include <algorithm>         // std::all_of
#include <functional>        // std::ref
#include <initializer_list>  // std::initializer_list
#include <iterator>          // std::distance
#include <memory>            // std::shared_ptr
#include <numeric>           // std::accumulate
#include <string>            // std::to_string
#include <utility>           // std::index_sequence
#include "sequence.hpp"




//=============================================================================
namespace nd2
{
    template<std::size_t Rank> struct shape_t;
    template<std::size_t Rank> struct index_t;
    template<std::size_t Rank> struct jumps_t;
    template<std::size_t Rank> struct memory_strides_t;
    template<std::size_t Rank> struct access_pattern_t;

    template<typename... Args> auto make_shape(Args... args);
    template<typename... Args> auto make_index(Args... args);
    template<typename... Args> auto make_jumps(Args... args);
    template<typename... Args> auto make_access_pattern(Args... args);
    template<std::size_t Rank> auto make_access_pattern(shape_t<Rank> shape);

    template<std::size_t Rank> auto uniform_shape(std::size_t value);
    template<std::size_t Rank> auto uniform_index(std::size_t value);
    template<std::size_t Rank> auto uniform_jumps(std::size_t value);

    template<std::size_t Rank> auto make_strides_row_major(const shape_t<Rank>& shape);
}




//=============================================================================
namespace nd2::detail
{
    template<typename Fn>
    auto apply_to(Fn fn) { return [fn] (auto t) { return std::apply(fn, t); }; }
}




//=============================================================================
template<std::size_t Rank>
struct nd2::shape_t
{
    shape_t() {}
    shape_t(sq::sequence_t<std::size_t, Rank> seq) : seq(seq) {}

    bool operator==(const shape_t& other) const { return seq == other.seq; }
    bool operator!=(const shape_t& other) const { return seq != other.seq; }
    const std::size_t& operator[](std::size_t i) const { return seq[i]; }

    std::size_t volume() const { return sq::product(seq); }
    index_t<Rank> last_index() const { return seq; }

    bool contains(const index_t<Rank>& index) const { return sq::all_of(sq::zip(index.seq, seq), detail::apply_to(std::less<>())); }
    template<typename... Args> bool contains(Args... args) const { return contains(make_index(args...)); }

    template<std::size_t N> auto select(const sq::sequence_t<std::size_t, N>& indexes) const { return shape_t<N>(sq::read_indexes(seq, indexes)); }
    template<typename... Args> auto select(Args... args) const { return select(sq::make_sequence(std::size_t(args)...)); }

    template<std::size_t N> auto remove(const sq::sequence_t<std::size_t, N>& is) const { return shape_t<Rank - N>(sq::remove_indexes(seq, is)); }
    template<typename... Args> auto remove(Args... args) const { return remove(sq::make_sequence(std::size_t(args)...)); }

    template<std::size_t NumElements>
    auto insert(
        const sq::sequence_t<std::size_t, NumElements>& elements,
        const sq::sequence_t<std::size_t, NumElements>& indexes) const
    {
        return shape_t<Rank + NumElements>(sq::insert_elements(seq, elements, indexes));
    }

    sq::sequence_t<std::size_t, Rank> seq;
};




//=============================================================================
template<std::size_t Rank>
struct nd2::index_t
{
    index_t() {}
    index_t(sq::sequence_t<std::size_t, Rank> seq) : seq(seq) {}

    bool operator==(const index_t& other) const { return seq == other.seq; }
    bool operator!=(const index_t& other) const { return seq != other.seq; }
    const std::size_t& operator[](std::size_t i) const { return seq[i]; }

    sq::sequence_t<std::size_t, Rank> seq;
};




//=============================================================================
template<std::size_t Rank>
struct nd2::jumps_t
{
    jumps_t() {}
    jumps_t(sq::sequence_t<std::size_t, Rank> seq) : seq(seq) {}

    bool operator==(const jumps_t& other) const { return seq == other.seq; }
    bool operator!=(const jumps_t& other) const { return seq != other.seq; }
    const std::size_t& operator[](std::size_t i) const { return seq[i]; }

    sq::sequence_t<std::size_t, Rank> seq;
};




//=============================================================================
template<std::size_t Rank>
struct nd2::memory_strides_t
{
public:
    memory_strides_t() {}
    memory_strides_t(sq::sequence_t<std::size_t, Rank> seq) : seq(seq) {}

    bool operator==(const memory_strides_t& other) const { return seq == other.seq; }
    bool operator!=(const memory_strides_t& other) const { return seq != other.seq; }
    std::size_t& operator[](std::size_t i) { return seq[i]; }
    const std::size_t& operator[](std::size_t i) const { return seq[i]; }

    std::size_t compute_offset(const index_t<Rank>& index) const { return sq::zip(index.seq, seq) | sq::apply(std::multiplies<>()) | sq::sum(); }
    template<typename... Args> std::size_t compute_offset(Args... args) const { return compute_offset(make_index(args...)); }

    sq::sequence_t<std::size_t, Rank> seq;
};




//=============================================================================
template<std::size_t Rank>
struct nd2::access_pattern_t
{
    // using value_type = index_t<Rank>;
    // static constexpr std::size_t rank = Rank;



    //=========================================================================
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = index_t<Rank>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++() { accessor.advance(current); return *this; }
        bool operator==(const iterator& other) const { return current == other.current; }
        bool operator!=(const iterator& other) const { return current != other.current; }
        const index_t<Rank>& operator*() const { return current; }

        access_pattern_t accessor;
        index_t<Rank> current;
    };




    //=========================================================================
    bool operator==(const access_pattern_t& other) const
    {
        return
        start == other.start &&
        final == other.final &&
        jumps == other.jumps;
    }

    bool operator!=(const access_pattern_t& other) const
    {
        return
        start != other.start ||
        final != other.final ||
        jumps != other.jumps;
    }




    //=========================================================================
    template<typename... Args> access_pattern_t with_start(Args... args) const { return { make_index(args...), final, jumps }; }
    template<typename... Args> access_pattern_t with_final(Args... args) const { return { start, make_index(args...), jumps }; }
    template<typename... Args> access_pattern_t with_jumps(Args... args) const { return { start, final, make_jumps(args...) }; }

    access_pattern_t with_start(index_t<Rank> arg) const { return { arg, final, jumps }; }
    access_pattern_t with_final(index_t<Rank> arg) const { return { start, arg, jumps }; }
    access_pattern_t with_jumps(jumps_t<Rank> arg) const { return { start, final, arg }; }




    /**
     * @brief      Return the number of elements hit by an iteration over this
     *             access pattern.
     *
     * @return     The number of elements
     */
    std::size_t size() const
    {
        return shape().volume();
    }




    /**
     * @brief      Return the shape of the mapped-to index space.
     *
     * @return     A shape
     */
    auto shape() const
    {
        auto s = shape_t<Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            s[n] = final[n] / jumps[n] - start[n] / jumps[n];
        }
        return s;
    }




    /**
     * @brief      Determine whether this accessor has any elements.
     *
     * @return     A boolean
     */
    bool empty() const
    {
        return size() == 0;
    }




    /**
     * @brief      Move an index forward, and return true if the new index is
     *             before the end.
     *
     * @param      index  The index to advance
     *
     * @return     A boolean
     * @note       The last index advances fastest.
     */
    bool advance(index_t<Rank>& index) const
    {
        int n = Rank - 1;

        index[n] += jumps[n];

        while (index[n] >= final[n])
        {
            if (n == 0)
            {
                index = final;
                return false;
            }
            index[n] = start[n];

            --n;

            index[n] += jumps[n];
        }
        return true;
    }




    /**
     * @brief      Map the given index through this accessor.
     *
     * @param[in]  index  The index to map
     *
     * @return     The mapped index
     */
    index_t<Rank> map_index(const index_t<Rank>& index) const
    {
        auto result = index_t<Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = start[n] + jumps[n] * index[n];

        return result;
    }




    /**
     * @brief      Return the index that generates the given mapped index.
     *
     * @param[in]  mapped_index  The mapped index
     *
     * @return     A boolean
     */
    index_t<Rank> inverse_map_index(const index_t<Rank>& mapped_index) const
    {
        auto result = index_t<Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = (mapped_index[n] - start[n]) / jumps[n];

        return result;
    }




    /**
     * @brief      Return true if this is a valid mapped-from index.
     *
     * @param[in]  index  The index possibly mapped from by this access pattern
     *
     * @return     A boolean
     */
    bool contains(const index_t<Rank>& index) const
    {
        return shape().contains(index);
    }
    template<typename... Args> bool contains(Args... args) const { return contains(make_index(args...)); }




    /**
     * @brief      Return true if an iteration over this accessor would generate
     *             the given index, that is, if the index is included in the set
     *             of mapped-to indexes.
     *
     * @param[in]  mapped_index  The index possibly mapped to by this access
     *                           pattern
     *
     * @return     A boolean
     */
    bool generates(const index_t<Rank>& mapped_index) const
    {
        for (std::size_t n = 0; n < Rank; ++n)
        {
            if ((mapped_index[n] <  start[n]) ||
                (mapped_index[n] >= final[n]) ||
                (mapped_index[n] -  start[n]) % jumps[n] != 0)
            {
                return false;
            }
        }
        return true;
    }
    template<typename... Args>
    bool generates(Args... args) const { return generates(make_index(args...)); }




    /**
     * @brief      Return false if this access pattern would generate any
     *             indexes not contained in the given shape.
     *
     * @param[in]  parent_shape  The shape possibly containing an index in this
     *                           access pattern
     *
     * @return     A boolean
     */
    bool within(const shape_t<Rank>& parent_shape) const
    {
        auto zero = uniform_index<Rank>(0);
        auto t1 = map_index(zero);
        auto t2 = map_index(shape().last_index());

        return (t1 >= zero && t1 <= parent_shape.last_index() &&
                t2 >= zero && t2 <= parent_shape.last_index());
    }




    //=========================================================================
    iterator begin() const { return { *this, start }; }
    iterator end() const { return { *this, final }; }




    //=========================================================================
    index_t<Rank> start = uniform_index<Rank>(0);
    index_t<Rank> final = uniform_index<Rank>(0);
    jumps_t<Rank> jumps = uniform_jumps<Rank>(1);
};




//=============================================================================
template<std::size_t Rank>
auto nd2::make_strides_row_major(const shape_t<Rank>& shape)
{
    auto result = memory_strides_t<Rank>();

    if constexpr (Rank > 0)
    {
        result[Rank - 1] = 1;
    }
    if constexpr (Rank > 1)
    {
        for (int n = Rank - 2; n >= 0; --n)
        {
            result[n] = result[n + 1] * shape[n + 1];
        }
    }
    return result;
}




//=============================================================================
template<typename... Args>
auto nd2::make_shape(Args... args)
{
    return shape_t<sizeof...(Args)>({std::size_t(args)...});
}

template<typename... Args>
auto nd2::make_index(Args... args)
{
    return index_t<sizeof...(Args)>({std::size_t(args)...});
}

template<typename... Args>
auto nd2::make_jumps(Args... args)
{
    return jumps_t<sizeof...(Args)>({std::size_t(args)...});
}

template<typename... Args>
auto nd2::make_access_pattern(Args... args)
{
    return access_pattern_t<sizeof...(Args)>().with_final(args...);
}

template<std::size_t Rank>
auto nd2::make_access_pattern(shape_t<Rank> shape)
{
    return access_pattern_t<Rank>().with_final(shape.last_index());
}




//=============================================================================
template<std::size_t Rank>
auto nd2::uniform_shape(std::size_t value)
{
    return nd2::shape_t{sq::uniform_sequence<Rank>(value)};
}

template<std::size_t Rank>
auto nd2::uniform_index(std::size_t value)
{
    return nd2::index_t{sq::uniform_sequence<Rank>(value)};
}

template<std::size_t Rank>
auto nd2::uniform_jumps(std::size_t value)
{
    return nd2::jumps_t{sq::uniform_sequence<Rank>(value)};
}

