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


    template<typename... Args> auto make_shape(Args... args);
    template<typename... Args> auto make_index(Args... args);
    template<typename... Args> auto make_jumps(Args... args);

    template<std::size_t Rank> auto uniform_shape(std::size_t value);
    template<std::size_t Rank> auto uniform_index(std::size_t value);
    template<std::size_t Rank> auto uniform_jumps(std::size_t value);
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
