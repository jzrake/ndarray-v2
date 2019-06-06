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
#include <functional> // std::plus, std::multiplies
#include <stdexcept>  // std::out_of_range
#include <tuple>      // std::tuple
#include <utility>    // std::make_index_sequence




//=============================================================================
namespace sq
{
    template<typename ValueType, std::size_t Rank>
    struct sequence_t;

    template<typename... Args, typename ValueType=std::common_type_t<Args...>>
    auto make_sequence(Args... args);

    template<std::size_t Rank, typename ValueType>
    auto uniform_sequence(ValueType uniform_value);

    template<std::size_t Rank>
    auto range_sequence();

    template<std::size_t Index, typename ValueType, std::size_t Rank>
    auto get(const sequence_t<ValueType, Rank>& seq);

    template<std::size_t Index, typename... ValueTypes, std::size_t Rank>
    auto get_from_each(sequence_t<ValueTypes, Rank>... seqs);

    template<typename... ValueTypes, std::size_t Rank>
    auto zip(sequence_t<ValueTypes, Rank>... seqs);

    template<typename ValueType, std::size_t Rank>
    auto head(const sequence_t<ValueType, Rank>& seq);
    inline auto head();

    template<typename ValueType, std::size_t Rank>
    auto last(const sequence_t<ValueType, Rank>& seq);
    inline auto last();

    template<typename ValueType, std::size_t Rank>
    auto init(const sequence_t<ValueType, Rank>& seq);
    inline auto init();

    template<typename ValueType, std::size_t Rank>
    auto tail(const sequence_t<ValueType, Rank>& seq);
    inline auto tail();

    template<std::size_t Index, typename ValueType, std::size_t Rank>
    auto partition(const sequence_t<ValueType, Rank>& seq);
    template<std::size_t Index>
    auto partition();

    template<typename ValueType, std::size_t RankA, std::size_t RankB>
    auto concat(const sequence_t<ValueType, RankA>& a, const sequence_t<ValueType, RankB>& b);
    template<typename ValueType, std::size_t RankB>
    auto concat(const sequence_t<ValueType, RankB>& b);

    template<typename ValueType, std::size_t Rank>
    auto erase(const sequence_t<ValueType, Rank>& seq, std::size_t index);
    inline auto erase(std::size_t index);

    template<typename ValueType, std::size_t Rank>
    auto insert(const sequence_t<ValueType, Rank>& seq, std::size_t index, ValueType value_to_insert);
    template<typename ValueType>
    auto insert(std::size_t index, ValueType v);

    template<typename ValueType, std::size_t Rank>
    auto append(const sequence_t<ValueType, Rank>& seq, ValueType value_to_append);
    template<typename ValueType>
    auto append(ValueType v);

    template<typename ValueType, std::size_t Rank>
    auto prepend(const sequence_t<ValueType, Rank>& seq, ValueType value_to_prepend);
    template<typename ValueType>
    auto prepend(ValueType v);

    template<typename Fn, typename ValueType, std::size_t Rank>
    auto map(Fn fn, const sequence_t<ValueType, Rank>& seq);
    template<typename Fn>
    auto map(Fn fn);

    template<typename Fn, typename ValueType, std::size_t Rank>
    auto apply(Fn fn, const sequence_t<ValueType, Rank>& seq);
    template<typename Fn>
    auto apply(Fn fn);

    template<typename Fn, typename ValueType, std::size_t Rank>
    auto reduce(Fn fn, const sequence_t<ValueType, Rank>& seq, ValueType seed);
    template<typename Fn, typename ValueType>
    auto reduce(Fn fn, ValueType seed);

    template<typename ValueType, std::size_t Rank>
    auto sum(const sequence_t<ValueType, Rank>& seq);
    inline auto sum();

    template<typename ValueType, std::size_t Rank>
    auto product(const sequence_t<ValueType, Rank>& seq);
    inline auto product();

    template<typename ValueType, typename Predicate, std::size_t Rank>
    auto all_of(const sequence_t<ValueType, Rank>& seq, Predicate pred);

    template<typename ValueType, typename Predicate, std::size_t Rank>
    auto any_of(const sequence_t<ValueType, Rank>& seq, Predicate pred);

    template<typename ValueType, std::size_t Rank>
    auto contains(const sequence_t<ValueType, Rank>& seq, ValueType value);
    template<typename ValueType>
    auto contains(ValueType value);

    template<typename ValueType, std::size_t Rank, std::size_t NumIndexes>
    auto read_indexes(const sequence_t<ValueType, Rank>& seq, const sequence_t<std::size_t, NumIndexes>& indexes);
    template<typename... Args>
    auto read_indexes(Args... indexes);

    template<typename ValueType, std::size_t Rank, std::size_t NumToInsert>
    auto insert_elements(
        const sequence_t<ValueType, Rank>& source,
        const sequence_t<ValueType, NumToInsert>& values,
        const sequence_t<std::size_t, NumToInsert>& indexes);

    template<typename ValueType, std::size_t Rank, std::size_t NumToRemove>
    auto remove_indexes(
        const sequence_t<ValueType, Rank>& source,
        const sequence_t<std::size_t, NumToRemove>& indexes);
}




//=============================================================================
namespace sq::detail
{
    template<typename Fn, std::size_t... Is>
    constexpr auto index_apply_impl(Fn&& fn, std::index_sequence<Is...>)
    {
        return fn(std::integral_constant<std::size_t, Is>{}...);
    }

    template<std::size_t NumberOfItems, typename Fn>
    constexpr auto index_apply(Fn&& fn)
    {
        return index_apply_impl(std::forward<Fn>(fn), std::make_index_sequence<NumberOfItems>{});
    }
}




/**
 * @brief      A statically sized, uniformly types sequence
 *
 * @tparam     ValueType  The type of the elements
 * @tparam     Rank       The number of elements
 */
template<typename ValueType, std::size_t Rank>
struct sq::sequence_t
{
    using value_type = ValueType;

    value_type* begin() { return &__data[0]; }
    value_type* end()   { return &__data[0] + Rank; }
    value_type* data()  { return &__data[0]; }
    value_type& operator[](std::size_t index) { return __data[index]; }
    value_type& at(std::size_t index)
    {
        if (index >= Rank)
            throw std::out_of_range("sequence_t::at");
        return __data[index];
    }

    const value_type* begin() const { return &__data[0]; }
    const value_type* end()   const { return &__data[0] + Rank; }
    const value_type* data()  const { return &__data[0]; }
    const value_type& operator[](std::size_t index) const { return __data[index]; }
    const value_type& at(std::size_t index) const
    {
        if (index >= Rank)
            throw std::out_of_range("sequence_t::at");
        return __data[index];
    }

    template<typename Fn>
    decltype(auto) operator|(Fn&& fn) const
    {
        return fn(*this);
    }

    bool operator==(const sequence_t& other) const { for (std::size_t i = 0; i < Rank; ++i) if (__data[i] != other[i]) return false; return true; }
    bool operator!=(const sequence_t& other) const { for (std::size_t i = 0; i < Rank; ++i) if (__data[i] != other[i]) return true; return false; }
    constexpr std::size_t size() const { return Rank; }

    ValueType __data[Rank];
};




/**
 * @brief      Make a new sequence with inferred type and size.
 *
 * @param[in]  args       The elements
 *
 * @tparam     Args       The element types
 * @tparam     ValueType  The inferred type, if a common type can be inferred
 *
 * @return     The sequence
 * @note       You can override the inferred value type by doing e.g.
 *             make_sequence<size_t>(1, 2).
 */
template<typename... Args, typename ValueType>
auto sq::make_sequence(Args... args)
{
    return sequence_t<ValueType, sizeof...(Args)> {ValueType(args)...};
}




/**
 * @brief      Return a sequence with all elements equal to the given value.
 *
 * @param[in]  uniform_value  The value
 *
 * @tparam     Rank           The sequence's rank
 * @tparam     ValueType      The sequence's value type
 *
 * @return     The sequence
 */
template<std::size_t Rank, typename ValueType>
auto sq::uniform_sequence(ValueType uniform_value)
{
    auto result = sequence_t<ValueType, Rank>();

    for (std::size_t i = 0; i < Rank; ++i)
        result[i] = uniform_value;

    return result;
}




/**
 * @brief      Return a sequence of increasing size_t values [0 .. Rank - 1].
 *
 * @tparam     Rank  The sequence's rank
 *
 * @return     The sequence
 */
template<std::size_t Rank>
auto sq::range_sequence()
{
    auto result = sequence_t<std::size_t, Rank>();

    for (std::size_t i = 0; i < Rank; ++i)
        result[i] = i;

    return result;
}




/**
 * @brief      Return the element at the template-parameter index, same as
 *             std::get on an std::array.
 *
 * @param[in]  seq        The sequence
 *
 * @tparam     Index      The index
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     The element
 */
template<std::size_t Index, typename ValueType, std::size_t Rank>
auto sq::get(const sequence_t<ValueType, Rank>& seq)
{
    static_assert(Index < Rank, "cannot get element at index greater than or equal to rank");
    return seq[Index];
}




/**
 * @brief      Return a tuple made from the i-th index of each of the given
 *             sequences.
 *
 * @param[in]  seqs        The sequences
 *
 * @tparam     Index       The index of each sequence to be put in the tuple
 * @tparam     ValueTypes  The value types of each sequences
 * @tparam     Rank        The common rank of all the sequences
 *
 * @return     An instance of std::tuple<ValueTupes...>
 */
template<std::size_t Index, typename... ValueTypes, std::size_t Rank>
auto sq::get_from_each(sequence_t<ValueTypes, Rank>... seqs)
{
    return sq::detail::index_apply<sizeof...(seqs)>([t = std::make_tuple(seqs...)] (auto... Is)
    {
        return std::make_tuple(get<Index>(std::get<Is>(t))...);
    });
}




/**
 * @brief      Turn the given sequences into a sequence of tuples.
 *
 * @param[in]  seqs        The sequences
 *
 * @tparam     ValueTypes  The value types of each sequence
 * @tparam     Rank        The rank common to all sequences
 *
 * @return     A new sequence
 */
template<typename... ValueTypes, std::size_t Rank>
auto sq::zip(sequence_t<ValueTypes, Rank>... seqs)
{
    return sq::detail::index_apply<Rank>([&] (auto... Is)
    {
        return make_sequence(get_from_each<Is>(seqs...)...);
    });
}




/**
 * @brief      Return the first element of a sequence.
 *
 * @param[in]  seq        The sequence
 *
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     The first element
 */
template<typename ValueType, std::size_t Rank>
auto sq::head(const sequence_t<ValueType, Rank>& seq)
{
    static_assert(Rank > 0, "cannot take the head of a zero-rank sequence");
    return seq[0];
}
auto sq::head() { return [] (auto&& seq) { return head(seq); }; }




/**
 * @brief      Return the last element of a sequence.
 *
 * @param[in]  seq        The sequence
 *
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     The last element
 */
template<typename ValueType, std::size_t Rank>
auto sq::last(const sequence_t<ValueType, Rank>& seq)
{
    static_assert(Rank > 0, "cannot take the last of a zero-rank sequence");
    return seq[Rank - 1];
}
auto sq::last() { return [] (auto&& seq) { return last(seq); }; }




/**
 * @brief      Return all but the last element of a sequence
 *
 * @param[in]  seq        The sequence
 *
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     A sequence containing all but the last part
 */
template<typename ValueType, std::size_t Rank>
auto sq::init(const sequence_t<ValueType, Rank>& seq)
{
    static_assert(Rank > 0, "cannot take the init of a zero-rank sequence");
    auto result = sequence_t<ValueType, Rank - 1>();

    for (std::size_t i = 0; i < Rank - 1; ++i)
        result[i] = seq[i];

    return result;
}
auto sq::init() { return [] (auto&& seq) { return init(seq); }; }




/**
 * @brief      Return all but the first element of a sequence
 *
 * @param[in]  seq        The sequence
 *
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     A sequence containing all but the head
 */
template<typename ValueType, std::size_t Rank>
auto sq::tail(const sequence_t<ValueType, Rank>& seq)
{
    static_assert(Rank > 0, "cannot take the tail of a zero-rank sequence");
    auto result = sequence_t<ValueType, Rank - 1>();

    for (std::size_t i = 0; i < Rank - 1; ++i)
        result[i] = seq[i + 1];

    return result;
}
auto sq::tail() { return [] (auto&& seq) { return tail(seq); }; }




/**
 * @brief      Split this sequence into a pair of sequences
 *
 * @param[in]  seq        The sequence to split
 *
 * @tparam     Index      The index at which to split the sequence
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     A pair of sequences
 */
template<std::size_t Index, typename ValueType, std::size_t Rank>
auto sq::partition(const sequence_t<ValueType, Rank>& seq)
{
    static_assert(Index <= Rank, "cannot partition sequence at index larger than rank");
    auto f = sequence_t<ValueType, Index>();
    auto s = sequence_t<ValueType, Rank - Index>();

    for (std::size_t i = 0; i < Index; ++i)
        f[i] = seq[i];

    for (std::size_t i = 0; i < Rank - Index; ++i)
        s[i] = seq[i + Index];

    return std::make_pair(f, s);
}
template<std::size_t Index>
auto sq::partition() { return [] (auto&& seq) { return partition<Index>(seq); }; }




/**
 * @brief      Concatenate two sequences of the same type but possibly different
 *             rank
 *
 * @param[in]  a          The first sequence
 * @param[in]  b          The second sequence
 *
 * @tparam     ValueType  The value type of both sequences
 * @tparam     RankA      The rank of the first sequence
 * @tparam     RankB      The rank of the second sequence
 *
 * @return     The combined sequence
 */
template<typename ValueType, std::size_t RankA, std::size_t RankB>
auto sq::concat(const sequence_t<ValueType, RankA>& a, const sequence_t<ValueType, RankB>& b)
{
    auto result = sequence_t<ValueType, RankA + RankB>();

    for (std::size_t i = 0; i < RankA; ++i)
        result[i] = a[i];

    for (std::size_t i = 0; i < RankB; ++i)
        result[i + RankA] = b[i];

    return result;
}
template<typename ValueType, std::size_t RankB>
auto sq::concat(const sequence_t<ValueType, RankB>& b) { return [b] (auto&& a) { return concat(a, b); }; }




/**
 * @brief      Return a sequence with the element at the given index removed
 *
 * @param[in]  seq        The sequence to remove an element from
 * @param[in]  index      The index of the element to remove
 *
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     The new sequence
 */
template<typename ValueType, std::size_t Rank>
auto sq::erase(const sequence_t<ValueType, Rank>& seq, std::size_t index)
{
    static_assert(Rank > 0, "cannot erase an element from a zero-rank sequence");

    if (index >= Rank)
    {
        throw std::out_of_range("sequence_t::erase");
    }
    auto result = sequence_t<ValueType, Rank - 1>();

    for (std::size_t i = 0; i < index; ++i)
        result[i] = seq[i];

    for (std::size_t i = index + 1; i < Rank; ++i)
        result[i - 1] = seq[i];

    return result;
}
auto sq::erase(std::size_t index) { return [index] (auto&& seq) { return erase(seq, index); }; }




/**
 * @brief      Return a sequence with an element inserted at the given index
 *
 * @param[in]  seq              The sequence to remove an element from
 * @param[in]  index            The index of the element to remove
 * @param[in]  value_to_insert  The value to insert
 *
 * @tparam     ValueType        The sequence's value type
 * @tparam     Rank             The sequence's rank
 *
 * @return     The new sequence
 */
template<typename ValueType, std::size_t Rank>
auto sq::insert(const sequence_t<ValueType, Rank>& seq, std::size_t index, ValueType value_to_insert)
{
    if (index > Rank)
    {
        throw std::out_of_range("sequence_t::insert");
    }
    auto result = sequence_t<ValueType, Rank + 1>();

    result[index] = value_to_insert;

    for (std::size_t i = 0; i < index; ++i)
        result[i] = seq[i];

    for (std::size_t i = index + 1; i < Rank + 1; ++i)
        result[i] = seq[i - 1];

    return result;
}
template<typename ValueType>
auto sq::insert(std::size_t index, ValueType v) { return [index, v] (auto&& seq) { return insert(seq, index, v); }; }




/**
 * @brief      Append an element to the end of a sequence
 *
 * @param[in]  seq              The sequence
 * @param[in]  value_to_append  The value to append
 *
 * @tparam     ValueType        The sequence's value type
 * @tparam     Rank             The sequence's rank
 *
 * @return     The new sequence
 */
template<typename ValueType, std::size_t Rank>
auto sq::append(const sequence_t<ValueType, Rank>& seq, ValueType value_to_append)
{
    return insert(seq, Rank, value_to_append);
}
template<typename ValueType>
auto sq::append(ValueType v) { return [v] (auto&& seq) { return append(seq, v); }; }





/**
 * @brief      Prepend an element to the beginning of a sequence
 *
 * @param[in]  seq              The sequence
 * @param[in]  value_to_prepend  The value to prepend
 *
 * @tparam     ValueType        The sequence's value type
 * @tparam     Rank             The sequence's rank
 *
 * @return     The new sequence
 */
template<typename ValueType, std::size_t Rank>
auto sq::prepend(const sequence_t<ValueType, Rank>& seq, ValueType value_to_prepend)
{
    return insert(seq, 0, value_to_prepend);
}
template<typename ValueType>
auto sq::prepend(ValueType v) { return [v] (auto&& seq) { return prepend(seq, v); }; }




/**
 * @brief      Transform the elements of a tuple, returning a new one of the
 *             same rank but of possibly different value type.
 *
 * @param[in]  fn         The function to map the elements through
 * @param[in]  seq        The sequence to map
 *
 * @tparam     Fn         The function type
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     A new sequence
 * @note       sequence_t is a functor with respect to this function.
 */
template<typename Fn, typename ValueType, std::size_t Rank>
auto sq::map(Fn fn, const sequence_t<ValueType, Rank>& seq)
{
    auto result = sequence_t<std::invoke_result_t<Fn, ValueType>, Rank>();

    for (std::size_t i = 0; i < Rank; ++i)
        result[i] = fn(seq[i]);

    return result;
}
template<typename Fn>
auto sq::map(Fn fn) { return [fn] (auto&& seq) { return map(fn, seq); }; }




/**
 * @brief      Map function of one or more arguments over a sequence whose value
 *             type is a std::tuple.
 *
 * @param[in]  fn         The function to apply to the sequence elements
 * @param[in]  seq        The sequence
 *
 * @tparam     Fn         The function type
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     A new sequence
 * @note       A tuple of sequence_t's is a functor with respect to this
 *             function, via zip.
 */
template<typename Fn, typename ValueType, std::size_t Rank>
auto sq::apply(Fn fn, const sequence_t<ValueType, Rank>& seq)
{
    return map([fn] (auto a) { return std::apply(fn, a); }, seq);
};
template<typename Fn>
auto sq::apply(Fn fn) { return [fn] (auto&& seq) { return apply(fn, seq); }; }




/**
 * @brief      Return a reduction (e.g. sum, product) of the given sequence
 *
 * @param[in]  fn         A binary function (result, element) -> result
 * @param[in]  seq        The sequence to reduce
 * @param[in]  seed       The seed
 *
 * @tparam     Fn         The function type
 * @tparam     ValueType  The sequence's value type
 * @tparam     Rank       The sequence's rank
 *
 * @return     A new sequence
 */
template<typename Fn, typename ValueType, std::size_t Rank>
auto sq::reduce(Fn fn, const sequence_t<ValueType, Rank>& seq, ValueType seed)
{
    for (std::size_t i = 0; i < Rank; ++i)
    {
        seed = fn(seed, seq[i]);
    }
    return seed;
};
template<typename Fn, typename ValueType>
auto sq::reduce(Fn fn, ValueType seed) { return [fn, seed] (auto&& seq) { return reduce(fn, seq, seed); }; }




/**
 * @brief      Specializes reduce with std::plus.
 *
 * @param[in]  seq        The sequence to sum
 *
 * @tparam     ValueType  The sequence value type
 * @tparam     Rank       The sequence rank
 *
 * @return     The sum of the elements
 */
template<typename ValueType, std::size_t Rank>
auto sq::sum(const sequence_t<ValueType, Rank>& seq)
{
    return reduce(std::plus<>(), seq, ValueType(0));
}
auto sq::sum() { return [] (auto&& seq) { return sum(seq); }; }




/**
 * @brief      Specializes reduce with std::multiplies.
 *
 * @param[in]  seq        The sequence whose product is computed
 *
 * @tparam     ValueType  The sequence value type
 * @tparam     Rank       The sequence rank
 *
 * @return     The product of the elements
 */
template<typename ValueType, std::size_t Rank>
auto sq::product(const sequence_t<ValueType, Rank>& seq)
{
    return reduce(std::multiplies<>(), seq, ValueType(1));
}
auto sq::product() { return [] (auto&& seq) { return product(seq); }; }




/**
 * @brief      Return whether all of the elements of a sequence satisfy the
 *             given predicate.
 *
 * @param[in]  seq        The sequence to test
 * @param[in]  pred       The predicate function
 *
 * @tparam     ValueType  The value type
 * @tparam     Predicate  The type of the predicate function
 * @tparam     Rank       The rank of the sequence
 *
 * @return     A boolean
 */
template<typename ValueType, typename Predicate, std::size_t Rank>
auto sq::all_of(const sequence_t<ValueType, Rank>& seq, Predicate pred)
{
    for (std::size_t i = 0; i < Rank; ++i)
        if (! pred(seq[i]))
            return false;
    return true;
}




/**
 * @brief      Return whether any of the elements of a sequence satisfy the
 *             given predicate.
 *
 * @param[in]  seq        The sequence to test
 * @param[in]  pred       The predicate function
 *
 * @tparam     ValueType  The value type
 * @tparam     Predicate  The type of the predicate function
 * @tparam     Rank       The rank of the sequence
 *
 * @return     A boolean
 */
template<typename ValueType, typename Predicate, std::size_t Rank>
auto sq::any_of(const sequence_t<ValueType, Rank>& seq, Predicate pred)
{
    for (std::size_t i = 0; i < Rank; ++i)
        if (pred(seq[i]))
            return true;
    return false;
}




/**
 * @brief      Return true of any of the sequence's elements are equal to the
 *             given value.
 *
 * @param[in]  seq        The sequence that might contain the value
 * @param[in]  value      The value to check for
 *
 * @tparam     ValueType  The sequence value type
 * @tparam     Rank       The sequence rank
 *
 * @return     A boolean
 */
template<typename ValueType, std::size_t Rank>
auto sq::contains(const sequence_t<ValueType, Rank>& seq, ValueType value)
{
    return sq::any_of(seq, [value] (auto s) { return s == value; });
}
template<typename ValueType>
auto sq::contains(ValueType value) { return [value] (auto&& seq) { return contains(seq, value); }; }




/**
 * @brief      Read indexes from a sequence.
 *
 * @param[in]  seq         The sequence to read from
 * @param[in]  indexes     The indexes to read
 *
 * @tparam     ValueType   The value type
 * @tparam     Rank        The rank of the sequence
 * @tparam     NumIndexes  The number of indexes to read
 *
 * @return     A new sequence
 */
template<typename ValueType, std::size_t Rank, std::size_t NumIndexes>
auto sq::read_indexes(const sequence_t<ValueType, Rank>& seq, const sequence_t<std::size_t, NumIndexes>& indexes)
{
    auto result = sequence_t<ValueType, NumIndexes>();

    for (std::size_t i = 0; i < NumIndexes; ++i)
        result[i] = seq.at(indexes[i]);

    return result;
}

template<typename... Args>
auto sq::read_indexes(Args... args)
{
    return [indexes = make_sequence(std::size_t(args)...)] (auto&& seq) { return read_indexes(seq, indexes); };
}




/**
 * @brief      Return a sequence with new elements inserted at the given
 *             indexes.
 *
 * @param[in]  source       The starting sequence
 * @param[in]  values       The elements to insert
 * @param[in]  indexes      The indexes at which to insert the new elements
 *
 * @tparam     ValueType    The value type
 * @tparam     Rank         The rank of the starting sequence
 * @tparam     NumToInsert  The number of elements being inserted
 *
 * @return     A lengthened sequence
 * @note       If an index is greater than rank, or if the indexes are not
 *             unique, this function throws out-of-range.
 */
template<typename ValueType, std::size_t Rank, std::size_t NumToInsert>
auto sq::insert_elements(
    const sequence_t<ValueType, Rank>& source,
    const sequence_t<ValueType, NumToInsert>& values,
    const sequence_t<std::size_t, NumToInsert>& indexes)
{
    auto source1_n = std::size_t(0);
    auto source2_n = std::size_t(0);
    auto result = sequence_t<ValueType, Rank + NumToInsert>();

    for (std::size_t n = 0; n < result.size(); ++n)
    {
        if (indexes | contains(n))
        {
            result[n] = values[source2_n++];
        }
        else
        {
            result[n] = source[source1_n++];
        }
    }

    if (source2_n != NumToInsert)
    {
        throw std::out_of_range("sq::insert_elements");
    }
    return result;
}




/**
 * @brief      Return a sequence with the elements at the given indexes removed.
 *
 * @param[in]  source       The starting sequence
 * @param[in]  indexes      The indexes to remove
 *
 * @tparam     ValueType    The value type
 * @tparam     Rank         The rank of the starting sequence
 * @tparam     NumToRemove  The number of indexes being removed
 *
 * @return     A shortened sequence
 * @note       This function throws out-of-range if any of the indexes are
 *             greater than or equal to Rank.
 */
template<typename ValueType, std::size_t Rank, std::size_t NumToRemove>
auto sq::remove_indexes(
    const sequence_t<ValueType, Rank>& source,
    const sequence_t<std::size_t, NumToRemove>& indexes)
{
    static_assert(NumToRemove <= Rank, "cannot remove more elements than sequence size");

    auto target_n = std::size_t(0);
    auto result = sequence_t<ValueType, Rank - NumToRemove>();

    for (std::size_t n = 0; n < source.size(); ++n)
    {
        if (! contains(indexes, n))
        {
            result[target_n++] = source[n];
        }
    }

    if (target_n != result.size())
    {
        throw std::out_of_range("sq::remove_elements");
    }
    return result;
}
