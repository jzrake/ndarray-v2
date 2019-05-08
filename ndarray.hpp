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
#include <utility>           // std::index_sequence




//=============================================================================
namespace nd
{


    // array support structs
    //=========================================================================
    template<std::size_t Rank, typename ValueType, typename DerivedType> class short_sequence_t;
    template<std::size_t Rank, typename ValueType> class basic_sequence_t;
    template<std::size_t Rank> class shape_t;
    template<std::size_t Rank> class index_t;
    template<std::size_t Rank> class jumps_t;
    template<std::size_t Rank> class memory_strides_t;
    template<std::size_t Rank> class access_pattern_t;
    template<typename Provider> class array_t;
    template<typename ValueType> class buffer_t;


    // array and access pattern factory functions
    //=========================================================================
    template<typename... Args> auto make_shape(Args... args);
    template<typename... Args> auto make_index(Args... args);
    template<typename... Args> auto make_jumps(Args... args);
    template<std::size_t Rank, typename Arg> auto make_uniform_shape(Arg arg);
    template<std::size_t Rank, typename Arg> auto make_uniform_index(Arg arg);
    template<std::size_t Rank, typename Arg> auto make_uniform_jumps(Arg arg);
    template<std::size_t Rank> auto make_strides_row_major(shape_t<Rank> shape);
    template<std::size_t Rank> auto make_access_pattern(shape_t<Rank> shape);
    template<typename... Args> auto make_access_pattern(Args... args);
    template<std::size_t NumPartitions, std::size_t Rank> auto partition_shape(shape_t<Rank> shape);


    // provider types
    //=========================================================================
    template<typename Function, std::size_t Rank> class basic_provider_t;
    template<std::size_t Rank, typename ValueType> class shared_provider_t;
    template<std::size_t Rank, typename ValueType> class unique_provider_t;
    template<std::size_t Rank, typename ValueType> class uniform_provider_t;


    // provider factory functions
    //=========================================================================
    template<typename ValueType, std::size_t Rank> auto make_shared_provider(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_shared_provider(Args... args);
    template<typename ValueType, std::size_t Rank> auto make_unique_provider(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_unique_provider(Args... args);
    template<typename ValueType, std::size_t Rank> auto make_uniform_provider(ValueType value, shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_uniform_provider(ValueType value, Args... args);
    template<typename Provider> auto evaluate_as_shared(Provider&&);
    template<typename Provider> auto evaluate_as_unique(Provider&&);


    // array factory functions
    //=========================================================================
    template<typename Provider> auto make_array(Provider&&);
    template<typename Mapping, std::size_t Rank> auto make_array(Mapping mapping, shape_t<Rank> shape);
    template<typename ValueType, std::size_t Rank> auto make_shared_array(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_shared_array(Args... args);
    template<typename ValueType, std::size_t Rank> auto make_unique_array(shape_t<Rank> shape);
    template<typename ValueType, typename... Args> auto make_unique_array(Args... args);
    template<std::size_t Rank> auto index_array(shape_t<Rank> shape);
    template<typename... Args> auto index_array(Args... args);
    template<typename... ArrayTypes> auto zip_arrays(ArrayTypes... arrays);
    template<typename... ArrayTypes> auto cartesian_product(ArrayTypes... arrays);
    template<typename ArrayType> auto where(ArrayType array);
    template<typename ValueType=int, typename... Args> auto zeros(Args... args);
    template<typename ValueType=int, typename... Args> auto ones(Args... args);
    template<typename ValueType, std::size_t Rank> auto promote(ValueType, shape_t<Rank>);


    // array operator support structs
    //=========================================================================
    class shifter_t;
    class axis_selector_t;
    template<std::size_t Rank> class selector_t;
    template<std::size_t Rank, typename ArrayType> class replacer_t;
    template<std::size_t RankDifference> class axis_freezer_t;
    template<typename ArrayType> class axis_reducer_t;
    template<typename ArrayType> class concatenator_t;


    // array operator factory functions
    //=========================================================================
    inline auto to_shared();
    inline auto to_unique();
    inline auto bounds_check();
    inline auto sum();
    inline auto all();
    inline auto any();
    inline auto shift_by(int delta);
    inline auto select_axis(std::size_t axis_to_select);
    inline auto freeze_axis(std::size_t axis_to_freeze);
    template<typename OperatorType> auto collect(OperatorType reduction);
    template<typename ArrayType> auto concat(ArrayType array_to_concat);
    template<typename ArrayType> auto read_indexes(ArrayType array_of_indexes);
    template<std::size_t Rank> auto reshape(shape_t<Rank> shape);
    template<typename... Args> auto reshape(Args... args);
    template<std::size_t Rank> auto select(access_pattern_t<Rank>);
    template<std::size_t Rank, typename ArrayType> auto replace(access_pattern_t<Rank>, ArrayType);
    template<std::size_t Rank> auto select_from(index_t<Rank> starting_index);
    template<typename... Args> auto select_from(Args... args);
    template<std::size_t Rank> auto replace_from(index_t<Rank> starting_index);
    template<typename... Args> auto replace_from(Args... args);
    template<std::size_t Rank> auto read_index(index_t<Rank>);
    template<typename... Args> auto read_index(Args... args);
    template<typename Function> auto transform(Function function);
    template<typename Function> auto binary_op(Function function);


    // array query support
    //=========================================================================
    template<typename ArrayType> using value_type_of = typename std::remove_reference_t<ArrayType>::value_type;
    template<typename ArrayType> constexpr std::size_t rank(ArrayType&&) { return std::remove_reference_t<ArrayType>::rank; }


    // convenience typedef's
    //=========================================================================
    template<typename ValueType, std::size_t Rank>
    using shared_array = array_t<shared_provider_t<Rank, ValueType>>;

    template<typename ValueType, std::size_t Rank>
    using unique_array = array_t<unique_provider_t<Rank, ValueType>>;


    // algorithm support structs
    //=========================================================================
    template<typename ValueType> class range_container_t;
    template<typename ValueType, typename ContainerTuple> class zipped_container_t;
    template<typename ContainerType, typename Function> class transformed_container_t;


    // std::algorithm wrappers for ranges
    //=========================================================================
    template<typename Range, typename Seed, typename Function> auto accumulate(Range&& rng, Seed&& seed, Function&& fn);
    template<typename Range, typename Predicate> auto all_of(Range&& rng, Predicate&& pred);
    template<typename Range, typename Predicate> auto any_of(Range&& rng, Predicate&& pred);
    template<typename Range> auto distance(Range&& rng);
    template<typename Range> auto enumerate(Range&& rng);
    template<typename ValueType> auto range(ValueType count);
    template<typename... ContainerTypes> auto zip(ContainerTypes&&... containers);


    // helper functions
    //=========================================================================
    namespace detail
    {
        template<typename Function, typename Tuple, std::size_t... Is>
        auto transform_tuple_impl(Function&& fn, const Tuple& t, std::index_sequence<Is...>);

        template<typename Function, typename Tuple>
        auto transform_tuple(Function&& fn, const Tuple& t);

        template<typename FunctionTuple, typename ArgumentTuple, std::size_t... Is>
        auto zip_apply_tuple_impl(FunctionTuple&& fn, ArgumentTuple&& args, std::index_sequence<Is...>);

        template<typename FunctionTuple, typename ArgumentTuple>
        auto zip_apply_tuple(FunctionTuple&& fn, ArgumentTuple&& args);

        template<typename ResultSequence, typename SourceSequence, typename IndexContainer>
        auto read_elements(const SourceSequence& source, IndexContainer indexes);

        template<typename ResultSequence, typename SourceSequence, typename IndexContainer, typename Sequence>
        auto insert_elements(const SourceSequence& source, IndexContainer indexes, Sequence values);

        template<typename ResultSequence, typename SourceSequence, typename IndexContainer>
        auto remove_elements(const SourceSequence& source, IndexContainer indexes);
    }
}




//=============================================================================
template<typename ValueType>
class nd::range_container_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = ValueType;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++() { ++current; return *this; }
        bool operator==(const iterator& other) const { return current == other.current; }
        bool operator!=(const iterator& other) const { return current != other.current; }
        const ValueType& operator*() const { return current; }
        ValueType current = 0;
        ValueType start = 0;
        ValueType final = 0;
    };

    //=========================================================================
    range_container_t(ValueType start, ValueType final) : start(start), final(final) {}
    iterator begin() const { return { 0, start, final }; }
    iterator end() const { return { final, start, final }; }

    template<typename Function>
    auto operator|(Function&& fn) const
    {
        return transformed_container_t<range_container_t, Function>(*this, fn);
    }

private:
    //=========================================================================
    ValueType start = 0;
    ValueType final = 0;
};




//=============================================================================
template<typename ValueType, typename ContainerTuple>
class nd::zipped_container_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    template<typename IteratorTuple>
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = zipped_container_t::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++()
        {
            iterators = detail::transform_tuple([] (auto x) { return ++x; }, iterators);
            return *this;
        }
        bool operator==(const iterator& other) const { return iterators == other.iterators; }
        bool operator!=(const iterator& other) const { return iterators != other.iterators; }
        auto operator*() const { return detail::transform_tuple([] (const auto& x) { return std::ref(*x); }, iterators); }

        IteratorTuple iterators;
    };

    //=========================================================================
    zipped_container_t(ContainerTuple&& containers) : containers(containers) {}

    auto begin() const
    {
        auto res = detail::transform_tuple([] (const auto& x) { return std::begin(x); }, containers);
         return iterator<decltype(res)>{res};
    }

    auto end() const
    {
        auto res = detail::transform_tuple([] (const auto& x) { return std::end(x); }, containers);
        return iterator<decltype(res)>{res};
    }

    template<typename Function>
    auto operator|(Function&& fn) const
    {
        return transformed_container_t<zipped_container_t, Function>(*this, fn);
    }

private:
    //=========================================================================
    ContainerTuple containers;
};




//=============================================================================
template<typename ContainerType, typename Function>
class nd::transformed_container_t
{
public:
    using value_type = std::invoke_result_t<Function, typename ContainerType::value_type>;

    //=========================================================================
    template<typename IteratorType>
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = transformed_container_t::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++() { ++current; return *this; }
        bool operator==(const iterator& other) const { return current == other.current; }
        bool operator!=(const iterator& other) const { return current != other.current; }
        auto operator*() const { return function(*current); }

        IteratorType current;
        const Function& function;
    };

    //=========================================================================
    transformed_container_t(const ContainerType& container, const Function& function)
    : container(container)
    , function(function) {}

    auto begin() const { return iterator<decltype(container.begin())> {container.begin(), function}; }
    auto end() const { return iterator<decltype(container.end())> {container.end(), function}; }

private:
    //=========================================================================
    const ContainerType& container;
    const Function& function;
};




//=============================================================================
template<typename Range, typename Seed, typename Function>
auto nd::accumulate(Range&& rng, Seed&& seed, Function&& fn)
{
    return std::accumulate(rng.begin(), rng.end(), std::forward<Seed>(seed), std::forward<Function>(fn));
}

template<typename Range, typename Predicate>
auto nd::all_of(Range&& rng, Predicate&& pred)
{
    return std::all_of(rng.begin(), rng.end(), pred);
}

template<typename Range, typename Predicate>
auto nd::any_of(Range&& rng, Predicate&& pred)
{
    return std::any_of(rng.begin(), rng.end(), pred);
}

template<typename Range>
auto nd::distance(Range&& rng)
{
    return std::distance(rng.begin(), rng.end());
}

template<typename Range>
auto nd::enumerate(Range&& rng)
{
    return zip(range(distance(std::forward<Range>(rng))), std::forward<Range>(rng));
}

template<typename ValueType>
auto nd::range(ValueType count)
{
    return nd::range_container_t<ValueType>(0, count);
}

template<typename... ContainerTypes>
auto nd::zip(ContainerTypes&&... containers)
{
    using ValueType = std::tuple<typename std::remove_reference_t<ContainerTypes>::value_type...>;
    using ContainerTuple = std::tuple<ContainerTypes...>;
    return nd::zipped_container_t<ValueType, ContainerTuple>(std::forward_as_tuple(containers...));
}




//=============================================================================
template<std::size_t Rank, typename ValueType, typename DerivedType>
class nd::short_sequence_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    static DerivedType uniform(ValueType arg)
    {
        DerivedType result;

        for (auto n : range(Rank))
        {
            result.memory[n] = arg;
        }
        return result;
    }

    template<typename Range>
    static DerivedType from_range(Range&& rng)
    {
        if (distance(rng) != Rank)
        {
            throw std::logic_error("sequence constructed from range of wrong size");
        }
        DerivedType result;

        for (const auto& [n, a] : enumerate(rng))
        {
            result.memory[n] = a;
        }
        return result;
    }

    short_sequence_t()
    {
        for (auto n : range(Rank))
        {
            memory[n] = ValueType();
        }
    }

    short_sequence_t(std::initializer_list<ValueType> args)
    {
        for (const auto& [n, a] : enumerate(args))
        {
            memory[n] = a;
        }
    }

    bool operator==(const DerivedType& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) == std::get<1>(t); });
    }

    bool operator!=(const DerivedType& other) const
    {
        return any_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) != std::get<1>(t); });
    }

    constexpr std::size_t size() const { return Rank; }
    const ValueType* data() const { return memory; }
    const ValueType* begin() const { return memory; }
    const ValueType* end() const { return memory + Rank; }
    const ValueType& operator[](std::size_t n) const { return memory[n]; }
    ValueType* data() { return memory; }
    ValueType* begin() { return memory; }
    ValueType* end() { return memory + Rank; }
    ValueType& operator[](std::size_t n) { return memory[n]; }

private:
    //=========================================================================
    ValueType memory[Rank];
};




//=============================================================================
template<std::size_t Size, typename ValueType>
class nd::basic_sequence_t : public nd::short_sequence_t<Size, ValueType, basic_sequence_t<Size, ValueType>>
{
};




//=============================================================================
template<std::size_t Rank>
class nd::shape_t : public nd::short_sequence_t<Rank, std::size_t, shape_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, shape_t<Rank>>::short_sequence_t;

    std::size_t volume() const { return accumulate(*this, 1, std::multiplies<>()); }

    bool contains(const index_t<Rank>& index) const
    {
        return all_of(zip(index, *this), [] (const auto& t) { return std::get<0>(t) < std::get<1>(t); });
    }

    template<typename... Args>
    bool contains(Args... args) const
    {
        return contains(make_index(args...));
    }

    template<typename IndexContainer>
    auto read_elements(IndexContainer indexes) const
    {
        return detail::read_elements<shape_t<indexes.size()>>(*this, indexes);
    }

    template<typename IndexContainer, typename Sequence>
    auto insert_elements(IndexContainer indexes, Sequence values) const
    {
        return detail::insert_elements<shape_t<Rank + indexes.size()>>(*this, indexes, values);
    }

    template<typename IndexContainer>
    auto remove_elements(IndexContainer indexes) const
    {
        return detail::remove_elements<shape_t<Rank - indexes.size()>>(*this, indexes);
    }

    index_t<Rank> last_index() const
    {
        auto result = index_t<Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = this->operator[](n);
        }
        return result;
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::index_t : public nd::short_sequence_t<Rank, std::size_t, index_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, index_t<Rank>>::short_sequence_t;

    template <size_t... Is>
    auto as_tuple(std::index_sequence<Is...>) const
    {
        return std::make_tuple(this->operator[](Is)...);
    }

    auto as_tuple() const
    {
        return as_tuple(std::make_index_sequence<Rank>());
    }

    template<typename IndexContainer>
    auto read_elements(IndexContainer indexes) const
    {
        return detail::read_elements<index_t<indexes.size()>>(*this, indexes);
    }

    template<typename IndexContainer, typename Sequence>
    auto insert_elements(IndexContainer indexes, Sequence values) const
    {
        return detail::insert_elements<index_t<Rank + indexes.size()>>(*this, indexes, values);
    }

    template<typename IndexContainer>
    auto remove_elements(IndexContainer indexes) const
    {
        return detail::remove_elements<index_t<Rank - indexes.size()>>(*this, indexes);
    }

    bool operator<(const index_t<Rank>& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) < std::get<1>(t); });
    }
    bool operator>(const index_t<Rank>& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) > std::get<1>(t); });
    }
    bool operator<=(const index_t<Rank>& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) <= std::get<1>(t); });
    }
    bool operator>=(const index_t<Rank>& other) const
    {
        return all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) >= std::get<1>(t); });
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::jumps_t : public nd::short_sequence_t<Rank, long, jumps_t<Rank>>
{
public:
    using short_sequence_t<Rank, long, jumps_t<Rank>>::short_sequence_t;

    template<typename IndexContainer>
    auto remove_elements(IndexContainer indexes) const
    {
        return detail::remove_elements<jumps_t<Rank - indexes.size()>>(*this, indexes);
    }

    template<typename IndexContainer, typename Sequence>
    auto insert_elements(IndexContainer indexes, Sequence values) const
    {
        return detail::insert_elements<jumps_t<Rank + indexes.size()>>(*this, indexes, values);
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::memory_strides_t : public nd::short_sequence_t<Rank, std::size_t, memory_strides_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, memory_strides_t<Rank>>::short_sequence_t;

    std::size_t compute_offset(const index_t<Rank>& index) const
    {
        auto mul_tuple = [] (auto t) { return std::get<0>(t) * std::get<1>(t); };
        return accumulate(zip(index, *this) | mul_tuple, 0, std::plus<>());
    }

    template<typename... Args>
    std::size_t compute_offset(Args... args) const
    {
        return compute_offset(make_index(args...));
    }
};




//=============================================================================
template<std::size_t Rank>
class nd::access_pattern_t
{
public:

    using value_type = index_t<Rank>;
    static constexpr std::size_t rank = Rank;

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
    template<typename... Args> access_pattern_t with_start(Args... args) const { return { make_index(args...), final, jumps }; }
    template<typename... Args> access_pattern_t with_final(Args... args) const { return { start, make_index(args...), jumps }; }
    template<typename... Args> access_pattern_t with_jumps(Args... args) const { return { start, final, make_jumps(args...) }; }

    access_pattern_t with_start(index_t<Rank> arg) const { return { arg, final, jumps }; }
    access_pattern_t with_final(index_t<Rank> arg) const { return { start, arg, jumps }; }
    access_pattern_t with_jumps(jumps_t<Rank> arg) const { return { start, final, arg }; }

    std::size_t size() const
    {
        return shape().volume();
    }

    auto shape() const
    {
        auto s = shape_t<Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            s[n] = final[n] / jumps[n] - start[n] / jumps[n];
        }
        return s;
    }

    bool empty() const
    {
        return any_of(shape(), [] (auto s) { return s == 0; });
    }

    bool contiguous() const
    {
        return
        start == make_uniform_index<Rank>(0) &&
        jumps == make_uniform_jumps<Rank>(1);
    }

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

    index_t<Rank> map_index(const index_t<Rank>& index) const
    {
        index_t<Rank> result;

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = start[n] + jumps[n] * index[n];
        }
        return result;
    }

    index_t<Rank> inverse_map_index(const index_t<Rank>& mapped_index) const
    {
        index_t<Rank> result;

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = (mapped_index[n] - start[n]) / jumps[n];
        }
        return result;
    }

    /**
     * Return true if this is a valid mapped-from index.
     */
    bool contains(const index_t<Rank>& index) const
    {
        return shape().contains(index);
    }

    template<typename... Args>
    bool contains(Args... args) const
    {
        return contains(make_index(args...));
    }

    /**
     * Return true if an iteration over this accessor would generate the given
     * index, that is, if it the index included in the set of mapped-to indexes.
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
    bool generates(Args... args) const
    {
        return generates(make_index(args...));
    }

    /**
     * Return false if this access pattern would generate any indexes not
     * contained in the given shape.
     */
    bool within(const shape_t<Rank>& parent_shape) const
    {
        auto zero = make_uniform_index<Rank>(0);
        auto t1 = map_index(zero);
        auto t2 = map_index(shape().last_index());

        return (t1 >= zero && t1 <= parent_shape.last_index() &&
                t2 >= zero && t2 <= parent_shape.last_index());
    }

    iterator begin() const { return { *this, start }; }
    iterator end() const { return { *this, final }; }

    //=========================================================================
    index_t<Rank> start = make_uniform_index<Rank>(0);
    index_t<Rank> final = make_uniform_index<Rank>(0);
    jumps_t<Rank> jumps = make_uniform_jumps<Rank>(1);
};




//=============================================================================
// Shape, index, and access pattern factories
//=============================================================================




template<typename... Args>
auto nd::make_shape(Args... args)
{
    return shape_t<sizeof...(Args)>({std::size_t(args)...});
}

template<typename... Args>
auto nd::make_index(Args... args)
{
    return index_t<sizeof...(Args)>({std::size_t(args)...});
}

template<typename... Args>
auto nd::make_jumps(Args... args)
{
    return jumps_t<sizeof...(Args)>({long(args)...});
}

template<std::size_t Rank, typename Arg>
auto nd::make_uniform_shape(Arg arg)
{
    return shape_t<Rank>::uniform(arg);
}

template<std::size_t Rank, typename Arg>
auto nd::make_uniform_index(Arg arg)
{
    return index_t<Rank>::uniform(arg);
}

template<std::size_t Rank, typename Arg>
auto nd::make_uniform_jumps(Arg arg)
{
    return jumps_t<Rank>::uniform(arg);
}

template<std::size_t Rank>
auto nd::make_strides_row_major(shape_t<Rank> shape)
{
    auto result = memory_strides_t<Rank>();

    result[Rank - 1] = 1;

    if constexpr (Rank > 1)
    {
        for (int n = Rank - 2; n >= 0; --n)
        {
            result[n] = result[n + 1] * shape[n + 1];
        }
    }
    return result;
}

template<std::size_t Rank>
auto nd::make_access_pattern(shape_t<Rank> shape)
{
    return access_pattern_t<Rank>().with_final(index_t<Rank>::from_range(shape));
}

template<typename... Args>
auto nd::make_access_pattern(Args... args)
{
    return access_pattern_t<sizeof...(Args)>().with_final(args...);
}

template<std::size_t NumPartitions, std::size_t Rank>
auto nd::partition_shape(shape_t<Rank> shape)
{
    // Note: this function should handle remainders better
    constexpr std::size_t D = 0;
    auto result = basic_sequence_t<NumPartitions, access_pattern_t<Rank>>();
    auto chunk_size = shape[D] / NumPartitions;

    for (std::size_t n = 0; n < NumPartitions; ++n)
    {
        auto pattern = make_access_pattern(shape);
        pattern.start[D] = chunk_size * n;
        pattern.final[D] = n == NumPartitions - 1 ? shape[D] : chunk_size * (n + 1);
        result[n] = pattern;
    }
    return result;
}




//=============================================================================
class nd::shifter_t
{
public:

    //=========================================================================
    shifter_t(std::size_t axis_to_shift, int delta) : axis_to_shift(axis_to_shift), delta(delta) {}

    template<typename ArrayType>
    auto operator()(ArrayType&& array) const
    {
        if (axis_to_shift >= rank(array))
        {
            throw std::logic_error("cannot shift axis greater than or equal to array rank");
        }
        if (std::size_t(std::abs(delta)) >= array.shape(axis_to_shift))
        {
            throw std::logic_error("cannot shift an array by more than its length on that axis");
        }
        auto mapping = [axis_to_shift=axis_to_shift, delta=delta, array] (auto index)
        {
            index[axis_to_shift] -= delta;
            return array(index);
        };
        auto shape = array.shape();
        shape[axis_to_shift] -= std::abs(delta);

        return make_array(mapping, shape);
    }

    auto along_axis(std::size_t new_axis_to_shift) const
    {
        return shifter_t(new_axis_to_shift, delta);
    }

private:
    //=========================================================================
    std::size_t axis_to_shift;
    int delta;
};




//=============================================================================
class nd::axis_selector_t
{
public:

    //=========================================================================
    axis_selector_t(std::size_t axis_to_select, std::size_t start, std::size_t final, bool is_final_from_the_end)
    : axis_to_select(axis_to_select)
    , start(start)
    , final(final)
    , is_final_from_the_end(is_final_from_the_end) {}

    template<typename ArrayType>
    auto operator()(ArrayType&& array) const
    {
        if (axis_to_select >= rank(array))
        {
            throw std::logic_error("cannot select axis greater than or equal to array rank");
        }
        auto mapping = [axis_to_select=axis_to_select, start=start, array] (auto index)
        {
            index[axis_to_select] += start;
            return array(index);
        };

        auto shape = array.shape();
        shape[axis_to_select] -= start + (is_final_from_the_end ? final : (shape[axis_to_select] - final));

        return make_array(mapping, shape);
    }

    auto from(std::size_t new_start) const
    {
        return axis_selector_t(axis_to_select, new_start, final, is_final_from_the_end);
    }
    auto to(std::size_t new_final) const
    {
        return axis_selector_t(axis_to_select, start, new_final, is_final_from_the_end);
    }
    auto from_the_end() const
    {
        return axis_selector_t(axis_to_select, start, final, true);        
    }

private:
    //=========================================================================
    std::size_t axis_to_select;
    std::size_t start;
    std::size_t final;
    bool is_final_from_the_end;
};




//=============================================================================
template<std::size_t Rank>
class nd::selector_t
{
public:

    //=========================================================================
    selector_t(access_pattern_t<Rank> region=access_pattern_t<Rank>()) : region(region) {}

    template<typename ArrayType>
    auto operator()(ArrayType&& array) const
    {
        if (! region.within(array.shape()))
        {
            throw std::logic_error("out-of-bounds selection");
        }
        auto mapping = [region=region, array] (auto&& index) { return array(region.map_index(index)); };
        return make_array(basic_provider_t<decltype(mapping), Rank>(mapping, region.shape()));
    }

    template<typename... Args> auto from   (Args... args) const { return from   (make_index(args...)); }
    template<typename... Args> auto to     (Args... args) const { return to     (make_index(args...)); }
    template<typename... Args> auto jumping(Args... args) const { return jumping(make_jumps(args...)); }
    auto from   (index_t<Rank> arg) const { return selector_t(region.with_start(arg)); }
    auto to     (index_t<Rank> arg) const { return selector_t(region.with_final(arg)); }
    auto jumping(jumps_t<Rank> arg) const { return selector_t(region.with_jumps(arg)); }

private:
    //=========================================================================
    access_pattern_t<Rank> region;
};




//=============================================================================
template<std::size_t Rank, typename ArrayType>
class nd::replacer_t
{
public:

    //=========================================================================
    replacer_t(access_pattern_t<Rank> region=access_pattern_t<Rank>()) : region(region) {}
    replacer_t(access_pattern_t<Rank> region, ArrayType replacement_array)
    : region(region)
    , replacement_array(replacement_array) {}

    template<typename PatchArrayType>
    auto operator()(PatchArrayType&& array_to_patch) const
    {
        if (region.shape() != replacement_array.shape())
        {
            throw std::logic_error("region to replace has a different shape than the replacement array");
        }

        auto mapping = [region=region, replacement_array=replacement_array, array_to_patch] (auto&& index)
        {
            if (region.generates(index))
            {
                return replacement_array(region.inverse_map_index(index));
            }
            return array_to_patch(index);
        };
        return make_array(mapping, array_to_patch.shape());
    }

    template<typename... Args> auto from   (Args... args) const { return from   (make_index(args...)); }
    template<typename... Args> auto to     (Args... args) const { return to     (make_index(args...)); }
    template<typename... Args> auto jumping(Args... args) const { return jumping(make_jumps(args...)); }
    auto from   (index_t<Rank> arg) const { return replacer_t(region.with_start(arg), replacement_array); }
    auto to     (index_t<Rank> arg) const { return replacer_t(region.with_final(arg), replacement_array); }
    auto jumping(jumps_t<Rank> arg) const { return replacer_t(region.with_jumps(arg), replacement_array); }

    template<typename OtherArrayType>
    auto with(OtherArrayType&& new_replacement_array) const
    {
        return replacer_t<Rank, OtherArrayType>(region, std::forward<OtherArrayType>(new_replacement_array));
    }

private:
    //=========================================================================
    access_pattern_t<Rank> region;
    ArrayType replacement_array;
};




//=============================================================================
template<std::size_t RankDifference>
class nd::axis_freezer_t
{
public:

    //=========================================================================
    axis_freezer_t(
        index_t<RankDifference> axes_to_freeze,
        index_t<RankDifference> index_to_freeze_at=make_uniform_index<RankDifference>(0))
    : axes_to_freeze(axes_to_freeze)
    , index_to_freeze_at(index_to_freeze_at) {}

    template<typename PatchArrayType>
    auto operator()(PatchArrayType array) const
    {
        if (any_of(axes_to_freeze, [array] (auto a) { return a >= rank(array); }))
        {
            throw std::logic_error("cannot freeze axis greater than or equal to array rank");
        }
        auto mapping = [axes_to_freeze=axes_to_freeze, index_to_freeze_at=index_to_freeze_at, array] (auto&& index)
        {
            return array(index.insert_elements(axes_to_freeze, index_to_freeze_at));
        };
        auto shape = array.shape().remove_elements(axes_to_freeze);

        return make_array(mapping, shape);
    }

    auto at_index(index_t<RankDifference> new_index_to_freeze_at) const
    {
        return axis_freezer_t(axes_to_freeze, new_index_to_freeze_at);
    }

    template<typename... Args>
    auto at_index(Args... new_index_to_freeze_at) const
    {
        static_assert(sizeof...(Args) == RankDifference);
        return at_index(make_index(new_index_to_freeze_at...));
    }

private:
    //=========================================================================
    index_t<RankDifference> axes_to_freeze;
    index_t<RankDifference> index_to_freeze_at;
};




//=============================================================================
template<typename OperatorType>
class nd::axis_reducer_t
{
public:

    //=========================================================================
    axis_reducer_t(std::size_t axis_to_reduce, OperatorType the_operator)
    : axis_to_reduce(axis_to_reduce)
    , the_operator(the_operator) {}

    template<typename ArrayType>
    auto operator()(ArrayType array) const
    {
        if (axis_to_reduce >= rank(array))
        {
            throw std::logic_error("cannot reduce axis greater than or equal to array rank");
        }
        constexpr std::size_t R = ArrayType::rank;

        auto mapping = [the_operator=the_operator, axis_to_reduce=axis_to_reduce, array] (auto&& index)
        {
            auto axes_to_freeze = index_t<R>::from_range(range(R)).remove_elements(make_index(axis_to_reduce));
            auto freezer = axis_freezer_t<R - 1>(axes_to_freeze).at_index(index.read_elements(axes_to_freeze));
            return the_operator(freezer(array));
        };
        auto shape = array.shape().remove_elements(make_index(axis_to_reduce));

        return make_array(mapping, shape);
    }

    auto along_axis(std::size_t new_axis_to_reduce) const
    {
        return axis_reducer_t(new_axis_to_reduce, the_operator);
    }

private:
    //=========================================================================
    std::size_t axis_to_reduce;
    OperatorType the_operator;
};




//=============================================================================
template<typename ArrayType>
class nd::concatenator_t
{
public:

    //=========================================================================
    concatenator_t(std::size_t axis_to_extend, ArrayType array_to_concat)
    : axis_to_extend(axis_to_extend)
    , array_to_concat(array_to_concat) {}

    template<typename SourceArrayType>
    auto operator()(SourceArrayType array) const
    {
        if (axis_to_extend >= rank(array))
        {
            throw std::logic_error("cannot concatenate on axis greater than or equal to array rank");
        }
        if (array_to_concat.shape().remove_elements(make_index(axis_to_extend))
            !=        array.shape().remove_elements(make_index(axis_to_extend)))
        {
            throw std::logic_error("the shape of the concatenated arrays can only differ on the concatenating axis");
        }

        auto mapping = [axis_to_extend=axis_to_extend, array_to_concat=array_to_concat, array] (auto index)
        {
            if (index[axis_to_extend] >= array.shape(axis_to_extend))
            {
                index[axis_to_extend] -= array.shape(axis_to_extend);
                return array_to_concat(index);
            }
            return array(index);
        };

        auto shape = array.shape();
        shape[axis_to_extend] += array_to_concat.shape(axis_to_extend);

        return make_array(mapping, shape);
    }

    auto on_axis(std::size_t new_axis_to_concat) const
    {
        return concatenator_t(new_axis_to_concat, array_to_concat);
    }

private:
    //=========================================================================
    std::size_t axis_to_extend;
    ArrayType array_to_concat;
};




//=============================================================================
template<typename Function, std::size_t Rank>
class nd::basic_provider_t
{
public:

    using value_type = std::invoke_result_t<Function, index_t<Rank>>;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    basic_provider_t(Function mapping, shape_t<Rank> the_shape) : mapping(mapping), the_shape(the_shape) {}
    decltype(auto) operator()(const index_t<Rank>& index) const { return mapping(index); }
    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }

private:
    //=========================================================================
    Function mapping;
    shape_t<Rank> the_shape;
};




//=============================================================================
template<std::size_t Rank, typename ValueType>
class nd::shared_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    shared_provider_t(nd::shape_t<Rank> the_shape, std::shared_ptr<nd::buffer_t<ValueType>> buffer)
    : the_shape(the_shape)
    , strides(make_strides_row_major(the_shape))
    , buffer(buffer)
    {
        if (the_shape.volume() != buffer->size())
        {
            throw std::logic_error("shape and buffer sizes do not match");
        }
    }

    const ValueType& operator()(const index_t<Rank>& index) const
    {
        return buffer->operator[](strides.compute_offset(index));
    }

    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }
    const ValueType* data() const { return buffer->data(); }
    template<std::size_t R> auto reshape(shape_t<R> new_shape) const { return shared_provider_t<R, ValueType>(new_shape, buffer); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
    memory_strides_t<Rank> strides;
    std::shared_ptr<buffer_t<ValueType>> buffer;
};




//=============================================================================
template<std::size_t Rank, typename ValueType>
class nd::unique_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    unique_provider_t(nd::shape_t<Rank> the_shape, nd::buffer_t<ValueType>&& buffer)
    : the_shape(the_shape)
    , strides(make_strides_row_major(the_shape))
    , buffer(std::move(buffer))
    {
        if (the_shape.volume() != unique_provider_t::buffer.size())
        {
            throw std::logic_error("shape and buffer sizes do not match");
        }
    }

    const ValueType& operator()(const index_t<Rank>& index) const { return buffer.operator[](strides.compute_offset(index)); }
    /* */ ValueType& operator()(const index_t<Rank>& index)       { return buffer.operator[](strides.compute_offset(index)); }
    template<typename... Args> const ValueType& operator()(Args... args) const { return operator()(make_index(args...)); }
    template<typename... Args> /* */ ValueType& operator()(Args... args)       { return operator()(make_index(args...)); }

    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }
    const ValueType* data() const { return buffer.data(); }
    ValueType* data() { return buffer.data(); }

    auto shared() const & { return shared_provider_t(the_shape, std::make_shared<buffer_t<ValueType>>(buffer.begin(), buffer.end())); }
    auto shared()      && { return shared_provider_t(the_shape, std::make_shared<buffer_t<ValueType>>(std::move(buffer))); }

    template<std::size_t R> auto reshape(shape_t<R> new_shape) const & { return unique_provider_t<R, ValueType>(new_shape, buffer_t<ValueType>(buffer.begin(), buffer.end())); }
    template<std::size_t R> auto reshape(shape_t<R> new_shape)      && { return unique_provider_t<R, ValueType>(new_shape, std::move(buffer)); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
    memory_strides_t<Rank> strides;
    buffer_t<ValueType> buffer;
};




//=============================================================================
template<std::size_t Rank, typename ValueType>
class nd::uniform_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t rank = Rank;

    //=========================================================================
    uniform_provider_t(shape_t<Rank> the_shape, ValueType the_value) : the_shape(the_shape), the_value(the_value) {}
    const ValueType& operator()(const index_t<Rank>&) const { return the_value; }
    auto shape() const { return the_shape; }
    auto size() const { return the_shape.volume(); }
    template<std::size_t NewRank> auto reshape(shape_t<NewRank> new_shape) const { return uniform_provider_t<NewRank, ValueType>(new_shape, the_value); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
    ValueType the_value;
};




//=============================================================================
template<typename ValueType>
class nd::buffer_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    ~buffer_t() { delete [] memory; }
    buffer_t() {}
    buffer_t(const buffer_t& other) = delete;
    buffer_t& operator=(const buffer_t& other) = delete;

    buffer_t(buffer_t&& other)
    {
        memory = other.memory;
        count = other.count;
        other.memory = nullptr;
        other.count = 0;
    }

    buffer_t(std::size_t count, ValueType value=ValueType())
    : count(count)
    , memory(new ValueType[count])
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            memory[n] = value;
        }
    }

    template<class IteratorType>
    buffer_t(IteratorType first, IteratorType last)
    : count(std::distance(first, last)), memory(new ValueType[count])
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            memory[n] = *first++;
        }
    }

    buffer_t& operator=(buffer_t&& other)
    {
        delete [] memory;
        memory = other.memory;
        count = other.count;

        other.memory = nullptr;
        other.count = 0;
        return *this;
    }

    bool operator==(const buffer_t& other) const
    {
        return count == other.count
        && all_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) == std::get<1>(t); });
    }

    bool operator!=(const buffer_t& other) const
    {
        return count != other.count
        || any_of(zip(*this, other), [] (const auto& t) { return std::get<0>(t) != std::get<1>(t); });
    }

    bool empty() const { return count == 0; }
    std::size_t size() const { return count; }

    const ValueType* data() const { return memory; }
    const ValueType* begin() const { return memory; }
    const ValueType* end() const { return memory + count; }
    const ValueType& operator[](std::size_t offset) const { return memory[offset]; }
    const ValueType& at(std::size_t offset) const
    {
        if (offset >= count)
        {
            throw std::out_of_range("buffer_t index out of range");
        }
        return memory[offset];
    }

    ValueType* data() { return memory; }
    ValueType* begin() { return memory; }
    ValueType* end() { return memory + count; }
    ValueType& operator[](std::size_t offset) { return memory[offset]; }
    ValueType& at(std::size_t offset)
    {
        if (offset >= count)
        {
            throw std::out_of_range("buffer_t index out of range");
        }
        return memory[offset];
    }

private:
    //=========================================================================
    std::size_t count = 0;
    ValueType* memory = nullptr;
};




//=============================================================================
// Provider factories
//=============================================================================




template<typename ValueType, std::size_t Rank>
auto nd::make_uniform_provider(ValueType value, shape_t<Rank> shape)
{
    return uniform_provider_t<Rank, ValueType>(shape, value);
}

template<typename ValueType, typename... Args>
auto nd::make_uniform_provider(ValueType value, Args... args)
{
    return make_uniform_provider(value, make_shape(args...));
}

template<typename ValueType, std::size_t Rank>
auto nd::make_shared_provider(shape_t<Rank> shape)
{
    auto buffer = std::make_shared<buffer_t<ValueType>>(shape.volume());
    return shared_provider_t<Rank, ValueType>(shape, buffer);
}

template<typename ValueType, typename... Args>
auto nd::make_shared_provider(Args... args)
{
    return make_shared_provider<ValueType>(make_shape(args...));
}

template<typename ValueType, std::size_t Rank>
auto nd::make_unique_provider(shape_t<Rank> shape)
{
    auto buffer = buffer_t<ValueType>(shape.volume());
    return unique_provider_t<Rank, ValueType>(shape, std::move(buffer));
}

template<typename ValueType, typename... Args>
auto nd::make_unique_provider(Args... args)
{
    return make_unique_provider<ValueType>(make_shape(args...));
}

template<typename Provider>
auto nd::evaluate_as_unique(Provider&& source_provider)
{
    using value_type = typename std::remove_reference_t<Provider>::value_type;
    auto target_shape = source_provider.shape();
    auto target_accessor = make_access_pattern(target_shape);
    auto target_provider = make_unique_provider<value_type>(target_shape);

    for (auto index : target_accessor)
    {
        target_provider(index) = source_provider(index);
    }
    return target_provider;
}

template<typename Provider>
auto nd::evaluate_as_shared(Provider&& provider)
{
    return evaluate_as_unique(std::forward<Provider>(provider)).shared();
}




//=============================================================================
// Array factories
//=============================================================================




/**
 * @brief      Make an array from the given provider.
 *
 * @param      provider  The provider
 *
 * @tparam     Provider  The type of the provider
 *
 * @return     The array
 */
template<typename Provider>
auto nd::make_array(Provider&& provider)
{
    return array_t<Provider>(std::forward<Provider>(provider));
}




/**
 * @brief      Make an array from the given index -> value mapping and shape.
 *
 * @param[in]  mapping  The mapping
 * @param[in]  shape    The shape
 *
 * @tparam     Mapping  The type of the index -> value mapping
 * @tparam     Rank     The rank of the array
 *
 * @return     The array
 *
 * @note       The array uses the `basic_provider_t`, whose `operator()` is a
 *             const method, but which preserves the reference type of the
 *             mapping's return value. That is, if `mapping` returns a const
 *             reference then so does the returned array when indexed. The
 *             returned array has no `data` or `reshape` methods, and whose.
 */
template<typename Mapping, std::size_t Rank>
auto nd::make_array(Mapping mapping, shape_t<Rank> shape)
{
    return make_array(basic_provider_t<Mapping, Rank>(mapping, shape));
}




/**
 * @brief      Make a shared (immutable, copyable, memory-backed) array with the
 *             given shape, initialized to the default-constructed ValueType.
 *
 * @param[in]  shape      The shape
 *
 * @tparam     ValueType  The value type of the array
 * @tparam     Rank       The rank of the array
 *
 * @return     The array
 */
template<typename ValueType, std::size_t Rank>
auto nd::make_shared_array(shape_t<Rank> shape)
{
    return make_array(make_shared_provider<ValueType>(shape));
}

template<typename ValueType, typename... Args>
auto nd::make_shared_array(Args... args)
{
    return make_array(make_shared_provider<ValueType>(args...));
}




/**
 * @brief      Make a unique (mutable, non-copyable, memory-backed) array with
 *             the given shape.
 *
 * @param[in]  shape      The shape
 *
 * @tparam     ValueType  The value type of the array
 * @tparam     Rank       The rank of the array
 *
 * @return     The array
 */
template<typename ValueType, std::size_t Rank>
auto nd::make_unique_array(shape_t<Rank> shape)
{
    return make_array(make_unique_provider<ValueType>(shape));
}

template<typename ValueType, typename... Args>
auto nd::make_unique_array(Args... args)
{
    return make_array(make_unique_provider<ValueType>(args...));
}




/**
 * @brief      Return an index-array of the given shape, mapping the index (i,
 *             j, ...) to itself.
 *
 * @param[in]  shape  The shape
 *
 * @tparam     Rank   The rank of the array
 *
 * @return     The array
 */
template<std::size_t Rank>
auto nd::index_array(shape_t<Rank> shape)
{
    auto mapping = [shape] (auto&& index) { return index; };
    return make_array(basic_provider_t<decltype(mapping), Rank>(mapping, shape));
}

template<typename... Args>
auto nd::index_array(Args... args)
{
    return index_array(make_shape(args...));
}




/**
 * @brief      Zip a sequence identically-shaped arrays together
 *
 * @param      arrays      The arrays
 *
 * @tparam     ArrayTypes  The types of the arrays
 *
 * @return     An array which returns tuples taken from the underlying arrays
 */
template<typename... ArrayTypes>
auto nd::zip_arrays(ArrayTypes... arrays)
{
    constexpr std::size_t Ranks[] = {ArrayTypes::rank...};
    shape_t<Ranks[0]> shapes[] = {arrays.shape()...};

    if (std::adjacent_find(std::begin(shapes), std::end(shapes), std::not_equal_to<>()) != std::end(shapes))
    {
        throw std::logic_error("cannot zip arrays with different shapes");
    }
    auto mapping = [arrays...] (auto&& index)
    {
        return std::make_tuple(arrays(index)...);
    };
    return make_array(mapping, shapes[0]);
}




/**
 * @brief      Return an array that is the cartesian product of the argument
 *             arrays, A(i, j, k) == make_tuple(a(i), b(j), c(k))
 *
 * @param[in]  arrays      A sequence of 1d arrays
 *
 * @tparam     ArrayTypes  The types of the argument arrays
 *
 * @return     The array
 */
template<typename... ArrayTypes>
auto nd::cartesian_product(ArrayTypes... arrays)
{
    shape_t<sizeof...(ArrayTypes)> shape = {arrays.size()...};

    auto mapping = [arrays...] (auto&& index)
    {
        return detail::zip_apply_tuple(std::forward_as_tuple(arrays...), index.as_tuple());
    };
    return make_array(mapping, shape);
}




/**
 * @brief      Return an array of zeros with the given shape
 *
 * @param[in]  args       shape arguments
 *
 * @tparam     ValueType  Defaults to int; use e.g. zeros<double>(...) for other
 *                        types
 * @tparam     Args       Argument types (should be a positive integral type)
 *
 * @return     An array of zeros, only requiring storage for a single element
 */
template<typename ValueType, typename... Args>
auto nd::zeros(Args... args)
{
    return make_array(nd::make_uniform_provider(ValueType(0), args...));
}




/**
 * @brief      Return an array of ones with the given shape
 *
 * @param[in]  args       shape arguments
 *
 * @tparam     ValueType  Defaults to int; use e.g. ones<double>(...) for other
 *                        types
 * @tparam     Args       Argument types (should be a positive integral type)
 *
 * @return     An array of ones, only requiring storage for a single element.
 */
template<typename ValueType, typename... Args>
auto nd::ones(Args... args)
{
    return make_array(nd::make_uniform_provider(ValueType(1), args...));
}




/**
 * @brief      Try to promote the argument to an array of the given shape
 *
 * @param      arg    The argument
 * @param[in]  shape  The shape
 *
 * @tparam     Arg    The argument type
 * @tparam     Rank   The rank of the shape to promote the argument to
 *
 * @return     An array with the given shape
 */
template<typename Arg, std::size_t Rank>
auto nd::promote(Arg arg, nd::shape_t<Rank> shape)
{
    if constexpr (std::is_arithmetic<Arg>::value)
    {
        return make_array(make_uniform_provider(arg, shape));
    }
    else
    {
        return arg;
    }
}




//=============================================================================
// Operator factories
//=============================================================================




/**
 * @brief      Return an operator that attempts to reshape its argument array to
 *             the given shape.
 *
 * @param[in]  new_shape  The new shape
 *
 * @tparam     Rank       The rank of the argument array
 *
 * @return     The operator
 */
template<std::size_t Rank>
auto nd::reshape(shape_t<Rank> new_shape)
{
    return [new_shape] (auto&& array)
    {
        const auto& provider = array.get_provider();

        if (new_shape.volume() != provider.size())
        {
            throw std::logic_error("cannot reshape array to a different size");
        }
        return make_array(provider.reshape(new_shape));
    };
}

template<typename... Args>
auto nd::reshape(Args... args)
{
    return reshape(make_shape(args...));
}




/**
 * @brief      Return an operator that, applied to any array will yield a
 *             shared, memory-backed version of that array.
 *
 * @return     The operator
 */
auto nd::to_shared()
{
    return [] (auto&& array)
    {
        return make_array(evaluate_as_unique(array.get_provider()));
    };
}




/**
 * @brief      Return an operator that, applied to any array will yield a
 *             unique, memory-backed version of that array.
 *
 * @return     The operator
 */
auto nd::to_unique()
{
    return [] (auto&& array)
    {
        return make_array(evaluate_as_shared(array.get_provider()));
    };
}




/**
 * @brief      Return an operator that turns an array into a bounds-checking
 *             array.
 *
 * @return     The array
 */
auto nd::bounds_check()
{
    return [] (auto&& array)
    {
        auto mapping = [array] (auto&& index)
        {
            if (! array.shape().contains(index))
            {
                throw std::out_of_range("index out-of-range");
            }
            return array(index);
        };
        return make_array(mapping, array.shape());
    };
}




/**
 * @brief      Return an operator that sums the elements of an array.
 *
 * @return     The operator
 *
 * @note       The return type is the same as the array value type, except if
 *             it's bool - in which case the return type is unsigned long.
 */
auto nd::sum()
{
    return [] (auto&& array)
    {
        using value_type = nd::value_type_of<decltype(array)>;
        using is_boolean = std::is_same<value_type, bool>;
        using result_type = std::conditional_t<is_boolean::value, unsigned long, value_type>;

        auto result = result_type();

        for (const auto& i : array.indexes())
        {
            result += array(i);
        }
        return result;
    };
}




/**
 * @brief      Return a reduce operator that returns true if all of its
 *             argument array's elements evaluate to true.
 *
 * @return     The operator
 */
auto nd::all()
{
    return [] (auto&& array)
    {
        for (const auto& i : array.indexes()) if (! array(i)) return false;
        return true;
    };
}




/**
 * @brief      Return a reduce operator that returns true if any of its
 *             argument array's elements evaluate to true.
 *
 * @return     The operator
 */
auto nd::any()
{
    return [] (auto&& array)
    {
        for (const auto& i : array.indexes()) if (array(i)) return true;
        return false;
    };
}




/**
 * @brief      Return an operator that shifts an array along an axis
 *
 * @param[in]  delta  The amount to shift by
 *
 * @return     The operator
 * 
 * @example    B = A | shift_by(-2).along_axis(1); // B(i, j) == A(i, j + 2)
 */
auto nd::shift_by(int delta)
{
    return shifter_t(0, delta);
}




/**
 * @brief      Return an operator that freezes one index its argument array,
 *             reducing its rank by 1.
 *
 * @param[in]  axis_to_freeze  The axis to freeze
 *
 * @return     The operator
 */
auto nd::freeze_axis(std::size_t axis_to_freeze)
{
    return axis_freezer_t<1>(make_index(axis_to_freeze));
}




/**
 * @brief      Return a reducer operator, which can apply the given operator
 *             along a given axis
 *
 * @param      reduction     The reduction
 *
 * @tparam     OperatorType  The type of function object to be applied along an
 *                           axis
 *
 * @return     The operator
 */
template<typename OperatorType>
auto nd::collect(OperatorType reduction)
{
    return axis_reducer_t<OperatorType>(0, std::forward<OperatorType>(reduction));
}




/**
 * @brief      Return an operator that concats the given array onto another.
 *
 * @param      array_to_concat  The array to concatenate
 *
 * @tparam     ArrayType        The type of the array to concatenate
 *
 * @return     The operator
 *
 * @note       The returned operator will fail to compile if applied to arrays
 *             of a different rank than the array to concatenate. It will throw
 *             a logic_error if the array shapes are incompatible.
 */
template<typename ArrayType>
auto nd::concat(ArrayType array_to_concat)
{
    return concatenator_t<ArrayType>(0, std::forward<ArrayType>(array_to_concat));
}




template<typename ArrayType>
auto nd::read_indexes(ArrayType array_of_indexes)
{
    return [array_of_indexes] (auto array_to_index)
    {
        auto mapping = [array_of_indexes, array_to_index] (auto&& index)
        {
            return array_to_index(array_of_indexes(index));
        };
        return make_array(mapping, array_of_indexes.shape());
    };
}




/**
 * @brief      Return an operator that selects a subset of an array.
 *
 * @param[in]  region_to_select  The region to select
 *
 * @tparam     Rank              Rank of both the source and target arrays
 *
 * @return     The operator
 */
template<std::size_t Rank>
auto nd::select(access_pattern_t<Rank> region_to_select)
{
    return selector_t<Rank>(region_to_select);
}

auto nd::select_axis(std::size_t axis_to_select)
{
    return axis_selector_t(axis_to_select, 0, 0, false);
}




/**
 * @brief      Replace a subset of an array with the contents of another.
 *
 * @param[in]  region_to_replace  The region to replace
 * @param      replacement_array  The replacement array
 *
 * @tparam     Rank               Rank of both the array to patch and the
 *                                replacement array
 * @tparam     ArrayType          The type of the replacement array
 *
 * @return     A function returning arrays which map their indexes to the
 *             replacement_array, if those indexes are in the region to replace
 */
template<std::size_t Rank, typename ArrayType>
auto nd::replace(access_pattern_t<Rank> region_to_replace, ArrayType replacement_array)
{
    return replacer_t<Rank, ArrayType>(region_to_replace, replacement_array);
}




/**
 * @brief      Return a select operator starting at the given index.
 *
 * @param[in]  starting_index  The starting index
 *
 * @tparam     Rank            The rank of the array to operate on
 *
 * @return     The operator
 */
template<std::size_t Rank>
auto nd::select_from(index_t<Rank> starting_index)
{
    return selector_t<Rank>().from(starting_index);
}

template<typename... Args>
auto nd::select_from(Args... args)
{
    return select_from(make_index(args...));
}




/**
 * @brief      Return a replace operator starting at the begin index.
 *
 * @param[in]  starting_index  The starting index
 *
 * @tparam     Rank            The rank of the array to operate on
 *
 * @return     The operator
 */
template<std::size_t Rank>
auto nd::replace_from(index_t<Rank> starting_index)
{
    auto zeros = make_array(make_uniform_provider(0, make_uniform_shape<Rank>(1)));
    return replacer_t<Rank, decltype(zeros)>({}, zeros).from(starting_index);
}

template<typename... Args>
auto nd::replace_from(Args... args)
{
    return replace_from(make_index(args...));
}




/**
 * @brief      Reads an index from an array.
 *
 * @param[in]  index_to_read  The index to read
 *
 * @tparam     Rank           The rank of the array that can be read from
 *
 * @return     The operator
 */
template<std::size_t Rank>
auto nd::read_index(index_t<Rank> index_to_read)
{
    return [index_to_read] (auto&& array)
    {
        return array(index_to_read);
    };
}

template<typename... Args>
auto nd::read_index(Args... args)
{
    return read_index(make_index(args...));
}




/**
 * @brief      Return an operator that transforms the values of an array using
 *             the given function object.
 *
 * @param      function  The function
 *
 * @tparam     Function  The type of the function object
 *
 * @return     The operator
 */
template<typename Function>
auto nd::transform(Function function)
{
    return [function] (auto array)
    {
        auto mapping = [array, function] (auto&& index) { return function(array(index)); };
        return make_array(mapping, array.shape());
    };
}




/**
 * @brief      Return a function that operates on two arrays, given a function
 *             that operates on their value types.
 *
 * @param      function  The function
 *
 * @tparam     Function  The function type
 *
 * @return     The operator
 */
template<typename Function>
auto nd::binary_op(Function function)
{
    return [function] (auto A, auto B)
    {
        if (A.shape() != B.shape())
        {
            throw std::logic_error("binary operation applied to arrays of different shapes");
        }
        auto mapping = [function, A, B] (auto&& index)
        {
            return function(A(index), B(index));
        };
        return make_array(mapping, A.shape());
    };
}




//=============================================================================
// More array factories, which must be defined after the operator factories
//=============================================================================




/**
 * @brief      Return a 1d array of containing the indexes where the given array
 *             evaluates to true
 *
 * @param      array      The array
 *
 * @tparam     ArrayType  The type of the argument array
 *
 * @return     An immutable, memory-backed 1d array of index_t<rank>, where rank
 *             is the rank of the argument array
 */
template<typename ArrayType>
auto nd::where(ArrayType array)
{
    auto bool_array = array | transform([] (auto x) { return bool(x); });
    auto index_list = make_unique_array<index_t<rank(bool_array)>>(bool_array | sum());

    std::size_t n = 0;

    for (auto index : bool_array.indexes())
    {
        if (bool_array(index))
        {
            index_list(n++) = index;
        }
    }
    return index_list.shared();
}




//=============================================================================
// The array class itself
//=============================================================================




/**
 * @brief      The actual array class template
 *
 * @tparam     Rank      The array dimensionality
 * @tparam     Provider  Type defining the index space and mapping from indexes
 *                       to values
 */
template<typename Provider>
class nd::array_t
{
public:

    using provider_type = Provider;
    using value_type = typename Provider::value_type;
    static constexpr std::size_t rank = Provider::rank;

    //=========================================================================
    array_t(Provider&& provider) : provider(std::move(provider)) {}

    // indexing functions
    //=========================================================================
    template<typename... Args> decltype(auto) operator()(Args... args) const { return provider(make_index(args...)); }
    template<typename... Args> decltype(auto) operator()(Args... args)       { return provider(make_index(args...)); }
    decltype(auto) operator()(const index_t<rank>& index) const { return provider(index); }
    decltype(auto) operator()(const index_t<rank>& index)       { return provider(index); }
    decltype(auto) data() const { return provider.data(); }
    decltype(auto) data()       { return provider.data(); }

    // query functions and operator support
    //=========================================================================
    auto shape() const { return provider.shape(); }
    auto shape(std::size_t axis) const { return provider.shape()[axis]; }
    auto size() const { return provider.size(); }
    const Provider& get_provider() const { return provider; }
    auto indexes() const { return make_access_pattern(provider.shape()); }
    template<typename Function> auto operator|(Function&& fn) const & { return fn(*this); }
    template<typename Function> auto operator|(Function&& fn)      && { return fn(std::move(*this)); }

    // methods converting this to a memory-backed array
    //=========================================================================
    auto unique() const { return make_array(evaluate_as_unique(provider)); }
    auto shared() const { return make_array(evaluate_as_shared(provider)); }

    // arithmetic operators
    //=========================================================================
    template<typename T> auto operator+(T&& A) const { return bin_op(std::forward<T>(A), std::plus<>()); }
    template<typename T> auto operator-(T&& A) const { return bin_op(std::forward<T>(A), std::minus<>()); }
    template<typename T> auto operator*(T&& A) const { return bin_op(std::forward<T>(A), std::multiplies<>()); }
    template<typename T> auto operator/(T&& A) const { return bin_op(std::forward<T>(A), std::divides<>()); }
    template<typename T> auto operator&&(T&& A) const { return bin_op(std::forward<T>(A), std::logical_and<>()); }
    template<typename T> auto operator||(T&& A) const { return bin_op(std::forward<T>(A), std::logical_or<>()); }
    template<typename T> auto operator==(T&& A) const { return bin_op(std::forward<T>(A), std::equal_to<>()); }
    template<typename T> auto operator!=(T&& A) const { return bin_op(std::forward<T>(A), std::not_equal_to<>()); }
    template<typename T> auto operator<=(T&& A) const { return bin_op(std::forward<T>(A), std::less_equal<>()); }
    template<typename T> auto operator>=(T&& A) const { return bin_op(std::forward<T>(A), std::greater_equal<>()); }
    template<typename T> auto operator<(T&& A) const { return bin_op(std::forward<T>(A), std::less<>()); }
    template<typename T> auto operator>(T&& A) const { return bin_op(std::forward<T>(A), std::greater<>()); }
    template<typename T> auto operator-() const { return transform(std::negate<>()); }
    template<typename T> auto operator!() const { return transform(std::logical_not<>()); }

private:
    //=========================================================================
    template<typename OtherType, typename Function>
    auto bin_op(OtherType&& other, Function&& function) const
    {
        auto F = binary_op(std::forward<Function>(function));
        auto B = promote(std::forward<OtherType>(other), shape());
        return F(*this, std::move(B));
    }
    Provider provider;
};




//=============================================================================
// Helper functions
//=============================================================================




template<typename Function, typename Tuple, std::size_t... Is>
auto nd::detail::transform_tuple_impl(Function&& fn, const Tuple& t, std::index_sequence<Is...>)
{
    return std::make_tuple(fn(std::get<Is>(t))...);
}

template<typename Function, typename Tuple>
auto nd::detail::transform_tuple(Function&& fn, const Tuple& t)
{
    return transform_tuple_impl(std::forward<Function>(fn), t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

template<typename FunctionTuple, typename ArgumentTuple, std::size_t... Is>
auto nd::detail::zip_apply_tuple_impl(FunctionTuple&& fn, ArgumentTuple&& args, std::index_sequence<Is...>)
{
    return std::make_tuple(std::get<Is>(fn)(std::get<Is>(args))...);
}

template<typename FunctionTuple, typename ArgumentTuple>
auto nd::detail::zip_apply_tuple(FunctionTuple&& fn, ArgumentTuple&& args)
{
    return zip_apply_tuple_impl(
        std::forward<FunctionTuple>(fn),
        std::forward<ArgumentTuple>(args),
        std::make_index_sequence<std::tuple_size<ArgumentTuple>::value>());
}

template<typename ResultSequence, typename SourceSequence, typename IndexContainer>
auto nd::detail::read_elements(const SourceSequence& source, IndexContainer indexes)
{
    auto target_n = std::size_t(0);
    auto result = ResultSequence();

    for (std::size_t n = 0; n < source.size(); ++n)
    {
        if (std::find(std::begin(indexes), std::end(indexes), n) != std::end(indexes))
        {
            result[target_n++] = source[n];
        }
    }
    return result;
}

template<typename ResultSequence, typename SourceSequence, typename IndexContainer, typename Sequence>
auto nd::detail::insert_elements(const SourceSequence& source, IndexContainer indexes, Sequence values)
{
    static_assert(indexes.size() == values.size());

    auto source1_n = std::size_t(0);
    auto source2_n = std::size_t(0);
    auto result = ResultSequence();

    for (std::size_t n = 0; n < result.size(); ++n)
    {
        if (std::find(std::begin(indexes), std::end(indexes), n) == std::end(indexes))
        {
            result[n] = source[source1_n++];
        }
        else
        {
            result[n] = values[source2_n++];
        }
    }
    return result;
}

template<typename ResultSequence, typename SourceSequence, typename IndexContainer>
auto nd::detail::remove_elements(const SourceSequence& source, IndexContainer indexes)
{
    auto target_n = std::size_t(0);
    auto result = ResultSequence();

    for (std::size_t n = 0; n < source.size(); ++n)
    {
        if (std::find(std::begin(indexes), std::end(indexes), n) == std::end(indexes))
        {
            result[target_n++] = source[n];
        }
    }
    return result;
}
