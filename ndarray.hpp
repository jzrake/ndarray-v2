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




//=============================================================================
namespace nd
{


    // array support structs
    //=========================================================================
    template<std::size_t Rank, typename ValueType, typename DerivedType> class short_sequence_t;
    template<std::size_t Rank>                                           class shape_t;
    template<std::size_t Rank>                                           class index_t;
    template<std::size_t Rank>                                           class jumps_t;
    template<std::size_t Rank>                                           class memory_strides_t;
    template<std::size_t Rank>                                           class access_pattern_t;
    template<typename ValueType, std::size_t Rank>                       class basic_sequence_t;
    template<typename ValueType>                                         class buffer_t;
    template<typename Provider>                                          class array_t;


    // array and access pattern factory functions
    //=========================================================================
    template<typename... Args>                            auto make_shape(Args... args);
    template<typename... Args>                            auto make_index(Args... args);
    template<typename... Args>                            auto make_jumps(Args... args);
    template<std::size_t Rank, typename Arg>              auto make_uniform_shape(Arg arg);
    template<std::size_t Rank, typename Arg>              auto make_uniform_index(Arg arg);
    template<std::size_t Rank, typename Arg>              auto make_uniform_jumps(Arg arg);
    template<std::size_t Rank>                            auto make_strides_row_major(shape_t<Rank> shape);
    template<std::size_t Rank>                            auto make_access_pattern(shape_t<Rank> shape);
    template<typename... Args>                            auto make_access_pattern(Args... args);
    template<std::size_t NumPartitions, std::size_t Rank> auto partition_shape(shape_t<Rank> shape);


    // provider types
    //=========================================================================
    template<typename Function,  std::size_t Rank> class basic_provider_t;
    template<typename ValueType, std::size_t Rank> class shared_provider_t;
    template<typename ValueType, std::size_t Rank> class unique_provider_t;


    // provider factory functions
    //=========================================================================
    template<typename ValueType, std::size_t Rank>     auto make_shared_provider(shape_t<Rank> shape);
    template<typename ValueType, typename... Args>     auto make_shared_provider(Args... args);
    template<typename ValueType, std::size_t Rank>     auto make_unique_provider(shape_t<Rank> shape);
    template<typename ValueType, typename... Args>     auto make_unique_provider(Args... args);
    template<typename Provider>                        auto evaluate_as_shared(Provider&&);
    template<typename Provider>                        auto evaluate_as_unique(Provider&&);


    // array factory functions
    //=========================================================================
    inline                                             auto arange(int count);
    inline                                             auto arange(int start, int final, int step=1);
    inline                                             auto linspace(double x0, double x1, std::size_t count);
    inline                                             auto divvy(std::size_t num_groups);
    template<typename Provider>                        auto make_array(Provider&&);
    template<typename Mapping, std::size_t Rank>       auto make_array(Mapping mapping, shape_t<Rank> shape);
    template<typename ContainerType>                   auto make_array_from(const ContainerType& container);
    template<typename ValueType, std::size_t Rank>     auto make_shared_array(shape_t<Rank> shape);
    template<typename ValueType, typename... Args>     auto make_shared_array(Args... args);
    template<typename ValueType, std::size_t Rank>     auto make_unique_array(shape_t<Rank> shape);
    template<typename ValueType, typename... Args>     auto make_unique_array(Args... args);
    template<std::size_t Rank>                         auto index_array(shape_t<Rank> shape);
    template<typename... Args>                         auto index_array(Args... args);
    template<typename... ArrayTypes>                   auto zip_arrays(ArrayTypes... arrays);
    template<typename ArrayType>                       auto unzip_array(ArrayType array);
    template<typename... ArrayTypes>                   auto cartesian_product(ArrayTypes... arrays);
    template<typename... ArrayTypes>                   auto meshgrid(ArrayTypes... arrays);
    template<typename ValueType=int, typename... Args> auto zeros(Args... args);
    template<typename ValueType=int, typename... Args> auto ones(Args... args);
    template<typename ValueType, std::size_t Rank>     auto promote(ValueType, shape_t<Rank>);


    // basic array operators
    //=========================================================================
    inline                       auto to_shared();
    inline                       auto to_unique();
    inline                       auto bounds_check();
    inline                       auto sum();
    inline                       auto all();
    inline                       auto any();
    inline                       auto min();
    inline                       auto max();
    template<typename ArrayType> auto min(ArrayType&& array);
    template<typename ArrayType> auto max(ArrayType&& array);
    template<typename Function>  auto map(Function function);
    template<typename Function>  auto apply(Function function);
    template<typename ArrayType> auto where(ArrayType array);
    template<typename Function>  auto binary_op(Function function);


    // extended operator support structs
    //=========================================================================
    /**/                                           class axis_selector_t;
    /**/                                           class axis_shifter_t;
    template<std::size_t RankDifference>           class axis_freezer_t;
    template<typename ArrayType>                   class axis_reducer_t;
    template<typename ArrayType>                   class concatenator_t;
    template<std::size_t Rank, typename ArrayType> class replacer_t;
    template<std::size_t Rank>                     class selector_t;


    // extended operators
    //=========================================================================
    inline                                         auto shift_by(int delta);
    inline                                         auto freeze_axis(std::size_t axis_to_freeze);
    inline                                         auto select_axis(std::size_t axis_to_select);
    template<std::size_t Rank>                     auto select_from(index_t<Rank> starting_index);
    template<typename... Args>                     auto select_from(Args... args);
    template<std::size_t Rank>                     auto select(access_pattern_t<Rank>);
    template<typename OperatorType>                auto collect(OperatorType reduction);
    template<typename ArrayType>                   auto concat(ArrayType array_to_concat);
    template<std::size_t Rank, typename ArrayType> auto replace(access_pattern_t<Rank>, ArrayType);
    template<std::size_t Rank>                     auto replace_from(index_t<Rank> starting_index);
    template<typename... Args>                     auto replace_from(Args... args);
    template<std::size_t Rank>                     auto reshape(shape_t<Rank> shape);
    template<typename... Args>                     auto reshape(Args... args);
    template<std::size_t Rank>                     auto read_index(index_t<Rank>);
    template<typename... Args>                     auto read_index(Args... args);
    template<typename ArrayType>                   auto read_indexes(ArrayType array_of_indexes);


    // to_string overloads
    //=========================================================================
    template<std::size_t Rank> auto to_string(const index_t<Rank>& index);
    template<std::size_t Rank> auto to_string(const shape_t<Rank>& index);
    template<std::size_t Rank> auto to_string(const access_pattern_t<Rank>& region);


    // convenience typedef's
    //=========================================================================
    template<typename ValueType, std::size_t Rank> using shared_array = array_t<shared_provider_t<ValueType, Rank>>;
    template<typename ValueType, std::size_t Rank> using unique_array = array_t<unique_provider_t<ValueType, Rank>>;
    template<typename ArrayType> using value_type_of = typename std::remove_reference_t<ArrayType>::value_type;


    // helper functions
    //=========================================================================
    namespace detail
    {
        template<std::size_t Index, typename ArrayType>
        auto get_through(const ArrayType& array);

        template<typename Function, typename Tuple, std::size_t... Is>
        auto transform_tuple_impl(Function&& fn, const Tuple& t, std::index_sequence<Is...>);

        template<typename Function, typename Tuple>
        auto transform_tuple(Function&& fn, const Tuple& t);

        template<typename FunctionTuple, typename ArgumentTuple, std::size_t... Is>
        auto zip_apply_tuple_impl(FunctionTuple&& fn, ArgumentTuple&& args, std::index_sequence<Is...>);

        template<typename FunctionTuple, typename ArgumentTuple>
        auto zip_apply_tuple(FunctionTuple&& fn, ArgumentTuple&& args);

        template<typename ArrayType, std::size_t... Is>
        auto unzip_array_impl(ArrayType array, std::index_sequence<Is...>);

        template<typename ResultSequence, typename SourceSequence, typename IndexContainer, typename Sequence>
        auto insert_elements(const SourceSequence& source, IndexContainer indexes, Sequence values);

        template<typename ResultSequence, typename SourceSequence, typename IndexContainer>
        auto remove_elements(const SourceSequence& source, IndexContainer indexes);

        template <typename... Ts> using void_t = void;

        template <typename T, typename = void>
        struct has_typedef_is_ndarray : std::false_type {};

        template <typename T>
        struct has_typedef_is_ndarray<T, void_t<typename T::is_ndarray>> : std::true_type {};
    }
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

        for (std::size_t n = 0; n < Rank; ++n)
            result.memory[n] = arg;

        return result;
    }

    template<typename Range>
    static DerivedType from_range(Range&& rng)
    {
        if (rng.size() != Rank)
            throw std::logic_error("sequence constructed from initializer list of wrong size");

        DerivedType result;
        std::size_t n = 0;

        for (auto a : rng)
            result[n++] = a;

        return result;
    }

    short_sequence_t()
    {
        for (std::size_t n = 0; n < Rank; ++n)
            memory[n] = ValueType();
    }

    short_sequence_t(std::initializer_list<ValueType> args)
    {
        if (args.size() != Rank)
            throw std::logic_error("sequence constructed from initializer list of wrong size");

        std::size_t n = 0;

        for (auto a : args)
            memory[n++] = a;
    }

    bool operator==(const DerivedType& b) const { for (std::size_t i = 0; i < Rank; ++i) { if (memory[i] != b[i]) return false; } return true; }
    bool operator!=(const DerivedType& b) const { for (std::size_t i = 0; i < Rank; ++i) { if (memory[i] != b[i]) return true; } return false; }
    constexpr std::size_t size() const { return Rank; }
    const ValueType* data() const { return memory; }
    const ValueType* begin() const { return memory; }
    const ValueType* end() const { return memory + Rank; }
    const ValueType& operator[](std::size_t n) const { return memory[n]; }
    ValueType* data() { return memory; }
    ValueType* begin() { return memory; }
    ValueType* end() { return memory + Rank; }
    ValueType& operator[](std::size_t n) { return memory[n]; }

    template<typename Function>
    auto transform(Function&& fn) const
    {
        DerivedType result;

        for (std::size_t i = 0; i < Rank; ++i)
            result[i] = fn(memory[i]);

        return result;
    }

private:
    //=========================================================================
    ValueType memory[Rank];
};




//=============================================================================
template<typename ValueType, std::size_t Size>
class nd::basic_sequence_t : public nd::short_sequence_t<Size, ValueType, basic_sequence_t<ValueType, Size>>
{
};




//=============================================================================
template<std::size_t Rank>
class nd::shape_t : public nd::short_sequence_t<Rank, std::size_t, shape_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, shape_t<Rank>>::short_sequence_t;

    std::size_t volume() const
    {
        return std::accumulate(this->begin(), this->end(), 1, std::multiplies<>());
    }

    bool contains(const index_t<Rank>& index) const
    {
        for (std::size_t i = 0; i < Rank; ++i)
            if (this->operator[](i) <= index[i])
                return false;
        return true;
    }

    template<typename... Args>
    bool contains(Args... args) const
    {
        return contains(make_index(args...));
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

    template <size_t... Is>
    auto as_tuple(std::index_sequence<Is...>) const { return std::make_tuple(this->operator[](Is)...); }
    auto as_tuple() const { return as_tuple(std::make_index_sequence<Rank>()); }

    bool operator< (const index_t<Rank>& b) const { for (std::size_t i = 0; i < Rank; ++i) { if (! (this->operator[](i) <  b[i])) return false; } return true; }
    bool operator> (const index_t<Rank>& b) const { for (std::size_t i = 0; i < Rank; ++i) { if (! (this->operator[](i) >  b[i])) return false; } return true; }
    bool operator<=(const index_t<Rank>& b) const { for (std::size_t i = 0; i < Rank; ++i) { if (! (this->operator[](i) <= b[i])) return false; } return true; }
    bool operator>=(const index_t<Rank>& b) const { for (std::size_t i = 0; i < Rank; ++i) { if (! (this->operator[](i) >= b[i])) return false; } return true; }
};




//=============================================================================
template<std::size_t Rank>
class nd::jumps_t : public nd::short_sequence_t<Rank, long, jumps_t<Rank>>
{
public:
    using short_sequence_t<Rank, long, jumps_t<Rank>>::short_sequence_t;
};




//=============================================================================
template<std::size_t Rank>
class nd::memory_strides_t : public nd::short_sequence_t<Rank, std::size_t, memory_strides_t<Rank>>
{
public:
    using short_sequence_t<Rank, std::size_t, memory_strides_t<Rank>>::short_sequence_t;

    std::size_t compute_offset(const index_t<Rank>& index) const
    {
        std::size_t result = 0;
   
        for (std::size_t i = 0; i < Rank; ++i)
        {
            result += index[i] * this->operator[](i);
        }
        return result;
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

    constexpr std::size_t rank() const { return Rank; }




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
        auto zero = make_uniform_index<Rank>(0);
        auto t1 = map_index(zero);
        auto t2 = map_index(shape().last_index());

        return (t1 >= zero && t1 <= parent_shape.last_index() &&
                t2 >= zero && t2 <= parent_shape.last_index());
    }




    //=========================================================================
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




/**
 * @brief      Return a sequence of access patterns that cover a shape by
 *             partitioning it on its first axis.
 *
 * @param[in]  shape          The shape to partition
 *
 * @tparam     NumPartitions  The number of partitions
 * @tparam     Rank           The shape rank
 *
 * @return     A sequence of access patterns
 */
template<std::size_t NumPartitions, std::size_t Rank>
auto nd::partition_shape(shape_t<Rank> shape)
{
    constexpr std::size_t distributed_axis = 0;
    auto result = basic_sequence_t<access_pattern_t<Rank>, NumPartitions>();

    for (std::size_t n = 0; n < NumPartitions; ++n)
    {
        auto p = make_access_pattern(shape);
        p.start[distributed_axis] = (n + 0) * shape[distributed_axis] / NumPartitions;
        p.final[distributed_axis] = (n + 1) * shape[distributed_axis] / NumPartitions;
        result[n] = p;
    }
    return result;
}




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
        if (axis_to_select >= array.rank())
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
class nd::axis_shifter_t
{
public:

    //=========================================================================
    axis_shifter_t(std::size_t axis_to_shift, int delta) : axis_to_shift(axis_to_shift), delta(delta) {}

    template<typename ArrayType>
    auto operator()(ArrayType&& array) const
    {
        if (axis_to_shift >= array.rank())
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
        return axis_shifter_t(new_axis_to_shift, delta);
    }

private:
    //=========================================================================
    std::size_t axis_to_shift;
    int delta;
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
        for (auto a : axes_to_freeze)
            if (a >= array.rank())
                throw std::logic_error("cannot freeze axis greater than or equal to array rank");

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
        if (axis_to_reduce >= array.rank())
        {
            throw std::logic_error("cannot reduce axis greater than or equal to array rank");
        }
        constexpr std::size_t R = ArrayType::array_rank;

        auto mapping = [the_operator=the_operator, axis_to_reduce=axis_to_reduce, array] (auto&& index)
        {
            auto axes_to_freeze = range_index<R>().remove_elements(make_index(axis_to_reduce));
            auto freezer = axis_freezer_t<R - 1>(axes_to_freeze).at_index(index);
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
    template<std::size_t Rank>
    static index_t<Rank> range_index()
    {
        auto result = index_t<Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = n;

        return result;
    };

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
        if (axis_to_extend >= array.rank())
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
template<typename Function, std::size_t Rank>
class nd::basic_provider_t
{
public:

    using value_type = std::invoke_result_t<Function, index_t<Rank>>;
    static constexpr std::size_t provider_rank = Rank;

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
template<typename ValueType, std::size_t Rank>
class nd::shared_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t provider_rank = Rank;

    //=========================================================================
    shared_provider_t() {}
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
    template<std::size_t R> auto reshape(shape_t<R> new_shape) const { return shared_provider_t<ValueType, R>(new_shape, buffer); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
    memory_strides_t<Rank> strides;
    std::shared_ptr<buffer_t<ValueType>> buffer;
};




//=============================================================================
template<typename ValueType, std::size_t Rank>
class nd::unique_provider_t
{
public:

    using value_type = ValueType;
    static constexpr std::size_t provider_rank = Rank;

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

    template<std::size_t R> auto reshape(shape_t<R> new_shape) const & { return unique_provider_t<ValueType, R>(new_shape, buffer_t<ValueType>(buffer.begin(), buffer.end())); }
    template<std::size_t R> auto reshape(shape_t<R> new_shape)      && { return unique_provider_t<ValueType, R>(new_shape, std::move(buffer)); }

private:
    //=========================================================================
    shape_t<Rank> the_shape;
    memory_strides_t<Rank> strides;
    buffer_t<ValueType> buffer;
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
auto nd::make_shared_provider(shape_t<Rank> shape)
{
    auto buffer = std::make_shared<buffer_t<ValueType>>(shape.volume());
    return shared_provider_t<ValueType, Rank>(shape, buffer);
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
    return unique_provider_t<ValueType, Rank>(shape, std::move(buffer));
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
 * @brief      Create a 1-dimensional array from an iterable container
 *
 * @param[in]  container      The container to source values from
 *
 * @tparam     ContainerType  The type of the container, std::vector, std::list,
 *                            std::array, etc.
 *
 * @return     The array
 */
template<typename ContainerType>
auto nd::make_array_from(const ContainerType& container)
{
    auto shape = make_shape(container.size());
    auto result = make_unique_array<typename ContainerType::value_type>(shape);
    auto n = std::size_t(0);

    for (auto source_value : container)
    {
        result(n++) = source_value;
    }
    return std::move(result).shared();
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
    auto mapping = [] (auto&& index) { return index; };
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
    constexpr std::size_t Ranks[] = {ArrayTypes::array_rank...};
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
 * @brief      Turn an array of tuples into a tuple of arrays
 *
 * @param[in]  array      The array of tuples
 *
 * @tparam     ArrayType  The type of the argument array
 *
 * @return     The tuple of arrays
 */
template<typename ArrayType>
auto nd::unzip_array(ArrayType array)
{
    auto Is = std::make_index_sequence<std::tuple_size<typename ArrayType::value_type>::value>();
    return detail::unzip_array_impl(array, Is);
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
 * @brief      Return a tuple of N-dimensional arrays, given N 1d arrays
 *
 * @param[in]  arrays      The arrays
 *
 * @tparam     ArrayTypes  The types of the input arrays
 *
 * @return     The tuple of arrays
 */
template<typename... ArrayTypes>
auto nd::meshgrid(ArrayTypes... arrays)
{
    return unzip_array(cartesian_product(arrays...));
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
    return make_array([] (auto) { return ValueType(0); }, make_shape(std::size_t(args)...));
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
    return make_array([] (auto) { return ValueType(1); }, make_shape(std::size_t(args)...));
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
    if constexpr (detail::has_typedef_is_ndarray<Arg>::value)
    {
        return arg;
    }
    else
    {
        return make_array([arg] (auto) { return arg; }, shape);
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
        return make_array(evaluate_as_shared(array.get_provider()));
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
        return make_array(evaluate_as_unique(array.get_provider()));
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
        using value_type = value_type_of<decltype(array)>;
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
 * @brief      Return an operator that gets the minimum value of an array.
 *
 * @return     The operator
 */
auto nd::min()
{
    return [] (auto&& array) { return min(std::forward<decltype(array)>(array)); };
}




/**
 * @brief      Return an operator that gets the maximum value of an array.
 *
 * @return     The operator
 */
auto nd::max()
{
    return [] (auto&& array) { return max(std::forward<decltype(array)>(array)); };
}




/**
 * @brief      Return the minimum value of an array.
 *
 * @param      array      The array
 *
 * @tparam     ArrayType  The type of the array
 *
 * @return     The minimum value
 */
template<typename ArrayType>
auto nd::min(ArrayType&& array)
{
    auto result = value_type_of<ArrayType>();
    auto first = true;

    for (const auto& i : array.indexes())
    {
        if (first || array(i) < result)
        {
            result = array(i);
        }
        first = false;
    }
    return result;
}




/**
 * @brief      Return the maximum value of an array.
 *
 * @param      array      The array
 *
 * @tparam     ArrayType  The type of the array
 *
 * @return     The maximum value
 */
template<typename ArrayType>
auto nd::max(ArrayType&& array)
{
    auto result = value_type_of<ArrayType>();
    auto first = true;

    for (const auto& i : array.indexes())
    {
        if (first || array(i) > result)
        {
            result = array(i);
        }
        first = false;
    }
    return result;
}




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
    auto bool_array = array | map([] (auto x) { return bool(x); });
    auto index_list = make_unique_array<index_t<bool_array.rank()>>(bool_array | sum());

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
    return axis_shifter_t(0, delta);
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
    return axis_reducer_t<OperatorType>(0, reduction);
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
    return concatenator_t<ArrayType>(0, array_to_concat);
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
 * @brief      Return an operator that generates an array B by indexing into a
 *             source array A. B has the value type of A, but the shape of the
 *             index array I. The value type of I must be index_t<A.rank()>.
 *
 * @param[in]  array_of_indexes  An N-dimensional array of M-dimensional indexes
 *
 * @tparam     ArrayType         The type of the index array
 *
 * @return     The operator
 */
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
    auto zeros = make_array([] (auto) { return 0; }, make_uniform_shape<Rank>(1));
    return replacer_t<Rank, decltype(zeros)>({}, zeros).from(starting_index);
}

template<typename... Args>
auto nd::replace_from(Args... args)
{
    return replace_from(make_index(args...));
}




/**
 * @brief      Return an operator that maps the values of an array using the
 *             given function object.
 *
 * @param      function  The function
 *
 * @tparam     Function  The type of the function object
 *
 * @return     The operator
 *
 * @note       This is the N-dimensional version of the transform operator
 */
template<typename Function>
auto nd::map(Function function)
{
    return [function] (auto array)
    {
        auto mapping = [array, function] (auto&& index) { return function(array(index)); };
        return make_array(mapping, array.shape());
    };
}




/**
 * @brief      Return an operator that calls std::apply(fn, arg) for each arg in
 *             the operand array. The value type of that array must be some type
 *             of std::tuple.
 *
 * @param[in]  fn        The function
 *
 * @tparam     Function  The type of the function object
 *
 * @return     The operator
 */
template<typename Function>
auto nd::apply(Function fn)
{
    return [fn] (auto array)
    {
        return array | nd::map([fn] (auto args) { return std::apply(fn, args); });
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



    //=========================================================================
    using provider_type = Provider;
    using value_type = typename Provider::value_type;
    using is_ndarray = std::true_type;
    static constexpr std::size_t array_rank = Provider::provider_rank;




    // iterator
    //=========================================================================
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = typename Provider::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++() { ++current; return *this; }
        bool operator==(const iterator& other) const { return current == other.current; }
        bool operator!=(const iterator& other) const { return current != other.current; }
        decltype(auto) operator*() const { return array(*current); }

        array_t array;
        typename access_pattern_t<array_rank>::iterator current;
    };




    //=========================================================================
    array_t() {}
    array_t(Provider&& provider) : provider(std::move(provider)) {}




    // indexing functions
    //=========================================================================
    template<typename... Args> decltype(auto) operator()(Args... args) const { return provider(make_index(args...)); }
    template<typename... Args> decltype(auto) operator()(Args... args)       { return provider(make_index(args...)); }
    decltype(auto) operator()(const index_t<array_rank>& index) const { return provider(index); }
    decltype(auto) operator()(const index_t<array_rank>& index)       { return provider(index); }
    decltype(auto) data()     const    { return provider.data(); }
    decltype(auto) data()              { return provider.data(); }
    decltype(auto) begin()    const    { return iterator {*this, indexes().begin()}; }
    decltype(auto) end()      const    { return iterator {*this, indexes().end()}; }
    constexpr std::size_t rank() const { return array_rank; }




    // query functions and operator support
    //=========================================================================
    auto shape() const { return provider.shape(); }
    auto shape(std::size_t axis) const { return provider.shape()[axis]; }
    auto size() const { return provider.size(); }
    auto indexes() const { return make_access_pattern(provider.shape()); }
    const Provider& get_provider() const { return provider; }
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
    auto operator-() const { return *this | map(std::negate<>()); }
    auto operator!() const { return *this | map(std::logical_not<>()); }




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
// non-template operators
//=============================================================================




/**
 * @brief      Return a 1d array [0 .. count - 1]
 *
 * @param[in]  count  The number of elements
 *
 * @return     The array, not requiring any storage
 */
auto nd::arange(int count)
{
    return make_array([] (auto index) { return index[0]; }, nd::make_shape(count));
}




/**
 * @brief      Return a 1d array [start, start + skip .. count - 1]
 *
 * @param[in]  start      The starting element
 * @param[in]  final      The final element (one past the end)
 * @param[in]  step       The step size
 *
 * @return     The array, not requiring any storage
 */
auto nd::arange(int start, int final, int step)
{
    if (step == 0 || final / step - start / step < 0)
    {
        throw std::invalid_argument("nd::range");
    }
    return make_array([=] (auto index)
    {
        return start + index[0] * step;
    }, nd::make_shape(final / step - start / step));
}




/**
 * @brief      Return a 1d array of equally spaced values
 *
 * @param[in]  x0         The starting value
 * @param[in]  x1         The final value (inclusive)
 * @param[in]  count      The number of elements in the return array
 *
 * @tparam     ValueType  The value type
 *
 * @return     The array
 */
auto nd::linspace(double x0, double x1, std::size_t count)
{
    auto mapping = [x0, x1, count] (auto index)
    {
        return x0 + (x1 - x0) * index[0] / (count - 1);
    };
    return make_array(mapping, make_shape(count));
}




/**
 * @brief      Return an operator that breaks up a 1d array into a 1d ragged
 *             array of size num_groups, whose elements are arrays with
 *             equitable size and whose disjoint union equals the original
 *             array.
 *
 * @param[in]  num_groups  The number groups to divvy up on
 *
 * @return     The operator
 * @note       This function is useful for parallelization tasks.
 */
auto nd::divvy(std::size_t num_groups)
{
    return [num_groups] (auto array)
    {
        static_assert(array.rank() == 1, "can only divvy a 1d array");

        return make_array([num_groups, array] (auto group_index)
        {
            std::size_t start = (group_index[0] + 0) * array.size() / num_groups;
            std::size_t final = (group_index[0] + 1) * array.size() / num_groups;

            return make_array([array, start] (auto element_index)
            {
                return array(start + element_index[0]);
            }, make_shape(final - start));
        }, make_shape(num_groups));
    };
}




//=============================================================================
// to_string overloads
//=============================================================================




template<std::size_t Rank>
auto nd::to_string(const nd::index_t<Rank>& index)
{
    auto result = std::string("[ ");

    for (std::size_t axis = 0; axis < Rank; ++axis)
    {
        result += std::to_string(index[axis]) + " ";
    }
    return result + "]";
}

template<std::size_t Rank>
auto nd::to_string(const nd::shape_t<Rank>& index)
{
    auto result = std::string("< ");

    for (std::size_t axis = 0; axis < Rank; ++axis)
    {
        result += std::to_string(index[axis]) + " ";
    }
    return result + ">";
}

template<std::size_t Rank>
auto nd::to_string(const nd::access_pattern_t<Rank>& region)
{
    return to_string(region.start) + " -> " + to_string(region.final);
}




//=============================================================================
// Helper functions
//=============================================================================




template<std::size_t Index, typename ArrayType>
auto nd::detail::get_through(const ArrayType& array)
{
    return make_array([array] (auto index) { return std::get<Index>(array(index)); }, array.shape());
};

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

template<typename ArrayType, std::size_t... Is>
auto nd::detail::unzip_array_impl(ArrayType array, std::index_sequence<Is...>)
{
    return std::make_tuple(get_through<Is>(array)...);
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
