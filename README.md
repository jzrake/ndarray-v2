# Introduction
_Header-only implementation of an N-dimensional array for modern C++_


## Overview
This library adopts an abstract notion of an array: an array is something that defines an N-dimensional index space (a _shape_), and a mapping from that space to values,

    `array := (shape, index => value)`

The mapped values may be generated however you want - either by looking them up in a memory buffer (as in a conventional array), or by calling a function with the index as a parameter.

You might regard the data structures and algorithms provided here as an N-dimensional counterpart of [C++ ranges](https://en.cppreference.com/w/cpp/ranges), as implemented by e.g. the [range-v3](https://github.com/ericniebler/range-v3) project. Such *functional programming* approaches to writing C++ code are gaining wider attention, and becoming more feasible as the language evolves.


## Implementation
An array is a class template parameterized around a _provider_. Operations applied to the array (such as selecting, replacing, or transforming the underlying values) return an array with a new provider, with a different type. This enables highly optimized and memory-efficient code to be generated by the compiler, but does so at the expense of generating a large number of types. Arrays can be converted to memory-backed arrays at any stage in an algorithm; doing so caches their values and collapses the type hierarchy.

- Uses C++17 features (make sure to set `-std=c++17`)
- Tested (so far) on:
    - clang 10.0.1
    - gcc 7.3.0
    - gcc 8.2.0



## Immutability
Arrays are immutable, meaning that you manipulate them by applying transformations to them, generating new arrays. The new arrays are light-weight objects that incur essentially-zero overhead when passed by value: if the array is memory-backed, it holds a `std::shared_ptr` to an immutable memory buffer, while operated-on arrays only hold function objects and other (also light-weight) arrays. Operating on an array does not (immediately) allocate a new memory buffer, and no calculations are done until the new array is indexed, or converted to a memory-backed array. Such lazy evaluation incurs some compile time overhead in exchange for runtime performace and reduced memory footprint (the compiler sees the whole type hierarchy and can scrunch it down to perform optimizations).

There is one exception to immutability, a `unique_array`, which is memory-backed and read/write, so it enables procedural loading of data into a memory-backed array. The `unique_array` owns its data buffer, and is move-constructible but not copy-constructible (following the semantics of `std::unique_ptr`). After loading data into it, it can be moved to a shared (immutable, copy-constructible) memory-backed array. Mutable arrays are made non-copyable to ensure that you're not accidentally passing around heavyweight objects by value.


## Quick-start examples
Create a 10 x 20 array of zero-initialized ints:
```C++
auto A = nd::zeros(10, 20);
```

Add an array of doubles to it:
```C++
auto B = A + nd::ones<double>(10, 20);
```

Reshape an array:
```C++
auto B = A.reshape(200); // may require B.size() == A.size() (see section on reshaping)
```

Transform an array element-wise:
```C++
auto B = A | nd::map([] (auto x) { return x * x; });
```

Create an array as a subset of another:
```C++
auto B = A | nd::select_from(0, 0).to(10, 10).jumping(2, 2); // B.shape() == {5, 5}
```

Apply the selection to just one axis (optionally from the end)
```C++
auto B = A | nd::select_axis(1).from(3).to(3).from_the_end(); // B.shape(1) == A.shape(1) - 6
```

Shift an array by some amount along an axis:
```C++
auto B = A | nd::shift_by(-2).along_axis(1); // B(i, j) == A(i, j + 2)
```

Create an array by substituting a region with values from another array:
```C++
auto B = A | nd::replace_from(0, 0).to(10, 5).with(nd::zeros(10, 5));
```

Reduce the dimensionality of an array by slicing:
```C++
auto B = A | nd::freeze_axis(0).at_index(2);
```

Take the sum of all elements:
```C++
auto total = A | nd::sum();
```

Or just on axis 1:
```C++
auto B = A | nd::collect(nd::sum()).along_axis(1);
```

Determine whether all corresponding elements of two arrays are equal:
```C++
if ((A == B) | nd::all()) { ... }
```

Determine if any two elements are unequal:
```C++
if ((A != B) | nd::any()) { ... }
```

Obtain a 1d array of indexes where a condition is satisfied:
```C++
auto indexes = nd::where(A != B && B < C);
```

Read the values from an array at those indexes:
```C++
auto values = D | nd::read_indexes(indexes);
```

Concatenate another array:
```C++
auto B = nd::ones(10, 10, 5) | nd::concat(nd::zeros(10, 10, 3)).on_axis(2);
```

Create an array of tuples from arrays of identical shape:
```C++
auto ABC = nd::zip(A, B, C); // ABC(0, 0) is a std::tuple
```

Create a tuple of arrays from an array of tuples:
```C++
auto [A, B, C] = nd::unzip(ABC);
```

Take the cartesian product of a sequence of arrays:
```C++
auto X = nd::cartesian_product(x, y, z); // X(i, j, k) == std::make_tuple(x(i), y(j), z(k))
```

Create N, N-dimensional arrays from N 1d arrays:
```C++
auto [X, Y] = nd::meshgrid(x, y); // X.shape() == Y.shape() == {x.size(), y.size()}
```


## Using the `unique_array`
For most use cases, you should be able to create the array you need by composing operators on it. However, it's sometimes necessary to modify the memory backing procedurally. This is the purpose of unique array (also called transients in other libraries based on immutable data).

There are two ways to create a unique array: from scratch,

```C++
auto A = nd::make_unique_array<double>(10, 20);
```

or from an existing array,

```C++
auto A = nd::ones(10, 20).unique();
```

Now, you can load data into the mutable array procedurally,

```C++
for (auto index : A.indexes())
{
    A(index) = index[0] + index[1];
}
```

Your unique array has data in it, but you can't really do anything with it. Remember, it can't be passed by value anywhere. That includes applying operators to it, since the operators use value (rather than reference) semantics. However, if you move the array to a shared one,

```C++
auto B = A.shared();
```

you now have an immutable, memory-backed array with the same data content as `A` had. But be aware... you did just incur a heavy-weight copy. Indeed, if you check, you'll see that

```C++
B.data() != A.data();
```

If you wanted, you could keep calling `A.shared()` to vend out new copies of its data.

__Note__: only memory-backed arrays have a `data` member function. You'd get a compile error if you were to do `(A | select_from(5, 10)).data()`.

Anyway, in many cases you don't need the unique array to generate more than a single immutable copy. So, you can use the move operator to avoid the copy,

```C++
auto B = std::move(A).shared();
```

Here, ownership of the data buffer is transferred to `B`, leaving `A` in a "valid but useless" state. You could reassign it to another unique array if you wanted to.


## Reshaping arrays
The ability to reshape an array depends on the provider type. Memory-backed arrays can be reshaped to another array of the same total size. A `uniform_array` (returned by the `ones` and `zeros`) can be reshaped arbitrarily. All other arrays cannot be reshaped.


## Writing new operators
Here is an example of how to write a custom operator. As a use-case, let's say you'd like to map an array `A` through a function `f`,
```C++
auto B = A | nd::map(f);
```

but `f` might throw an exception. We'll write an operator called `value_on_exception`, that catches the error and returns a value -1 as a default value:

```C++
auto B = A | nd::map(f) | value_on_exception(-1);
```

Here is the code for the `value_on_exception` operator:

```C++
template<typename ValueType>
auto value_on_exception(ValueType default_value)
{
    return [default_value] (auto array)
    {
        auto mapping = [array] (auto index)
        {
            try {
                return array(index);
            }
            catch (...)
            {
                return default_value;
            }
        };
        return make_array(mapping, array.shape());
    };
}
```

The arguments to `make_array` are a mapping (from N-dimensional indexes to some values), and an N-dimensional shape. In this case, the shape of new array is the same as that of the operand. This construct should free your imagination to cook up some interesting operators. As an exercise, try implementing a `transpose_axes` operation, or a `circular_shift`, or a `laplacian`.


## Multi-threaded execution
Arrays are not just objects for storing and retrieving data; they are types that can encode entire algorithms, which may involve considerable number crunching to evaluate. In general, you'll build your algorithm by composing a sequence of operators, and then evaluate the whole thing to a memory-backed array,
```C++
auto the_algorithm(auto A, auto B)
{
    return ((A | map(sqrt)) + B | some_operator)
    | collect(standard_deviation()).along_axis(1)
    | to_shared();
}
```
This evaluation is _embarressingly parallel_ as a result of the immutability: each worker can execute `array::operator()` over a subset of the index space without worrying about race conditions! If we have a function to partition the index space, we can dispatch separate threads to evaluate each piece of the partition. It would be nice if our algorithm could just as easily be evaluated like this:
```C++
auto the_algorithm(auto A, auto B)
{
    return ((A | map(sqrt)) + B | some_operator)
    | collect(standard_deviation()).along_axis(1)
    | evaluate_on<4>();
}
```

Here is an implementation of an `evaluate_on` operator. It uses the `nd::partition_shape` function to create a sequence of disjoint access patterns which cover the index space.

```C++
template<std::size_t NumThreads>
auto evaluate_on()
{
    return [] (auto array)
    {
        using value_type = typename decltype(array)::value_type;
        auto provider = nd::make_unique_provider<value_type>(array.shape());
        auto evaluate_partial = [&] (auto accessor)
        {
            return [accessor, array, &provider]
            {
                for (auto index : accessor)
                {
                    provider(index) = array(index);
                }
            };
        };
        auto threads = nd::basic_sequence_t<std::thread, NumThreads>();
        auto regions = nd::partition_shape<NumThreads>(array.shape());

        for (auto [n, accessor] : nd::enumerate(regions))
            threads[n] = std::thread(evaluate_partial(accessor));

        for (auto& thread : threads)
            thread.join();

        return nd::make_array(std::move(provider).shared());
    };
}
```

The actual mileage you'll get out of this approach may vary with type of memory access patterns your arrays are using, and what type of calculations are being done. Typically, the more work you do per evaluation of `operator()`, the better. The `partition_shape` function divvies the shape on axis 0, which is appropriate for C-style arrays. If you have written a custom memory-backed provider which instead accesses memory Fortran-style, you should partition the shape on the last axis, since otherwise your threads will contend for cache lines (see [false sharing](https://en.wikipedia.org/wiki/False_sharing)).

Note that reductions are also a parallelizable operation - you could easily adapt this example to write a multi-threaded `reduce_on` operator.
