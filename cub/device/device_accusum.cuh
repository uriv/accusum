/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_device.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_reduce.cuh>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_scan.cuh>

#include <cuda_profiler_api.h>

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

template <int N>
struct Log2RoundDown
{
    enum {VALUE = Log2<N>::VALUE - (PowerOfTwo<N>::VALUE ? 0 : 1) };
};

template <int N>
struct RoundUpToPowerOfTwo
{
    enum {VALUE = 1 << Log2<N>::VALUE };
};

/**
 * \brief Fixed-size vector that is based on statically-addressed items.
 *
 * When the vector is dynamically indexed, it performs a runtime search for the right item.
 *
 */
template<typename T, int N> struct RegVector;

// Base case in a recursive definition
template<typename T> struct RegVector<T,1> {
    enum {
        LENGTH = 1,
        SIZE_BYTES = sizeof(T)
    };
    typedef T Type;
    typedef T BaseType;
    T value;
    __host__ __device__ __forceinline__
    RegVector() {}
    __host__ __device__ __forceinline__
    RegVector(const T init[1]) : value(init[0]) {}
    __host__ __device__ __forceinline__
    RegVector(const T& fill) : value(fill) {}
    __host__ __device__ __forceinline__
    RegVector(const T& first, const T& default_val) : value(first) {}
    __host__ __device__ __forceinline__
    T& operator[](int idx) { assert(idx == 0); return value; }
    __host__ __device__ __forceinline__
    const T& operator[](int idx) const { assert(idx == 0); return value; }
    template<int I> __host__ __device__ __forceinline__
    T& operator[](Int2Type<0> idx) { return value; }
    template<int I> __host__ __device__ __forceinline__
    const T& operator[](Int2Type<0> idx) const { return value; }
};

template<typename T, int N> struct RegVector : public RegVector<T,N-1>
{
    enum {
        LENGTH = N,
        SIZE_BYTES = N * sizeof(T)
    };
    typedef T Type;
    typedef T BaseType;
    T value;
    __host__ __device__ __forceinline__
    RegVector() {}
    __host__ __device__ __forceinline__
    RegVector(const T init[N]) : RegVector<T,N-1>(init), value(init[N-1]) {}
    __host__ __device__ __forceinline__
    RegVector(const T& fill) : RegVector<T,N-1>(fill), value(fill) {}
    __host__ __device__ __forceinline__
    RegVector(const T& first, const T& default_val) : RegVector<T,N-1>(first, default_val), value(default_val) {}

    __host__ __device__ __forceinline__ T& operator[](int idx) {
        if (idx == N-1)
            return value;
        else
            return (*(RegVector<T,N-1>*)this)[idx];
    }
    __host__ __device__ __forceinline__ const T& operator[](int idx) const {
        if (idx == N-1)
            return value;
        else
            return (*(RegVector<T,N-1>*)this)[idx];
    }
    template<int I> __host__ __device__ __forceinline__ T& operator[](Int2Type<I> idx) { return ((RegVector<T,I+1>*)this)->value; }
    template<int I> __host__ __device__ __forceinline__ const T& operator[](Int2Type<I> idx) const { return ((RegVector<T,I+1>*)this)->value; }
};

struct ExtremeFlags
{
    int nan;
//    char inf[2];    //< infinity [+,-]
//    char nan;       //< nan
//    char _pad;
};


template<typename Tdest, typename Tsrc> __host__ __device__ __forceinline__
Tdest reinterpret_bits (const Tsrc& from)
{
    union {
        Tsrc t1;
        Tdest t2;
    } val;
    val.t1 = from;
    return val.t2;
}

/**
 * \brief Accumulates doubles with extended precision into a set of double words.
 *
 * \tparam Expansions           Number of words to store the sum
 *
 *
 * Example:
 *
 * AccumulatorDouble<2> accum(0.0);
 * accum.Add(1e-12);
 * accum.Add(1e+12);
 * accum.Add(1.0);
 * double a = accum[0];
 * double b = accum[1];
 * accum.print();
 *
 */
template<int Expansions>
struct AccumulatorDouble
{
    enum { SIZE = Expansions };

    typedef AccumulatorDouble<Expansions> Type;
    typedef RegVector<double,Expansions> TVec;

    TVec _vec;  //vector of SIZE doubles stored as named registers. _vec[0] is the most significant word.

    __host__ __device__ __forceinline__ AccumulatorDouble() {}                                   //< uninitialized

    __host__ __device__ __forceinline__ AccumulatorDouble(double fill) : _vec(fill) {}          //< fill all words with a value

    __host__ __device__ __forceinline__ AccumulatorDouble(double first, double default_val)    //< sets word 0 to first, and all the others to default_val
    : _vec(first, default_val) {}

    /**
     * \brief Loads values from an array
     */
    __host__ __device__ __forceinline__ void Load(const double vals[SIZE])
    {
        _vec = TVec(vals);
    }

    /**
     * \brief Stores values to an array
     */
    __host__ __device__ __forceinline__ void Store(double vals[SIZE]) const
    {
#pragma unroll
        for (int i = 0; i < SIZE; i++)
        {
            vals[i] = _vec[i];
        }
    }

    /**
     * \brief Adds a double-precision floating point value to the result.
     *
     * Returns the remainder that could not be saved in the accumulator
     */
    __host__ __device__ __forceinline__ double Add(const double &v)
    {
        return add(v);
    }

    /**
     * \brief Adds the value of another accumulator to this one
     */
    __host__ __device__ __forceinline__ void Add(const Type &v)
    {
        add(v);
    }

    /**
     * \brief Serially adds the values of an array to the accumulator
     */
    __host__ __device__ __forceinline__ void Add(const double* arr, int len)
    {
        if (len < 0)
        {
            return;
        }

        for (int i = 0; i < len; i++)
        {
            add(arr[i]);
        }
    }

    /**
     * \brief Serially adds the values of a fixed-size array to the accumulator
     */
    template<int LENGTH>
    __host__ __device__ __forceinline__ void Add(const double* arr)
    {
#pragma unroll
        for (int i = 0; i < LENGTH; i++)
        {
            add(arr[i]);
        }
    }

    /**
     * \brief Adds two accumulators. Makes this class a functor for adding accumulators.
     */
    __host__ __device__ __forceinline__ Type operator()(const Type &a, const Type &b) const
    {
        Type sum = a;
        sum.Add(b);
        return sum;
    }

    /**
     * \brief Returns the value at a given index.
     *
     * Supports runtime and compile-time indexing.
     */
    __host__ __device__ __forceinline__ double& operator[](int i) { return _vec[i]; }
    __host__ __device__ __forceinline__ const double& operator[](int i) const { return _vec[i]; }
    template<int INDEX> __host__ __device__ __forceinline__ double& operator[](Int2Type<INDEX> i) { return _vec[i]; }
    template<int INDEX> __host__ __device__ __forceinline__ const double& operator[](Int2Type<INDEX> i) const { return _vec[i]; }

    /**
     * \brief Removes overlap between words by running them through a new accumulator
     */
    __host__ __device__ __forceinline__ void Normalize()
    {
        Type tmp(_vec[0], 0.);
#pragma unroll
        for (int i = 1; i < SIZE; i++)
        {
            tmp.Add(_vec[i]);
        }
        _vec = tmp._vec;
    }

    /**
     * \brief Prints the values of the accumulation vector to stdout
     */
    __host__ __device__ __forceinline__ void print() const
    {
        double words[SIZE];
        Store(words);
        printf("[ ");
        if (SIZE > 0)
        {
            printf("%g", words[0]);
#if CUB_PTX_ARCH == 0
            printf(" [0x%016llX] ", reinterpret_bits<unsigned long long>(words[0]));
#else
            printf(" [0x%016llX] ", __double_as_longlong(words[0]));
#endif
        }
        for(int i=1; i<SIZE; i++)
        {
            printf(", %g", words[i]);
            printf(" [0x%016llX] ", reinterpret_bits<unsigned long long>(words[i]));
        }
        printf(" ]");
    }

    /**
     * \brief Applies a functor to each element of the accumulation vector
     *
     * \par Snippet
     * The code snippet below illustrates a possible use of the function to print
     * the values.
     * \par
     * \code
     * struct printy {
     *     __host__ __device__ void operator()(double& d)
     *     {
     *         printf("{%g} ", d);
     *     }
     * };
     *... later in the code ...
     * printy op;
     * normalizer.ForEachWord(op);
     *
     * \endcode
     *
     */
    template<typename OP>
    __host__ __device__ __forceinline__ void ForEachWord(OP& op)
    {
        ForEachWord(op, Int2Type<0>());
    }

protected:
    __host__ __device__ __forceinline__
    double twoSum(double a, double b, double& rem)
    {
    #if 0
        return (fabs(a) >= fabs(b) ? quickTwoSum(a,b) : quickTwoSum(b,a));
    #else
        double s, v;
        s = a + b;
        v = s - a;
        rem = (a - (s - v)) + (b - v);
        return s;
    #endif
    }

    __host__ __device__ __forceinline__
    double quickTwoSum(double a, double b, double& rem)
    {
        double s;
        s = a + b;
        rem = b - (s - a);
        return s;
    }

    __host__ __device__ __forceinline__
    RegVector<double, 2> twoSum(double a, double b)
    {
        double s, r;
        s = twoSum(a, b, r);
        return RegVector<double, 2>(s, r);
    }


    __host__ __device__ __forceinline__
    RegVector<double, 2> quickTwoSum(double a, double b)
    {
        double s, r;
        s = quickTwoSum(a, b, r);
        return RegVector<double, 2>(s, r);
    }

    __host__ __device__ __forceinline__ void add(const Type& v)
    {
#pragma unroll
        for (int i = 0; i < SIZE; i++)
        {
            double rem = add(v[i]);
            if (rem != 0.0)
            {
                Normalize();
                add(rem);
            }
        }
    }

    __host__ __device__ __forceinline__ double add(const double& v)
    {
        // TODO: tune with/without break for best performance
        const bool short_circuit = true;
        double rem = v;                     // remainder
#pragma unroll
        for(int i = 0; i < SIZE; i++)
        {
            if (!short_circuit || rem != 0.0)
            {
                _vec[i] = twoSum(_vec[i], rem, rem);   //< add rem and get new remainder
                if (isinf(_vec[i]))
                {
                    rem = 0.0;
                }
            }
        }
        return rem;
    }

    __host__ __device__ __forceinline__ TVec fix_inf(TVec accum)
    {
        if (isinf(accum[0]))
        {
            return TVec(accum[0], 0.0);
        }
        return accum;
    }
};


/**
 * Specialized more efficient version for 2-wide accumulators
 */

template<>
__host__ __device__ __forceinline__ double AccumulatorDouble<2>::add(const double& b)
{
    const RegVector<double, 2>& a = _vec;
    RegVector<double, 2> s, t;
    s = twoSum(a[0], b);
    s[1] += a[1];
    fix_inf(s);
    s = quickTwoSum(s[0], s[1]);
    _vec = s;
    return 0.;
}

template<>
__host__ __device__ __forceinline__ void AccumulatorDouble<2>::add(const AccumulatorDouble<2>& other)
{
    // OPTIMIZE: instead of running fix_inf, raise an +inf/-inf flag if isinf is true. If flag is up, ignore reduction result.
    //    if (other[1] == 0.0)
    //    {
    //        add(other[0]);
    //    }
    const RegVector<double, 2>& a = _vec;
    RegVector<double, 2> b = other._vec;
    RegVector<double, 2> s, t;
    s = fix_inf(twoSum(a[0], b[0]));
    t = twoSum(a[1], b[1]);
    s[1] += t[0];
    s = fix_inf(quickTwoSum(s[0], s[1]));
    s[1] += t[1];
    s = fix_inf(quickTwoSum(s[0], s[1]));
    _vec = s;
}

/*
 * sum operator for accumulators
 */
template<int Expansion>
__host__ __device__ __forceinline__
AccumulatorDouble<Expansion> operator+(const AccumulatorDouble<Expansion>& a, const AccumulatorDouble<Expansion>& b)
{
    AccumulatorDouble<Expansion> sum = a;
    sum.Add(b);
    return sum;
}

/**
 * Computes the binning configuration based on setup arguments at compile time.
 */
template <
    int         BLOCK_DIM_X,
    int         ITEMS_PER_THREAD,
    int         EXPANSIONS,
    int         RADIX_SORT_BITS             = 4,
    int         BLOCK_DIM_Y                  = 1,
    int         BLOCK_DIM_Z                  = 1>
struct AccumulatorBinsMetadata
{
private:
    /* auxiliary computations */
    struct _temp {
        enum {
            _MAX_POW2_ENUM = 1 << (sizeof(int) * 8 - 2),                                                     //< maximum power-of-two enum. enum values must be representable as an int
            _DOUBLE_MANTISSA_BITS   = 52,
            _DOUBLE_EXPONENT_BITS   = 11,
            _BLOCK_THREADS          = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
            _ITEMS_PER_ITERATION    = _BLOCK_THREADS * ITEMS_PER_THREAD,                                    //< number of items reduced in each iteration
            _MAX_OVERFLOW_FREQUENCY = 256,                                                                    //< minimum number of iterations before an overflow could happen
            _EXTENDED_MANTISSA_BITS = _DOUBLE_MANTISSA_BITS * (EXPANSIONS - 1),                              //< extra mantissa bits due to using multiple double words
            _MAX_EXPONENT_BITS_PER_ACCUM = Log2RoundDown<_EXTENDED_MANTISSA_BITS>::VALUE,                     //< maximum number of exponent bits that can be covered by a single accumulator
            _LOG_ITEMS_PER_ACCUM_LOW  = _EXTENDED_MANTISSA_BITS - (1 << _MAX_EXPONENT_BITS_PER_ACCUM),        //< (log2) for max number of exponent bits, the number of items that can be added to an accumulator before it may overflow
            _LOG_ITEMS_PER_ACCUM_MIN  = Log2<_MAX_OVERFLOW_FREQUENCY * _ITEMS_PER_ITERATION>::VALUE,          //< (log2) minimum allowed number of items that can be added to an accumulator before it may overflow
            _EXPONENT_BITS_PER_BIN   = _MAX_EXPONENT_BITS_PER_ACCUM - (_LOG_ITEMS_PER_ACCUM_LOW >= _LOG_ITEMS_PER_ACCUM_MIN ? 0 : 1),  //< each bin covers a range of numbers that have the same lower X exponent bits
            _NUM_BINS                = 1 << (_DOUBLE_EXPONENT_BITS - _EXPONENT_BITS_PER_BIN),                 //< number of bins
            _LOG_BIN_CAPACITY        = _EXTENDED_MANTISSA_BITS - (1 << _EXPONENT_BITS_PER_BIN),               //< (log2) maximum number of items that can be added to a bin before it could overflow
            _BIN_CAPACITY_TRUNC      = 1 << CUB_MIN(_LOG_BIN_CAPACITY, Log2<_MAX_POW2_ENUM>::VALUE),          //< number of items that can be added to a bin, truncated by the max power-of-two enum
            _BIN_SIZE_BYTES          = sizeof(AccumulatorDouble<EXPANSIONS>),                               //< (log2) maximum integer. used to prevent enum overflow
            _SORT_BITS               = _DOUBLE_EXPONENT_BITS - _EXPONENT_BITS_PER_BIN,                        //< number of exponent bits to sort
            _RADIX_BITS              = CUB_MIN(RADIX_SORT_BITS, _SORT_BITS),                                  //< radix bits for sorting
        };
    };
public:
    typedef AccumulatorDouble<EXPANSIONS> BinType;

    enum {
        EXPONENT_BITS_PER_BIN   = _temp::_EXPONENT_BITS_PER_BIN,  //< items with the same lower X exponent bits are accumulated to the same bin
        NUM_BINS                = _temp::_NUM_BINS,               //< number of bins
        BIN_CAPACITY            = _temp::_BIN_CAPACITY_TRUNC,     //< maximum number of items that can be accumulated in a bin before it could overflow
        BIN_SIZE_BYTES          = _temp::_BIN_SIZE_BYTES,         //< size of accumulator (bin) in bytes
        RADIX_BITS              = _temp::_RADIX_BITS,             //< radix bits for sorting

    };
    __host__ __device__
    static void info()
    {
        printf(
            "EXPONENT_BITS_PER_BIN  = %d \n"
            "NUM_BINS               = %d \n"
            "BIN_CAPACITY           = %d \n"
            "BIN_SIZE_BYTES         = %d \n",
            EXPONENT_BITS_PER_BIN,
            NUM_BINS             ,
            BIN_CAPACITY         ,
            BIN_SIZE_BYTES
            );
    }
};

/**
 * DeviceAccurateSum provides operations to accurately sum an array of doubles without round-off error
 */
template <
    int         BLOCK_DIM_X                  = 64,
    int         ITEMS_PER_THREAD            = 2,
    int         EXPANSIONS                   = 2,
    int         RADIX_BITS                   = 4,
    int         BLOCK_DIM_Y                  = 1,
    int         BLOCK_DIM_Z                  = 1
    >
class DeviceAccurateSum
{
public:
    enum {
        BLOCK_THREADS           = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
        DOUBLE_MANTISSA_BITS    = 52,
        DOUBLE_EXPONENT_BITS    = 11,
    };

    typedef AccumulatorBinsMetadata<BLOCK_DIM_X, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, BLOCK_DIM_Y, BLOCK_DIM_Z> Meta;
    typedef NumericTraits<double>::UnsignedBits UnsignedBits;
    typedef int   BinIdT;
    typedef int   FlagT;      // also used for counter
    typedef AccumulatorDouble<EXPANSIONS> Accumulator;

    // A structure that binds bin-id and accumulator, to be used in scan
    struct AccumBinPair
    {
        Accumulator             accum;
        BinIdT                  bin;
        int                     _pad;       //< pad struct to multiple of 8bytes
    };

    static const bool CONFIG_SORT_MEMOIZE = ((CUB_PTX_ARCH >= 350) ? true : false);
    typedef BlockLoad<double*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;
    typedef BlockRadixSort<
        UnsignedBits, BLOCK_THREADS, ITEMS_PER_THREAD, NullType, Meta::RADIX_BITS,
        CONFIG_SORT_MEMOIZE, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeEightByte> BlockRadixSort;
    typedef BlockDiscontinuity<BinIdT, BLOCK_THREADS> BlockDiscontinuity;
    typedef BlockScan<AccumBinPair, BLOCK_THREADS, BLOCK_SCAN_WARP_SCANS> BlockScan;

    /// Shared memory storage layout type for DeviceAccurateSum::SumToBins()
    // The temp space for sort is typically larger than for load, flag, and scan combined
    struct _TempStorage
    {
        union
        {
            typename BlockRadixSort::TempStorage     sort;
            struct {
                typename BlockLoad::TempStorage          load;
                typename BlockDiscontinuity::TempStorage flag;
                typename BlockScan::TempStorage          scan;
            };
        };
        ExtremeFlags extreme_flags;
    };

    _TempStorage &temp_storage;

    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }

//public:
    struct TempStorage : Uninitialized<_TempStorage> {};

    __device__ __forceinline__ DeviceAccurateSum()
    :
        temp_storage(PrivateStorage())
    {}

    __device__ __forceinline__ DeviceAccurateSum(
        TempStorage &temp_storage)
    :
        temp_storage(temp_storage.Alias())
        {}

    /**
     * Defines
     */
    __device__ __forceinline__ void SumToBins(
        double         *d_in,                 //< [in]  input array
        int             num_items,            //< [in]  input array size
        void           *d_accumulators,       //< [out] accumulator bins
        size_t          accumulators_bytes,   //< [in]  size of accumulator bins array in bytes
        void           *d_megabins,
        size_t          megabins_bytes,
        ExtremeFlags    *d_extreme_flags)
    {
        if (num_items == 0)
        {
            return;             //< for warmup
        }

        enum {
            TILE_SIZE             = (BLOCK_THREADS * ITEMS_PER_THREAD),
            EXP_BIT_SORT_BEGIN    = (DOUBLE_MANTISSA_BITS + Meta::EXPONENT_BITS_PER_BIN),
            EXP_BIT_SORT_END      = (DOUBLE_MANTISSA_BITS + DOUBLE_EXPONENT_BITS),
        };

        ///////////////////////////////////////////////////////////////////////
        __shared__ Accumulator bins[Meta::NUM_BINS];
        ///////////////////////////////////////////////////////////////////////
        double items[ITEMS_PER_THREAD];
        ///////////////////////////////////////////////////////////////////////

        if (d_extreme_flags->nan)   //< if nan flag already marked then abort. result is nan.
        {
            return;
        }

        /// INITIALIZE BINS IN SHARED MEMORY
        {
            double* iptr = (double*)bins;
            const int COUNT = Meta::NUM_BINS * Meta::BIN_SIZE_BYTES / sizeof(double);
            #pragma unroll
            for (int i = 0; i < COUNT / BLOCK_THREADS; i++)
            {
                iptr[i * BLOCK_THREADS + threadIdx.x] = 0;
            }
            if (COUNT % BLOCK_THREADS > 0)
            {
                int i = COUNT / BLOCK_THREADS;
                if (i * BLOCK_THREADS + threadIdx.x < COUNT)
                {
                    iptr[i * BLOCK_THREADS + threadIdx.x] = 0;
                }
            }
        }

        if (threadIdx.x == 0)
        {
            temp_storage.extreme_flags.nan = 0;
        }

        /// PROCESS INPUT

        // add offset that depends on block index
        d_accumulators = (void*)((char*)d_accumulators + blockIdx.x * Meta::NUM_BINS * Meta::BIN_SIZE_BYTES);
        d_in += blockIdx.x * TILE_SIZE;

        int items_per_block = num_items / gridDim.x;

        // Loop over tiles
        #pragma unroll 4
        for (int item = 0; item < items_per_block; item += TILE_SIZE)
        {
            // BINS ARE BEING UPDATED
            // ITEMS ARE NOT BEING USED
            // Load tile
            BlockLoad(temp_storage.load).Load(d_in, items);
            __syncthreads();
            // BINS ARE NOT BEING UPDATED
            // ITEMS ARE BEING USED

            // Reduce values and update bins
            bool overwrite_bins = (item == 0);
            ///////////////////////////////////////////////

            /// RADIX SORT BY (SOME) EXPONENT BITS
            UnsignedBits (*cvt_to_ubits)[ITEMS_PER_THREAD] = (UnsignedBits(*)[ITEMS_PER_THREAD])items;
            BlockRadixSort(temp_storage.sort).Sort(*cvt_to_ubits,EXP_BIT_SORT_BEGIN,EXP_BIT_SORT_END);

            /// REDUCE-BY-KEY (SETUP)
            BinIdT          bin_ids     [ITEMS_PER_THREAD];
            FlagT           tail_flags  [ITEMS_PER_THREAD];
            AccumBinPair    zip         [ITEMS_PER_THREAD];
            AccumBinPair    zipout      [ITEMS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                if (isnan(items[i]))
                {
                    temp_storage.extreme_flags.nan = 1;
                }
                bin_ids[i] = bin_id(items[i]);
                zip[i].accum = Accumulator(items[i], 0.0);
                zip[i].bin = bin_ids[i];
            }
            __syncthreads();

            // BINS ARE BEING UPDATED
            // ITEMS ARE NOT BEING USED

            /// REDUCE BY KEY
            BlockScan(temp_storage.scan).InclusiveScan(zip, zipout, ReductionOp());
            BlockDiscontinuity(temp_storage.flag).FlagTails(tail_flags, bin_ids, cub::Inequality());

            /// UPDATE BINS
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                if (tail_flags[i])
                {
                    // this is the reduction result for this bin
                    if (overwrite_bins)
                    {
                        bins[bin_ids[i]] = zipout[i].accum;
                    }
                    else
                    {
                        bins[bin_ids[i]].Add(zipout[i].accum);
                    }
                }
            }

            d_in += gridDim.x * TILE_SIZE;
        }
        __syncthreads();
        // BINS ARE NOT BEING UPDATED

        /// STORE BINS TO GLOBAL MEM
        {
            typedef double StoreUnit;
            StoreUnit* isptr = (StoreUnit*)bins;
            StoreUnit* igptr = (StoreUnit*)d_accumulators;
            const int COUNT = Meta::NUM_BINS * Meta::BIN_SIZE_BYTES / sizeof(StoreUnit);
            #pragma unroll
            for (int i = 0; i < COUNT / BLOCK_THREADS; i++)
            {
                double val = isptr[i * BLOCK_THREADS + threadIdx.x];
                igptr[i * BLOCK_THREADS + threadIdx.x] = (isnan(val) ? 0.0 : val);
//                if (!isnan(val))
//                {
//                    int bin_id = (i * BLOCK_THREADS + threadIdx.x) / EXPANSIONS;
//                    atomicAddToBin_< EXPANSIONS + 1 >(((AccumulatorDouble<EXPANSIONS+1>*)d_megabins)[bin_id], val);
//                }
            }
            if (COUNT % BLOCK_THREADS > 0)
            {
                int i = COUNT / BLOCK_THREADS;
                if (i * BLOCK_THREADS + threadIdx.x < COUNT)
                {
                    double val = isptr[i * BLOCK_THREADS + threadIdx.x];
                    igptr[i * BLOCK_THREADS + threadIdx.x] = (isnan(val) ? 0.0 : val);
                }
            }
        }

        if (threadIdx.x == 0)
        {
            if (temp_storage.extreme_flags.nan)
            {
                d_extreme_flags->nan = 1;
            }
        }
    }

//private:

    struct ReductionOp
    {
        __device__ __forceinline__ AccumBinPair operator()(
            const AccumBinPair   &first,
            const AccumBinPair   &second)
        {
            AccumBinPair retval = second;
            if (first.bin == second.bin)
            {
                retval.accum.Add(first.accum);
            }
            return retval;
        }
    };

    static __device__ __forceinline__ BinIdT bin_id(const double& v)
    {
        enum { EXP_BIT_SORT_BEGIN  = (DOUBLE_MANTISSA_BITS + Meta::EXPONENT_BITS_PER_BIN) };

        // maybe with __double2hiint (double x) ...
        // maybe with frexp ...

        unsigned long long llv;
        llv = __double_as_longlong(abs(v));
        int tmp1 = (int)(llv >> EXP_BIT_SORT_BEGIN);
        return (BinIdT)tmp1;
    }

    __device__ double atomicAdd_(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    template <int BIN_EXPANSIONS>
    __device__ void atomicAddToBin_(AccumulatorDouble<BIN_EXPANSIONS>& bin, double val)
    {
        double a, b, s, r, av, bv, ar, br;

        b = val;
        #pragma unroll
        for (int i = 0; i < EXPANSIONS - 1; i++)
        {
            if (b == 0.0) break;
            a = atomicAdd_(&bin[i], b);    // returns a and stores (a+b) in bin[0]
            s = a + b;                              // recompute s=(a+b)
            bv = s - a;
            av = s - bv;
            br = b - bv;
            ar = a - av;
            r = ar + br;
            b = (isnan(r) ? 0.0 : r);
        }
        if (b != 0.0)
        {
            atomicAdd_(&bin[EXPANSIONS - 1], b);      //< don't compute carry for last word
        }
    }
};

template<
    int GRID_SIZE,
    int BLOCK_THREADS,
    int NUM_BINS,
    int NUM_BIN_COPIES
>
struct DeviceAccurateSumSmemAtomic
{
    __device__ void SumToBins(
        double* d_in,
        int num_items,
        double2* d_bins
        )
    {
        __shared__ double bins[NUM_BIN_COPIES * (2 * NUM_BINS + 1)];
        double vals[1];

        int items_per_block = num_items / GRID_SIZE;

        d_in += items_per_block * blockIdx.x;
        int bin_group = LaneId() % NUM_BIN_COPIES;
        double2* mybins = (double2*)(&bins[bin_group * (2 * NUM_BINS + 1)]);

        // initialize bins in shared memory
        double* pdbl_bins = (double*)bins;
        const int DBL_BINS_LEN = sizeof(bins) / sizeof(double);
        #pragma unroll
        for (int i = threadIdx.x; i < DBL_BINS_LEN; i += BLOCK_THREADS)
        {
            pdbl_bins[i] = 0.0;
        }
        __syncthreads();

        // accumulate in bins
        #pragma unroll 64
        for (int i = threadIdx.x; i < items_per_block; i += BLOCK_THREADS)
        {
            vals[0] = d_in[i];
            int ibin = binid(vals[0]);
            atomicAddToBin(vals[0], ibin, mybins);
        }
        __syncthreads();

        // store bins in global mem
        double* pdbl_d_bins = ((double*)d_bins) + (blockIdx.x * sizeof(bins) / sizeof(double));
        #pragma unroll
        for (int i = threadIdx.x; i < NUM_BINS * 2 * NUM_BIN_COPIES; i += BLOCK_THREADS)
        {
            // place bins with the same id consecutively in output
//            int i_bin = (i / NUM_BIN_COPIES) / 2;
//            int i_word = (i / NUM_BIN_COPIES) % 2;
//            int i_bin_copy = i % NUM_BIN_COPIES;
//            pdbl_d_bins[i] = pdbl_bins[i_word + 2 * i_bin + (2 * NUM_BINS + 1) * i_bin_copy];
            int j = i + (i / (2 * NUM_BINS + 1));       //< skip the padding
            pdbl_d_bins[i] = pdbl_bins[j];
        }
    }       // SumToBins

    __device__ int binid(double d)
    {
        long long ll = __double_as_longlong(d);
        int bin = (int)((ll >> 52) & 0x7ff) / NUM_BINS;
        return bin;
    }

    __device__ double& at(double2& d2, int idx)
    {
        return (idx == 0 ? d2.x : d2.y);
    }

    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    __device__ void atomicAddToBin(double val, int bin, double2* bins)
    {
        double a, b, x, y, av, bv, ar, br;
        b = val;
        a = atomicAdd(&at(bins[bin], 0), b);    // returns a and stores (a+b) in bin[0]
        x = a + b;                              // recompute s=(a+b)
        bv = x - a;
        av = x - bv;
        br = b - bv;
        ar = a - av;
        y = ar + br;

        if (y != 0.0)
        {
            atomicAdd(&at(bins[bin], 1), y);
        }
    }
};

/**
 * \brief Computes an accurate summation of doubles into sets of bins using the sort-reduce method
 *
 * Each thread-block produces one set of bins.
 * Each bin contains the sum of the items in the thread-block's
 *   share of the input that have an exponent in the bin's exponent range.
 *   The bins in each set cover the entire double-precision exponent range.
 */
template <
int         BLOCK_THREADS,
int         ITEMS_PER_THREAD,
int         EXPANSIONS,
int         RADIX_BITS,
int         MIN_CONCURRENT_BLOCKS>
__launch_bounds__ (BLOCK_THREADS, MIN_CONCURRENT_BLOCKS)
__global__ void DeviceAccurateSumKernel(
    double         *d_in,
    int             num_items,
    void           *d_accumulators,
    size_t          accumulators_bytes,
    void           *d_megabins,
    size_t          megabins_bytes,
    ExtremeFlags    *d_extreme_flags)
{
    typedef DeviceAccurateSum<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS> DeviceAccurateSum;
    __shared__ typename DeviceAccurateSum::TempStorage temp_storage;
    DeviceAccurateSum(temp_storage).SumToBins(d_in, num_items, d_accumulators, accumulators_bytes, d_megabins, megabins_bytes, d_extreme_flags);
}

/**
 * \brief Computes an accurate summation of doubles into sets of bins using the smem-atomic method
 *
 * Each thread-block produces a number of bin sets.
 * Each bin contains the sum of the items in the thread-block's
 *   share of the input that have an exponent in the bin's exponent range.
 *   The bins in each set cover the entire double-precision exponent range.
 */
template<
    int GRID_SIZE,
    int BLOCK_THREADS,
    int NUM_CONCURRENT_BLOCKS,
    int NUM_BINS,
    int NUM_BIN_COPIES
>
__launch_bounds__(BLOCK_THREADS, NUM_CONCURRENT_BLOCKS)
__global__ void accuadd_kernel(
    double* d_in,
    int num_items,
    double2* d_bins
    )
{
    typedef DeviceAccurateSumSmemAtomic<GRID_SIZE, BLOCK_THREADS, NUM_BINS, NUM_BIN_COPIES> DeviceAccurateSumSmemAtomic;
    DeviceAccurateSumSmemAtomic().SumToBins(d_in, num_items, d_bins);
}

enum AccurateFPSumAlgorithm {
    ACCUSUM_SORT_REDUCE = 0,        //< sum binning with sort-reduce
    ACCUSUM_SMEM_ATOMIC = 1,        //< sum binning with atomic smem
};

struct DeviceAccurateFPSum
{
    struct DefaultSetup
    {
        enum {
            Method                  = ACCUSUM_SORT_REDUCE,
            WarpsPerBlock           = 4,
            BlocksPerSm             = 24,
            ItemsPerThread          = 3,
            Expansions              = 2,
            RadixBits               = 3,
            MinConcurrentBlocks     = 3,
        };
    };

    template <
        int BLOCK_THREADS,
        int MIN_GRID_SIZE,
        int ITEMS_PER_THREAD,
        int EXPANSIONS,
        int RADIX_BITS,
        int MIN_CONCURRENT_BLOCKS>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t SumSortReduce
    (
        double         *d_in,
        int             num_items,
        double          *d_out,
        void           *d_temp_storage,
        size_t          &temp_storage_bytes,
        cudaStream_t    stream                  = 0)
    {
        typedef AccumulatorBinsMetadata<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS> BinMeta;

        void *d_bin_sets = NULL;
        void *d_megabins = NULL;
        ExtremeFlags *d_extreme_flags = NULL;
        void *h_bin_sets = NULL;
        void *h_megabins = NULL;
        ExtremeFlags *h_extreme_flags = NULL;
        cudaError_t error = cudaSuccess;

        do {
            int device_id, sm_count;
            if (error = CubDebug(cudaGetDevice(&device_id))) break;
            if (error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id))) break;
            int grid_size = CUB_ROUND_UP_NEAREST(CUB_MAX(MIN_GRID_SIZE, CUB_QUOTIENT_CEILING(num_items, BinMeta::BIN_CAPACITY)), sm_count);

            // Temporary storage allocation requirements
            void* allocations[2];
            size_t allocation_sizes[2] =
            {
                grid_size * BinMeta::NUM_BINS * BinMeta::BIN_SIZE_BYTES,      // for the per-block bin sets
                //            sizeof(double) * (EXPANSIONS + 1) * BinMeta::NUM_BINS,     // for mega-bins
                sizeof(ExtremeFlags)                                        // for nan,inf,inf flags
            };

            if (error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            d_bin_sets          = allocations[0];
            d_megabins          = NULL; //allocations[0];
            d_extreme_flags     = (ExtremeFlags*)allocations[1];
            h_bin_sets          = malloc(allocation_sizes[0]);
            h_megabins          = NULL; //malloc(allocation_sizes[0]);
            h_extreme_flags     = (ExtremeFlags*)malloc(allocation_sizes[1]);

            if (h_bin_sets == NULL || h_extreme_flags == NULL)
            {
                error = cudaErrorMemoryAllocation;
                break;
            }

            if (error = CubDebug(cudaMemsetAsync(d_temp_storage, 0, temp_storage_bytes, stream))) break;

            cudaProfilerStart();
            DeviceAccurateSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS>
            <<<grid_size, BLOCK_THREADS, 0, stream>>>(
                d_in,
                num_items,
                d_bin_sets,
                0,//temp_reduce_size,
                d_megabins,
                0,//temp_megabins_size,
                d_extreme_flags);
            cudaProfilerStop();
            if (error = CubDebug(cudaMemcpyAsync(h_bin_sets, d_bin_sets, allocation_sizes[0], cudaMemcpyDeviceToHost, stream))) break;
            //        if (error = CubDebug(cudaMemcpy(h_megabins, d_megabins, allocation_sizes[0], cudaMemcpyDeviceToHost, stream))) break;
            if (error = CubDebug(cudaMemcpyAsync(h_extreme_flags, d_extreme_flags, allocation_sizes[1], cudaMemcpyDeviceToHost, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;
            double result;
            if (h_extreme_flags->nan)
            {
                result = 0.0 / 0.0;  //NaN
            }
            else
            {
                AccumulatorDouble<EXPANSIONS+1> total_sum(0.0);
                for (int i = 0; i < allocation_sizes[0] / sizeof(double); i++)
                {
                    total_sum.Add(((double*)h_bin_sets)[i]);
                }
                total_sum.Normalize();
                result = total_sum[0];
            }

            if (error = CubDebug(cudaMemcpyAsync(d_out, &result, sizeof(double), cudaMemcpyHostToDevice, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;
        } while(0);

        if(h_bin_sets != NULL)          free(h_bin_sets);
        if(h_megabins != NULL)          free(h_megabins);
        if(h_extreme_flags != NULL)     free(h_extreme_flags);

        return error;
    }

    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t SumSmemAtomic
    (
        double         *d_in,
        int             num_items,
        double          *d_out,
        void           *d_temp_storage,
        size_t          &temp_storage_bytes,
        cudaStream_t    stream                  = 0)
    {
        enum
        {
            NUM_SM                  = 12,       //< number of SMs
            NUM_CONCURRENT_BLOCKS   = 3,        //< number of blocks that run on an SM concurrently
            NUM_BLOCK_WAVES         = 8,        //< multiplication factor for number of blocks
            ACCU_GRID_SIZE = NUM_SM * NUM_CONCURRENT_BLOCKS * NUM_BLOCK_WAVES,
            ACCU_TILE_SIZE          = 672,
            NUM_BIN_COPIES_SMEM     = 15,       //< number of set of bins in shared memory
            NUM_BINS                = 64,
            EXPANSIONS              = 2,
        };

        /**
         * NOTE: The parameter NUM_BLOCK_WAVES needs to be tuned.
         * Increasing NUM_BLOCK_WAVES => more efficient utilization of the SM, but more work for CPU
         */
        void* d_bin_sets = NULL;
        void* h_bin_sets = NULL;
        cudaError_t error = cudaSuccess;

        do {

            int device_id, sm_count;
            if (error = CubDebug(cudaGetDevice(&device_id))) break;
            if (error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id))) break;
            if (sm_count != NUM_SM)
            {
                printf("[SumSmemAtomic] Please change NUM_SM (=%d) to %d\n", NUM_SM, sm_count);
                error = cudaErrorInvalidValue;
                break;
            }

            if (num_items % (ACCU_TILE_SIZE * ACCU_GRID_SIZE))
            {
                printf("[SumSmemAtomic] The smem-atomic accurate summation method currently only supports arrays whose size is a multiple of %d\n", (ACCU_TILE_SIZE * ACCU_GRID_SIZE));
                error = cudaErrorInvalidValue;
                break;
            }

            //num_items = CUB_ROUND_UP_NEAREST(num_items, ACCU_TILE_SIZE * ACCU_GRID_SIZE);
            int accu_reduce_size = ACCU_GRID_SIZE * NUM_BIN_COPIES_SMEM * (EXPANSIONS * NUM_BINS + 1) * sizeof(double);

            if (d_temp_storage == NULL)
            {
                temp_storage_bytes = accu_reduce_size;
                return cudaSuccess;
            }



            h_bin_sets = malloc(accu_reduce_size);
            d_bin_sets = d_temp_storage;
            if (h_bin_sets == NULL)
            {
                error = cudaErrorMemoryAllocation;
                break;
            }

            if (error = CubDebug(cudaMemsetAsync(d_temp_storage, 0, accu_reduce_size, stream))) break;

            cudaProfilerStart();
            // Run kernel once to prime caches and check result
            accuadd_kernel<ACCU_GRID_SIZE, ACCU_TILE_SIZE, NUM_CONCURRENT_BLOCKS, NUM_BINS, NUM_BIN_COPIES_SMEM>
            <<<ACCU_GRID_SIZE, ACCU_TILE_SIZE, 0, stream>>>(
                d_in,
                num_items,
                (double2*)d_bin_sets);

            if (error = CubDebug(cudaMemcpyAsync(h_bin_sets, d_bin_sets, accu_reduce_size, cudaMemcpyDeviceToHost, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;
            double result;
            AccumulatorDouble<EXPANSIONS+1> total_sum(0.0);
            for (int i = 0; i < accu_reduce_size / sizeof(double); i++)
            {
                total_sum.Add(((double*)h_bin_sets)[i]);
            }
            total_sum.Normalize();
            result = total_sum[0];
            if (error = CubDebug(cudaMemcpyAsync(d_out, &result, sizeof(double), cudaMemcpyHostToDevice, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;
            cudaProfilerStop();
        } while(0);

        if (h_bin_sets) free(h_bin_sets);
        return error;
    }

    static cudaError_t Sum(
        double         *d_in,
        int             num_items,
        double         *d_out,
        void           *d_temp_storage,
        size_t          &temp_storage_bytes,
        cudaStream_t    stream                  = 0
        )
    {

        enum {
            WARP_SIZE       = 32,
            NUM_SM          = 14,
            BLOCK_THREADS   = WARP_SIZE * DefaultSetup::WarpsPerBlock,
            GRID_SIZE       = DefaultSetup::BlocksPerSm * NUM_SM,
        };
        if (DefaultSetup::Method == (int)ACCUSUM_SORT_REDUCE)
        {
            return SumSortReduce<
                BLOCK_THREADS,
                GRID_SIZE,
                DefaultSetup::ItemsPerThread,
                DefaultSetup::Expansions,
                DefaultSetup::RadixBits,
                DefaultSetup::MinConcurrentBlocks
            >
            (d_in, num_items, d_out, d_temp_storage, temp_storage_bytes, stream);
        }
        else    //ACCUSUM_SMEM_ATOMIC
        {
            return SumSmemAtomic(d_in, num_items, d_out, d_temp_storage, temp_storage_bytes, stream);
        }
    }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
