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

/******************************************************************************
 * Test of DeviceHistogram utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <algorithm>

#include <npp.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_histogram.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------


// Dispatch types
enum Backend
{
    CUB,        // CUB method
    NPP,        // NPP method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


bool                    g_verbose_input     = false;
bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);




//---------------------------------------------------------------------
// Dispatch to NPP histogram
//---------------------------------------------------------------------

/**
 * Dispatch to single-channel 8b NPP histo-even
 */
template <typename CounterT, typename LevelT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchEven(
    Int2Type<1>             num_channels,
    Int2Type<1>             num_active_channels,
    Int2Type<NPP>           dispatch_to,
    int                     timing_timing_iterations,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    unsigned char       *d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
    CounterT            *d_histogram[1],          ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int                 num_levels[1],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              lower_level[1],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT              upper_level[1],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT             row_stride_bytes,                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef unsigned char SampleT;

    cudaError_t error = cudaSuccess;
    NppiSize oSizeROI = {
        num_row_pixels,
        num_rows
    };

    if (d_temp_storage_bytes == NULL)
    {
        int nDeviceBufferSize;
        nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, num_levels[0] ,&nDeviceBufferSize);
        temp_storage_bytes = nDeviceBufferSize;
    }
    else
    {
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            // compute the histogram
            nppiHistogramEven_8u_C1R(
                d_samples,
                row_stride_bytes,
                oSizeROI,
                d_histogram[0],
                num_levels[0],
                lower_level[0],
                upper_level[0],
                (Npp8u*) d_temp_storage);
        }
    }

    return error;
}


/**
 * Dispatch to 3/4 8b NPP histo-even
 */
template <typename CounterT, typename LevelT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchEven(
    Int2Type<4>          num_channels,
    Int2Type<3>   num_active_channels,
    Int2Type<NPP>           dispatch_to,
    int                     timing_timing_iterations,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    unsigned char       *d_samples,               ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
    CounterT            *d_histogram[3],          ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int                 num_levels[3],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              lower_level[3],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT              upper_level[3],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    OffsetT             num_row_pixels,           ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT             num_rows,                 ///< [in] The number of rows in the region of interest
    OffsetT             row_stride_bytes,               ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef unsigned char SampleT;

    cudaError_t error = cudaSuccess;
    NppiSize oSizeROI = {
        num_row_pixels,
        num_rows
    };

    if (d_temp_storage_bytes == NULL)
    {
        int nDeviceBufferSize;
        nppiHistogramEvenGetBufferSize_8u_AC4R(oSizeROI, num_levels ,&nDeviceBufferSize);
        temp_storage_bytes = nDeviceBufferSize;
    }
    else
    {
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            // compute the histogram
            nppiHistogramEven_8u_AC4R(
                d_samples,
                row_stride_bytes,
                oSizeROI,
                d_histogram,
                num_levels,
                lower_level,
                upper_level,
                (Npp8u*) d_temp_storage);
        }
    }

    return error;
}

//---------------------------------------------------------------------
// Dispatch to different DeviceHistogram entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to CUB single histogram-even entrypoint
 */
template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchEven(
    Int2Type<1>             num_channels,
    Int2Type<1>             num_active_channels,
    Int2Type<CUB>           dispatch_to,
    int                     timing_timing_iterations,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
    CounterT            *d_histogram[1],                            ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int                 num_levels[1],                              ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              lower_level[1],                             ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT              upper_level[1],                             ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT             row_stride_bytes,                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<SampleIteratorT>::value_type SampleT;

    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::HistogramEven(
            d_temp_storage,
            temp_storage_bytes,
            (const SampleT *) d_samples,
            d_histogram[0],
            num_levels[0],
            lower_level[0],
            upper_level[0],
            num_row_pixels,
            num_rows,
            row_stride_bytes,
            stream,
            debug_synchronous);
    }
    return error;
}

/**
 * Dispatch to CUB multi histogram-even entrypoint
 */
template <int NUM_ACTIVE_CHANNELS, int NUM_CHANNELS, typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchEven(
    Int2Type<NUM_CHANNELS>          num_channels,
    Int2Type<NUM_ACTIVE_CHANNELS>   num_active_channels,
    Int2Type<CUB>           dispatch_to,
    int                     timing_timing_iterations,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
    CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],          ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int                 num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              lower_level[NUM_ACTIVE_CHANNELS],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT              upper_level[NUM_ACTIVE_CHANNELS],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT             row_stride_bytes,                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<SampleIteratorT>::value_type SampleT;

    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage,
            temp_storage_bytes,
            (const SampleT *) d_samples,
            d_histogram,
            num_levels,
            lower_level,
            upper_level,
            num_row_pixels,
            num_rows,
            row_stride_bytes,
            stream,
            debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to CUB single histogram-range entrypoint
 */
template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchRange(
    Int2Type<1>             num_channels,
    Int2Type<1>             num_active_channels,
    Int2Type<CUB>           dispatch_to,
    int                     timing_timing_iterations,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
    CounterT            *d_histogram[1],                            ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int                 num_levels[1],                              ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              *d_levels[1],                               ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
    OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT             row_stride_bytes,                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<SampleIteratorT>::value_type SampleT;

    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::HistogramRange(
            d_temp_storage,
            temp_storage_bytes,
            (const SampleT *) d_samples,
            d_histogram[0],
            num_levels[0],
            d_levels[0],
            num_row_pixels,
            num_rows,
            row_stride_bytes,
            stream,
            debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to CUB multi histogram-range entrypoint
 */
template <int NUM_ACTIVE_CHANNELS, int NUM_CHANNELS, typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t DispatchRange(
    Int2Type<NUM_CHANNELS>          num_channels,
    Int2Type<NUM_ACTIVE_CHANNELS>   num_active_channels,
    Int2Type<CUB>           dispatch_to,
    int                     timing_timing_iterations,
    size_t                  *d_temp_storage_bytes,
    cudaError_t             *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
    CounterT            *d_histogram[NUM_ACTIVE_CHANNELS],          ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    int                 num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT              *d_levels[NUM_ACTIVE_CHANNELS],             ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
    OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT             row_stride_bytes,                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    typedef typename std::iterator_traits<SampleIteratorT>::value_type SampleT;

    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage,
            temp_storage_bytes,
            (const SampleT *) d_samples,
            d_histogram,
            num_levels,
            d_levels,
            num_row_pixels,
            num_rows,
            row_stride_bytes,
            stream,
            debug_synchronous);
    }
    return error;
}



//---------------------------------------------------------------------
// CUDA nested-parallelism test kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceHistogram
 * /
template <int BINS, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleT, typename SampleIteratorT, typename CounterT, int ALGORITHM>
__global__ void CnpDispatchKernel(
    Int2Type<ALGORITHM> algorithm,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              temp_storage_bytes,
    SampleT             *d_samples,
    SampleIteratorT      d_sample_itr,
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS> d_out_histograms,
    int                 num_samples,
    bool                debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch<BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(algorithm, Int2Type<false>(), timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_out_histograms.array, num_samples, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/ **
 * Dispatch to CDP kernel
 * /
template <int BINS, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleT, typename SampleIteratorT, typename CounterT, int ALGORITHM>
cudaError_t Dispatch(
    Int2Type<ALGORITHM> algorithm,
    Int2Type<true>      use_cdp,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void                *d_temp_storage,
    size_t              &temp_storage_bytes,
    SampleT             *d_samples,
    SampleIteratorT      d_sample_itr,
    CounterT        *d_histograms[NUM_ACTIVE_CHANNELS],
    int                 num_samples,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    // Setup array wrapper for histogram channel output (because we can't pass static arrays as kernel parameters)
    ArrayWrapper<CounterT*, NUM_ACTIVE_CHANNELS> d_histo_wrapper;
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
        d_histo_wrapper.array[CHANNEL] = d_histograms[CHANNEL];

    // Invoke kernel to invoke device-side dispatch
    CnpDispatchKernel<BINS, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, ALGORITHM><<<1,1>>>(algorithm, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_samples, d_sample_itr, d_histo_wrapper, num_samples, debug_synchronous);

    // Copy out temp_storage_bytes
    CubDebugExit(cudaMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, cudaMemcpyDeviceToHost));

    // Copy out error
    cudaError_t retval;
    CubDebugExit(cudaMemcpy(&retval, d_cdp_error, sizeof(cudaError_t) * 1, cudaMemcpyDeviceToHost));
    return retval;
}
*/


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

// Searches for bin given a list of bin-boundary levels
template <typename LevelT>
struct SearchTransform
{
    LevelT          *levels;      // Pointer to levels array
    int             num_levels;   // Number of levels in array

    // Functor for converting samples to bin-ids (num_levels is returned if sample is out of range)
    template <typename SampleT>
    int operator()(SampleT sample)
    {
        int bin = std::upper_bound(levels, levels + num_levels, (LevelT) sample) - levels - 1;
        if (bin < 0)
        {
            // Sample out of range
            return num_levels;
        }
        return bin;
    }
};


// Scales samples to evenly-spaced bins
template <typename LevelT>
struct ScaleTransform
{
    int    num_levels;  // Number of levels in array
    LevelT max;         // Max sample level (exclusive)
    LevelT min;         // Min sample level (inclusive)
    LevelT scale;       // Bin scaling factor

    void Init(
        int    num_levels,  // Number of levels in array
        LevelT max,         // Max sample level (exclusive)
        LevelT min,         // Min sample level (inclusive)
        LevelT scale)       // Bin scaling factor
    {
        this->num_levels = num_levels;
        this->max = max;
        this->min = min;
        this->scale = scale;
    }

    // Functor for converting samples to bin-ids  (num_levels is returned if sample is out of range)
    template <typename SampleT>
    int operator()(SampleT sample)
    {
        if ((sample < min) || (sample >= max))
        {
            // Sample out of range
            return num_levels;
        }

        return (int) ((((LevelT) sample) - min) / scale);
    }
};


/**
 * Generate sample
 */
template <typename T, typename LevelT>
void Sample(T &datum, LevelT max_value, int entropy_reduction)
{
    unsigned int max = (unsigned int) -1;
    unsigned int bits;
    RandomBits(bits, entropy_reduction);
    float fraction = (float(bits) / max);

    datum = (T) (fraction * max_value);
}


/**
 * Initialize histogram problem (and solution)
 */
template <
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        LevelT,
    typename        SampleT,
    typename        CounterT,
    typename        TransformOp,
    typename        OffsetT>
void Initialize(
    LevelT          max_value,
    int             entropy_reduction,
    SampleT         *h_samples,
    int             num_levels[NUM_ACTIVE_CHANNELS],        ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    TransformOp     transform_op[NUM_ACTIVE_CHANNELS],      ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    CounterT        *h_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    OffsetT         num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,                               ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes)                             ///< [in] The number of bytes between starts of consecutive rows in the region of interest
{
    printf("Initializing... "); fflush(stdout);

    // Init bins
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
        for (int bin = 0; bin < num_levels[CHANNEL] - 1; ++bin)
        {
            h_histogram[CHANNEL][bin] = 0;
        }
    }

    // Initialize samples
    if (g_verbose_input) printf("Samples: \n");
    for (OffsetT row = 0; row < num_rows; ++row)
    {
        for (OffsetT pixel = 0; pixel < num_row_pixels; ++pixel)
        {
            if (g_verbose_input) printf("[");
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                // Sample offset
                OffsetT offset = (row * (row_stride_bytes / sizeof(SampleT))) + (pixel * NUM_CHANNELS) + channel;

                // Init sample value
                Sample(h_samples[offset], max_value, entropy_reduction);
                if (g_verbose_input)
                {
                    if (channel > 0) printf(", ");
                    std::cout << CoutCast(h_samples[offset]);
                }

                // Update sample bin
                int bin = transform_op[channel](h_samples[offset]);
                if (g_verbose_input) printf(" (%d)", bin); fflush(stdout);
                if ((bin >= 0) && (bin < num_levels[channel] - 1))
                {
                    // valid bin
                    h_histogram[channel][bin]++;
                }
            }
            if (g_verbose_input) printf("]");
        }
        if (g_verbose_input) printf("\n\n");
    }

    printf("Done\n"); fflush(stdout);
}


/**
 * Test histogram-even
 */
template <
    Backend         BACKEND,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestEven(
    LevelT          max_value,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT          lower_level[NUM_ACTIVE_CHANNELS],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT          upper_level[NUM_ACTIVE_CHANNELS],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    OffsetT         num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes,                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    const char*     type_string)
{
    OffsetT total_samples = num_rows * (row_stride_bytes / sizeof(SampleT));

    printf("\n----------------------------\n%s cub::DeviceHistogramEven %d pixels (%d height, %d width, %d-byte row stride), %d %d-byte %s samples (entropy reduction %d), %d/%d channels, max sample ",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == NPP) ? "NPP" : "CUB",
        num_row_pixels * num_rows, num_rows, num_row_pixels, row_stride_bytes,
        total_samples, (int) sizeof(SampleT), type_string, entropy_reduction,
        NUM_ACTIVE_CHANNELS, NUM_CHANNELS);
    std::cout << CoutCast(max_value) << "\n";
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
        std::cout << "\n\tChannel " << channel << ": " << num_levels[channel] - 1 << " bins [" << lower_level[channel] << ", " << upper_level[channel] << ")\n";
    fflush(stdout);

    // Allocate and initialize host and device data

    SampleT*                    h_samples = new SampleT[total_samples];
    CounterT*                   h_histogram[NUM_ACTIVE_CHANNELS];
    ScaleTransform<LevelT>      transform_op[NUM_ACTIVE_CHANNELS];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int bins = num_levels[channel] - 1;
        h_histogram[channel] = new CounterT[bins];

        transform_op[channel].Init(
            num_levels[channel],
            upper_level[channel],
            lower_level[channel],
            ((upper_level[channel] - lower_level[channel]) / bins));
    }

    Initialize<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        max_value, entropy_reduction, h_samples, num_levels, transform_op, h_histogram, num_row_pixels, num_rows, row_stride_bytes);

    // Allocate and initialize device data

    SampleT*        d_samples = NULL;
    CounterT*       d_histogram[NUM_ACTIVE_CHANNELS];

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples, sizeof(SampleT) * total_samples));
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * total_samples, cudaMemcpyHostToDevice));
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram[channel], sizeof(CounterT) * (num_levels[channel] - 1)));
        CubDebugExit(cudaMemset(d_histogram[channel], 0, sizeof(CounterT) * (num_levels[channel] - 1)));
    }

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    DispatchEven(
        Int2Type<NUM_CHANNELS>(), Int2Type<NUM_ACTIVE_CHANNELS>(), Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride_bytes,
        0, true);

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run warmup/correctness iteration
    DispatchEven(
        Int2Type<NUM_CHANNELS>(), Int2Type<NUM_ACTIVE_CHANNELS>(), Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride_bytes,
        0, true);

    // Flush any stdout/stderr
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    int error = 0;
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int channel_error = CompareDeviceResults(h_histogram[channel], d_histogram[channel], num_levels[channel] - 1, true, g_verbose);
        printf("\tChannel %d %s", channel, channel_error ? "FAIL" : "PASS\n");
        error |= channel_error;
    }

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();

    DispatchEven(
        Int2Type<NUM_CHANNELS>(), Int2Type<NUM_ACTIVE_CHANNELS>(), Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride_bytes,
        0, false);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(total_samples) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(SampleT);
        printf("\t%.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
            avg_millis,
            grate,
            grate * NUM_ACTIVE_CHANNELS / NUM_CHANNELS,
            grate / NUM_CHANNELS,
            gbandwidth);
    }

    printf("\n\n");

    // Cleanup
    if (h_samples) delete[] h_samples;

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        if (h_histogram[channel])
            delete[] h_histogram[channel];

        if (d_histogram[channel])
            CubDebugExit(g_allocator.DeviceFree(d_histogram[channel]));
    }

    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, error);
}





/**
 * Test histogram-range
 */
template <
    Backend         BACKEND,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestRange(
    LevelT          max_value,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT*         levels[NUM_ACTIVE_CHANNELS],                ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    OffsetT         num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes,                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    const char*     type_string)
{
    OffsetT total_samples = num_rows * (row_stride_bytes / sizeof(SampleT));

    printf("\n----------------------------\n%s cub::DeviceHistogramRange %d pixels (%d height, %d width, %d-byte row stride), %d %d-byte %s samples (entropy reduction %d), %d/%d channels, max sample ",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == NPP) ? "NPP" : "CUB",
        num_row_pixels * num_rows, num_rows, num_row_pixels, row_stride_bytes,
        total_samples, (int) sizeof(SampleT), type_string, entropy_reduction,
        NUM_ACTIVE_CHANNELS, NUM_CHANNELS);
    std::cout << CoutCast(max_value) << "\n";
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        printf("Channel %d: %d bins [", channel, num_levels[channel] - 1);
        std::cout << levels[channel][0];
        for (int level = 1; level < num_levels[channel]; ++level)
            std::cout << ", " << levels[channel][level];
        printf("]\n");
    }
    fflush(stdout);

    // Allocate and initialize host and device data
    SampleT*                    h_samples = new SampleT[total_samples];
    CounterT*                   h_histogram[NUM_ACTIVE_CHANNELS];
    SearchTransform<LevelT>     transform_op[NUM_ACTIVE_CHANNELS];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        transform_op[channel].levels = levels[channel];
        transform_op[channel].num_levels = num_levels[channel];

        int bins = num_levels[channel] - 1;
        h_histogram[channel] = new CounterT[bins];
    }

    Initialize<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        max_value, entropy_reduction, h_samples, num_levels, transform_op, h_histogram, num_row_pixels, num_rows, row_stride_bytes);

    // Allocate and initialize device data
    SampleT*        d_samples = NULL;
    LevelT*         d_levels[NUM_ACTIVE_CHANNELS];
    CounterT*       d_histogram[NUM_ACTIVE_CHANNELS];

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples, sizeof(SampleT) * total_samples));
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * total_samples, cudaMemcpyHostToDevice));

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_levels[channel], sizeof(LevelT) * num_levels[channel]));
        CubDebugExit(cudaMemcpy(d_levels[channel], levels[channel],         sizeof(LevelT) * num_levels[channel], cudaMemcpyHostToDevice));

        int bins = num_levels[channel] - 1;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram[channel],  sizeof(CounterT) * bins));
        CubDebugExit(cudaMemset(d_histogram[channel], 0,                        sizeof(CounterT) * bins));
    }

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    DispatchRange(
        Int2Type<NUM_CHANNELS>(), Int2Type<NUM_ACTIVE_CHANNELS>(), Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, d_levels,
        num_row_pixels, num_rows, row_stride_bytes,
        0, true);

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run warmup/correctness iteration
    DispatchRange(
        Int2Type<NUM_CHANNELS>(), Int2Type<NUM_ACTIVE_CHANNELS>(), Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, d_levels,
        num_row_pixels, num_rows, row_stride_bytes,
        0, true);

    // Flush any stdout/stderr
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    int error = 0;
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int channel_error = CompareDeviceResults(h_histogram[channel], d_histogram[channel], num_levels[channel] - 1, true, g_verbose);
        printf("\tChannel %d %s", channel, channel_error ? "FAIL" : "PASS\n");
        fflush(stdout);

        error |= channel_error;
    }

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();

    DispatchRange(
        Int2Type<NUM_CHANNELS>(), Int2Type<NUM_ACTIVE_CHANNELS>(), Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, d_levels,
        num_row_pixels, num_rows, row_stride_bytes,
        0, false);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(total_samples) / avg_millis / 1000.0 / 1000.0;
        float gbandwidth = grate * sizeof(SampleT);
        printf("\t%.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
            avg_millis,
            grate,
            grate * NUM_ACTIVE_CHANNELS / NUM_CHANNELS,
            grate / NUM_CHANNELS,
            gbandwidth);
    }

    printf("\n\n");

    // Cleanup
    if (h_samples) delete[] h_samples;

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        if (h_histogram[channel])
            delete[] h_histogram[channel];

        if (d_histogram[channel])
            CubDebugExit(g_allocator.DeviceFree(d_histogram[channel]));

        if (d_levels[channel])
            CubDebugExit(g_allocator.DeviceFree(d_levels[channel]));
    }

    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, error);
}


/**
 * Test histogram-even
 */
template <
    Backend         BACKEND,
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestEven(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    int             row_stride_bytes,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],
    int             max_levels,
    LevelT          max_value,
    const char*     type_string)
{
    LevelT lower_level[NUM_ACTIVE_CHANNELS];
    LevelT upper_level[NUM_ACTIVE_CHANNELS];

    int max_bins = max_levels - 1;
    LevelT level_increment = max_value / max_bins;

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int num_bins = num_levels[channel] - 1;
        lower_level[channel] = (max_value - (num_bins * level_increment)) / 2;
        upper_level[channel] = (max_value + (num_bins * level_increment)) / 2;
    }

    TestEven<BACKEND, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
        max_value, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes, type_string);
}



/**
 * Test histogram-range
 */
template <
    Backend         BACKEND,
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestRange(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    int             row_stride_bytes,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],
    int             max_levels,
    LevelT          max_value,
    const char*     type_string)
{
    int max_bins = max_levels - 1;
    LevelT level_increment = max_value / max_bins;

    LevelT* levels[NUM_ACTIVE_CHANNELS];
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        levels[channel] = new LevelT[num_levels[channel]];
        for (int level = 0; level < num_levels[channel]; ++level)
            levels[channel][level] = level * level_increment;
    }

    TestRange<BACKEND, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
        max_value, entropy_reduction, num_levels, levels, num_row_pixels, num_rows, row_stride_bytes, type_string);

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
        delete[] levels[channel];

}



/**
 * Test different entrypoints
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    int             row_stride_bytes,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],
    int             max_levels,
    LevelT          max_value,
    const char*     type_string)
{
    TestEven<CUB, SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, type_string);

    TestRange<CUB, SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, type_string);
}


/**
 * Test different number of levels
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    OffsetT         row_stride_bytes,
    int             entropy_reduction,
    LevelT          max_value,
    const char*     type_string)
{
    int num_levels[NUM_ACTIVE_CHANNELS];

    // All the same level
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        num_levels[channel] = 257;
    }
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, num_levels[0], max_value, type_string);

    // All different levels
    num_levels[0] = (sizeof(SampleT) == 1) ? 129 : 1025;
    for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        num_levels[channel] = (num_levels[channel - 1] / 2) + 1;
    }
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, num_levels[0], max_value, type_string);
}



/**
 * Test different entropy-levels
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    OffsetT         row_stride_bytes,
    LevelT          max_value,
    const char*     type_string)
{
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, 0,   max_value, type_string);

    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, -1,  max_value, type_string);

    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, 5,   max_value, type_string);

}


/**
 * Test different row strides
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    LevelT          max_value,
    const char*     type_string)
{
    OffsetT row_stride_bytes = num_row_pixels * NUM_CHANNELS * sizeof(SampleT);

    // No padding
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, max_value, type_string);

    // 13 samples padding
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes + (13 * sizeof(SampleT)), max_value, type_string);
}


/**
 * Test different problem sizes
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    LevelT          max_value,
    const char*     type_string)
{
    // 1080 image
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        OffsetT(1920), OffsetT(1080), max_value, type_string);

    // 720 image
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        OffsetT(1280), OffsetT(720), max_value, type_string);

    // Sample different image sizes
    for (OffsetT rows = 1; rows < 1000000; rows *= 100)
    {
        for (OffsetT cols = 1; cols < (1000000 / rows); cols *= 100)
        {
            Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
                cols, rows, max_value, type_string);
        }
    }

    // Randomly select linear problem size between 1:10,000,000
    unsigned int max_int = (unsigned int) -1;
    for (int i = 0; i < 10; ++i)
    {
        unsigned int num_items;
        RandomBits(num_items);
        num_items = (unsigned int) ((double(num_items) * double(10000000)) / double(max_int));
        num_items = CUB_MAX(1, num_items);

        Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
            OffsetT(num_items), 1, max_value, type_string);
    }
}



/**
 * Test different channel interleavings
 */
template <
    typename        SampleT,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    LevelT          max_value,
    const char*     type_string)
{
    Test<SampleT, 1, 1, CounterT, LevelT, OffsetT>(max_value, type_string);
    Test<SampleT, 4, 3, CounterT, LevelT, OffsetT>(max_value, type_string);
    Test<SampleT, 3, 3, CounterT, LevelT, OffsetT>(max_value, type_string);
    Test<SampleT, 4, 4, CounterT, LevelT, OffsetT>(max_value, type_string);
}




//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------



/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_row_pixels = -1;
    int entropy_reduction = 0;
    int num_rows = 1;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose_input = args.CheckCmdLineFlag("v2");
    args.GetCmdLineArgument("n", num_row_pixels);

    int row_stride_pixels = num_row_pixels;

    args.GetCmdLineArgument("rows", num_rows);
    args.GetCmdLineArgument("stride", row_stride_pixels);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    bool compare_npp = args.CheckCmdLineFlag("npp");


    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<pixels per row> "
            "[--rows=<number of rows> "
            "[--stride=<row stride in pixels> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--entropy=<entropy-reduction factor (default 0)>]"
            "[--v] "
            "[--cdp]"
            "[--npp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get ptx version
    int ptx_version;
    CubDebugExit(PtxVersion(ptx_version));

    if (num_row_pixels < 0)
    {
        num_row_pixels      = 1920 * 1080;
        row_stride_pixels   = num_row_pixels;
    }

#if defined(QUICKER_TEST)

    // Compile/run quick tests
    {
        // HistogramEven: unsigned char 256 bins
        typedef unsigned char       SampleT;
        typedef int                 LevelT;

        LevelT  max_value           = 256;
        int     num_levels[1]       = {257};
        int     max_levels          = 257;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 1;

        TestEven<CUB, SampleT, 1, 1, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned char));
        if (compare_npp)
            TestEven<NPP, SampleT, 1, 1, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned char));
    }

    {
        // HistogramEven: 4/4 multichannel Unsigned char 256 bins
        typedef unsigned char       SampleT;
        typedef int                 LevelT;

        LevelT  max_value           = 256;
        int     num_levels[4]       = {257, 257, 257, 257};
        int     max_levels          = 257;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 4;

        TestEven<CUB, SampleT, 4, 4, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned char));
    }

    {
        // HistogramEven: 3/4 multichannel Unsigned char 256 bins
        typedef unsigned char       SampleT;
        typedef int                 LevelT;

        LevelT  max_value           = 256;
        int     num_levels[3]       = {257, 257, 257};
        int     max_levels          = 257;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 4;

        TestEven<CUB, SampleT, 4, 3, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned char));
        if (compare_npp)
            TestEven<NPP, SampleT, 4, 3, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned char));
    }

    {
        // HistogramEven: short [0,1024] 256 bins
        typedef unsigned short      SampleT;
        typedef unsigned short      LevelT;

        LevelT  max_value           = 1024;
        int     num_levels[1]       = {257};
        int     max_levels          = 257;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 1;

        TestEven<CUB, SampleT, 1, 1, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned short));
    }

    {
        // HistogramEven: float [0,1.0] 256 bins
        typedef float               SampleT;
        typedef float               LevelT;

        LevelT  max_value           = 1.0;
        int     num_levels[1]       = {257};
        int     max_levels          = 257;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 1;

        TestEven<CUB, SampleT, 1, 1, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(float));
    }

#elif defined(QUICK_TEST)

    {
        // HistogramRange: signed char 256 bins
        typedef signed char         SampleT;
        typedef int                 LevelT;

        LevelT  max_value           = 256;
        int     num_levels[1]       = {257};
        int     max_levels          = 257;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 1;

        TestRange<CUB, SampleT, 1, 1, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned char));
    }

    {
        // HistogramRange: 3/4 channel, unsigned char, varied bins (256, 128, 64)
        typedef unsigned char       SampleT;
        typedef int                 LevelT;

        LevelT  max_value           = 256;
        int     num_levels[3]       = {257, 129, 65};
        int     max_levels          = 257;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 4;

        TestRange<CUB, SampleT, 4, 3, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned char));
    }

    if (ptx_version > 120)                          // Don't check doubles on PTX120 or below because they're down-converted
    {
        // HistogramEven: double [0,1.0] 64 bins
        typedef double              SampleT;
        typedef double              LevelT;

        LevelT  max_value           = 1.0;
        int     num_levels[1]       = {65};
        int     max_levels          = 65;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 1;

        TestEven<CUB, SampleT, 1, 1, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(double));
    }

    {
        // HistogramEven: short [0,1024] 512 bins
        typedef unsigned short      SampleT;
        typedef unsigned short      LevelT;

        LevelT  max_value           = 1024;
        int     num_levels[1]       = {513};
        int     max_levels          = 513;
        int     row_stride_bytes    = sizeof(SampleT) * row_stride_pixels * 1;

        TestEven<CUB, SampleT, 1, 1, int, LevelT, int>(num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_levels, max_value, CUB_TYPE_STRING(unsigned short));
    }

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        Test <unsigned char,    int, int,   int>(256, CUB_TYPE_STRING(unsigned char));
        Test <signed char,      int, int,   int>(256, CUB_TYPE_STRING(signed char));
        Test <unsigned short,   int, int,   int>(256, CUB_TYPE_STRING(unsigned short));
        Test <float,            int, float, int>(1.0, CUB_TYPE_STRING(float));

		// Test down-conversion of size_t offsets to int
        if (sizeof(size_t) != sizeof(int))
        {
            Test <unsigned char,    int, int,   size_t>(256, CUB_TYPE_STRING(unsigned char));
        }
    }

#endif

    return 0;
}



