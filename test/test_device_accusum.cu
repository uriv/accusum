#define CUB_STDERR

#include <stdio.h>
#include <assert.h>
#include <string>
#include <sstream>
#include <limits>
#include <algorithm>
#include <unistd.h>
#include <math.h>

#include <device_functions.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <curand.h>

#include <mpfr.h>

#include <cub/device/device_accusum.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose       = false;
int                     g_repeat        = 0;
CachingDeviceAllocator  g_allocator(true);
float                   g_test_thrpt_gdbl_sec[1<<20] = { 0.f };
float                   g_max_thrpt = -1.f;
int                     g_max_thrpt_test = -1;
int                     g_max_thrpt_params[8] = {0};

/*
 * Type of generated input
 */
enum GenModeDbl
{
    GEN_UNINITIALIZED = 0,
    GEN_C0NSTANT_1,                         //< constant 1.0
    GEN_CONSTANT_FULL_MANTISSA,             //< constant 1.111111111111...1 (full mantissa)
    GEN_TWO_VALS_FULL_MANTISSA,             //< alternating two values: 1.1111111111...11 and 1.1111111111...10
    GEN_TWO_VALS_LAST_MANTISSA_BIT,         //< alternating two values: 1.0 and 1.000000000...01
    GEN_RANDOM,                             //< random within predefined exponent range
    GEN_RANDOM_POSITIVE,                    //< random within predefined exponent range (positive only)
    GEN_RANDOM_NEGATIVE,                    //< random within predefined exponent range (negative only)
    GEN_RANDOM_MANTISSA,                    //< 1.xxxxxxxxx...xx (random xxx...x mantissa)
    GEN_RANDOM_ALL_BITS,                    //< all bits random
};

const char *GenModeDblNames[] =
{
    "GEN_UNINITIALIZED",
    "GEN_C0NSTANT_1",
    "GEN_CONSTANT_FULL_MANTISSA",
    "GEN_TWO_VALS_FULL_MANTISSA",
    "GEN_TWO_VALS_LAST_MANTISSA_BIT",
    "GEN_RANDOM",
    "GEN_RANDOM_POSITIVE",
    "GEN_RANDOM_NEGATIVE",
    "GEN_RANDOM_MANTISSA",
    "GEN_RANDOM_ALL_BITS",
};


//---------------------------------------------------------------------
// Template classes and functions implementation
//---------------------------------------------------------------------

/////////////////////////////////////////////////////////////////

extern "C" double sum_mpfr(double *data, int size) {
    mpfr_t tmp;
    mpfr_t* mpfr_data;
    mpfr_ptr* mpfr_ptrs;
    int i;
    double result;
    const int DOUBLE_PREC = 53;
    CpuTimer timer;
    const bool use_mpfr_sum = true;     //< choose between mpfr_sum and a loop of mpfr_add

    if (use_mpfr_sum)
    {
        mpfr_init2(tmp, DOUBLE_PREC);
        mpfr_set_d(tmp, 0.0, MPFR_RNDN);
        mpfr_data = (mpfr_t*)malloc(size * sizeof(mpfr_t));
        mpfr_ptrs = (mpfr_ptr*)malloc(size * sizeof(mpfr_ptr));
        for (i = 0; i < size; i++)
        {
            mpfr_init2(mpfr_data[i], DOUBLE_PREC);
            mpfr_set_d(mpfr_data[i], data[i], MPFR_RNDN);
            mpfr_ptrs[i] = mpfr_data[i];
        }
        timer.Start();
        mpfr_sum(tmp, mpfr_ptrs, size, MPFR_RNDN);
        timer.Stop();

        for (i = 0; i < size; i++)
        {
            mpfr_clear (mpfr_data[i]);
        }
        free(mpfr_data);
        free(mpfr_ptrs);
    }
    else
    {
        mpfr_init2(tmp, 2048);
        mpfr_set_d(tmp, 0.0, MPFR_RNDN);
        timer.Start();
        for (i = 0; i < size; i++)
        {
            mpfr_add_d(tmp, tmp, data[i], MPFR_RNDN);
        }
        timer.Stop();
    }
    result = mpfr_get_d(tmp, MPFR_RNDN);
//    printf("MPFR Sum is %g [%016llX]  ", result, reinterpret_bits<unsigned long long>(result));
//    mpfr_out_str(stdout, 16, 0, tmp, MPFR_RNDD);
//    printf("\n");

//    float mpfr_sum_time = timer.ElapsedMillis();
//    printf("MPFR %u items %7.3f ms   |   %f GDbl/sec\n", size, mpfr_sum_time, (float)size/mpfr_sum_time*1e-6);

    mpfr_clear (tmp);
    return result;
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);}} while(0)

/**
 * Kernel that generates the input from randomly generated bits (or ignores initial values)
 */
__global__ void fix_input(double* items, long long int num_items, GenModeDbl INPUT_TYPE)
{
    const long long SIGN_MASK = (1LL << 63);
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;

    #pragma unroll 1024
    for (long long int i = global_tid; i < num_items; i += blockDim.x * gridDim.x)
    {
        switch (INPUT_TYPE)
        {
        case GEN_C0NSTANT_1:
            items[i] = 1.0;
            continue;
        case GEN_CONSTANT_FULL_MANTISSA:
            items[i] = __longlong_as_double(__double_as_longlong(1.0) | 0xFFFFFFFFFFFFFULL);
            continue;
        case GEN_TWO_VALS_FULL_MANTISSA:
            if (i % 2 == 0) items[i] = __longlong_as_double(__double_as_longlong(1.0) | 0xFFFFFFFFFFFFFULL); //< set all mantissa bits to 1
            else            items[i] = __longlong_as_double(__double_as_longlong(1.0) | 0xFFFFFFFFFFFFEULL); //< last bit is 0
            continue;
        case GEN_TWO_VALS_LAST_MANTISSA_BIT:
            if (i % 2 == 0) items[i] = 1.0;
            else            items[i] = __longlong_as_double(__double_as_longlong(1.0) | 0x1ULL); //< last bit is 1
            continue;
        case GEN_RANDOM:
        case GEN_RANDOM_POSITIVE:
        case GEN_RANDOM_NEGATIVE:
        case GEN_RANDOM_MANTISSA:
        {
            long long val = __double_as_longlong(items[i]);
            long long exp = val & (0x7ffULL << 52);
            if (INPUT_TYPE == GEN_RANDOM_MANTISSA)
            {
                exp = 0x3ffULL << 52; /*=1.0*/
            }
            else
            {
                exp = min(exp, (2047ULL - 14 /*<<PARAM*/) << 52);                        // BOUND EXPONENT HIGH
                exp = max(exp, (0ULL    + 14 /*<<PARAM*/) << 52);                        // BOUND EXPONENT LOW
            }
            val = val & ~(0x7ffULL << 52);  // clean exp bits
            val = val | exp;                // set new exp bits
            if (INPUT_TYPE == GEN_RANDOM_POSITIVE || INPUT_TYPE == GEN_RANDOM_MANTISSA)
            {
                val = val & ~SIGN_MASK;
            }
            if (INPUT_TYPE == GEN_RANDOM_NEGATIVE)
            {
                val = val | SIGN_MASK;
            }
            items[i] = __longlong_as_double(val);
            continue;
        }
        case GEN_RANDOM_ALL_BITS:
            /* nothing to do */
            continue;
        }

    }
}

/*
 * Generates input values for a specified input type (distribution of values)
 */
template <GenModeDbl GEN_MODE>
void GenerateInput(
    double     *h_in,
    double     *d_in,
    double     *h_reference,
    int         num_items,
    unsigned long long seed = 1234ULL)
{
    int num_blocks_fix_input = 28;
    int num_threads_fix_input = 256;
    switch (GEN_MODE)
    {
    case GEN_UNINITIALIZED:
        AssertEquals(1,-1);
        break;
    case GEN_C0NSTANT_1:
    case GEN_CONSTANT_FULL_MANTISSA:
    case GEN_TWO_VALS_FULL_MANTISSA:
    case GEN_TWO_VALS_LAST_MANTISSA_BIT:

        fix_input<<<num_blocks_fix_input, num_threads_fix_input>>>(d_in, num_items, GEN_MODE);
        CubDebugExit(cudaDeviceSynchronize());
        break;
    case GEN_RANDOM:
    case GEN_RANDOM_POSITIVE:
    case GEN_RANDOM_NEGATIVE:
    case GEN_RANDOM_MANTISSA:
    {
        curandGenerator_t gen;
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
        CURAND_CALL(curandGenerate(gen, (unsigned int*)d_in, num_items*2 ));  //< generate 2 32-bit words for every double
        CURAND_CALL(curandDestroyGenerator(gen));
        fix_input<<<num_blocks_fix_input, num_threads_fix_input>>>(d_in, num_items, GEN_MODE);
        CubDebugExit(cudaDeviceSynchronize());
        break;
    }
    }

    CUDA_CALL(cudaMemcpy(h_in, d_in, num_items * sizeof(double), cudaMemcpyDeviceToHost));

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, min(1024,num_items));
        printf("\n");
    }
}

/*
 * Performance testing for Accusum with various configurations
 */
struct AccusumBenchmark
{
    enum AccuSumMethod {
        ACCUSUM_SORT_REDUCE = 0,    //< sort-reduce accurate summation method
        ACCUSUM_SMEM_ATOMIC         //< atomics on shared memory summation method
    };

    static const int MAX_BLOCK_THREADS = 256;
    static const int MAX_ITEMS_PER_THREAD = 8;
    static const int MAX_EXPANSIONS = 8;
    static const int MAX_ITEMS = (1 << 26);
    static const int MAX_GRID_SIZE_LOWBOUND = 1024;

    double *h_in       ;
    double *h_out      ;
    double *h_reference;
    clock_t *h_elapsed  ;
    double *d_in       ;
    double *d_out      ;
    void   *d_global_bins ;
    double *h_global_bins ;
    double *h_map_temp_reduce;
    ExtremeFlags* d_extreme_flags;
    ExtremeFlags* h_extreme_flags;

    int device_id;
    int sm_count;

    GenModeDbl gen_mode;
    unsigned long long _seed;

    AccusumBenchmark() : gen_mode(GEN_UNINITIALIZED), _seed(1234ULL)
    {
        static const int MIN_EXPANSIONS         = 2;
        static const int MAX_BINS               = AccumulatorBinsMetadata<MAX_BLOCK_THREADS, MAX_ITEMS_PER_THREAD, MIN_EXPANSIONS>::NUM_BINS;
        static const int MAX_ITEMS_PER_BLOCK    = AccumulatorBinsMetadata<MAX_BLOCK_THREADS, MAX_ITEMS_PER_THREAD, MAX_EXPANSIONS>::BIN_CAPACITY;
        static const int MAX_BIN_SIZE           = AccumulatorBinsMetadata<MAX_BLOCK_THREADS, MAX_ITEMS_PER_THREAD, MAX_EXPANSIONS>::BIN_SIZE_BYTES;
        CubDebugExit(cudaGetDevice(&device_id));
        CubDebugExit(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
        int max_grid_size          = CUB_ROUND_UP_NEAREST(CUB_MAX(MAX_GRID_SIZE_LOWBOUND, MAX_ITEMS / MAX_ITEMS_PER_BLOCK), sm_count);

        h_in               = new double[MAX_ITEMS];
        h_out              = new double[1];
        h_reference        = new double[MAX_ITEMS];
        h_elapsed          = new clock_t[max_grid_size];

        h_global_bins      = new double[(MAX_EXPANSIONS + 1) * MAX_BINS];
        h_extreme_flags    = new ExtremeFlags[1];

        size_t max_temp_reduce_size = max_grid_size * MAX_BINS * MAX_BIN_SIZE;     //< more memory than required, but simply computed
        max_temp_reduce_size *= 15;   //< smem-atomic method uses 15 bins per block

        CubDebugExit(cudaMalloc((void**)&d_in,          sizeof(double) * MAX_ITEMS));
        CubDebugExit(cudaMalloc((void**)&d_out,         sizeof(double)));
        CubDebugExit(cudaHostAlloc((void**)&h_map_temp_reduce, max_temp_reduce_size, cudaHostAllocMapped));

        CubDebugExit(cudaMalloc((void**)&d_global_bins,         sizeof(double) * (MAX_EXPANSIONS + 1) * MAX_BINS ));
        CubDebugExit(cudaMalloc((void**)&d_extreme_flags,         sizeof(ExtremeFlags)));
    }

    ~AccusumBenchmark()
    {
        if (h_in) delete[] h_in;
        if (h_out) delete[] h_out;
        if (h_reference) delete[] h_reference;
        if (h_elapsed) delete[] h_elapsed;
        if (d_in) CubDebugExit(cudaFree(d_in));
        if (d_out) CubDebugExit(cudaFree(d_out));
        if (h_map_temp_reduce) cudaFreeHost(h_map_temp_reduce);

        if (h_global_bins) delete[] h_global_bins;
        if (d_out) CubDebugExit(cudaFree(d_global_bins));
        if (h_extreme_flags) delete[] h_extreme_flags;
        if (d_extreme_flags) CubDebugExit(cudaFree(d_extreme_flags));
    }

    /*
     * Simple test with some default configuration
     */
    void Test()
    {
        // see definition of Test<...> below
        Test<ACCUSUM_SORT_REDUCE, 128, 2, 2, 4, 8, GEN_RANDOM>();
    }

    template<
        int METHOD,
        int BLOCK_THREADS,
        int ITEMS_PER_THREAD,
        int EXPANSIONS,
        int RADIX_BITS,
        int MIN_CONCURRENT_BLOCKS,
        GenModeDbl GEN_MODE
    >
    float Test(int num_items = (1 << 25), bool validate = false, unsigned long long seed = 1234ULL)
    {
        float kernel_time = 0.f;
        float thrpt_gdbl_sec = 0.f;

        assert(num_items <= MAX_ITEMS);

        // generate input, if not previously generated
        if (gen_mode != GEN_MODE || _seed != seed)
        {
            GenerateInput<GEN_MODE>(h_in, d_in, h_reference, MAX_ITEMS, _seed);
            gen_mode = GEN_MODE;
            _seed = seed;
        }

        if (METHOD == ACCUSUM_SORT_REDUCE)
        {
            // AccumulatorBinsMetadata class provides compile-time configuration
            // parameters computed from the input configuration parameters
            typedef AccumulatorBinsMetadata<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS> BinMeta;

//            // round up number of items to nearest multiple of items processed in parallel
//            int device_id, sm_count;
//            CubDebugExit(cudaGetDevice(&device_id));
//            CubDebugExit(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
//            const int min_grid_size = sm_count * BLOCKS_PER_SM;
//            const int grid_size = CUB_ROUND_UP_NEAREST(CUB_MAX(min_grid_size, CUB_QUOTIENT_CEILING(num_items, BinMeta::BIN_CAPACITY)), sm_count);
//            const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
//            num_items = CUB_ROUND_UP_NEAREST(num_items, TILE_SIZE * grid_size);

            // get size and allocate temporary storage
            void   *d_temp_storage = NULL;
            void   *h_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            DeviceAccurateFPSum::SumSortReduce<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS>
                (d_in, num_items, d_out, d_temp_storage, h_temp_storage, temp_storage_bytes);
            cudaDeviceSynchronize();
            CubDebugExit(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes ));
            h_temp_storage = malloc(temp_storage_bytes);
            if (h_temp_storage == NULL)
            {
                printf("Cannot allocate temporary buffer in host memory\n");
                exit(-1);
            }

            // compute reference sum
            if (validate)
            {
                *h_reference = sum_mpfr(h_in, num_items);
            }

            cudaDeviceSynchronize();

            GpuTimer timer;
            timer.Start();
            DeviceAccurateFPSum::SumSortReduce<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS>
            (d_in, num_items, d_out, d_temp_storage, h_temp_storage, temp_storage_bytes);
            cudaDeviceSynchronize();
            timer.Stop();
            CubDebugExit(cudaMemcpy(h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
            kernel_time = timer.ElapsedMillis();
            thrpt_gdbl_sec = (float)num_items / kernel_time * 1e-6f;
//            printf("%u items %7.3f ms   |   %f GDbl/sec\n", num_items, kernel_time, thrpt_gdbl_sec);
            if (d_temp_storage) cudaFree(d_temp_storage);
            if (h_temp_storage) free(h_temp_storage);
        }
        else if (METHOD == ACCUSUM_SMEM_ATOMIC)
        {
            /////////////////////
            // Test accumulation in shared mem
            // Benchmark: K40c  random data    4.0GDbl/sec
            //                  constant data  0.2GDbl/sec

            // get size and allocate temporary storage
            void   *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            CubDebugExit(DeviceAccurateFPSum::SumSmemAtomic(d_in, num_items, d_out, d_temp_storage, temp_storage_bytes));
            CubDebugExit(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes ));

            // compute reference sum
            if (validate)
            {
                *h_reference = sum_mpfr(h_in, num_items);
            }

            GpuTimer timer;
            timer.Start();
            CubDebugExit(DeviceAccurateFPSum::SumSmemAtomic(d_in, num_items, d_out, d_temp_storage, temp_storage_bytes));
            cudaDeviceSynchronize();
            timer.Stop();
            CubDebugExit(cudaMemcpy(h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
            kernel_time = timer.ElapsedMillis();
            thrpt_gdbl_sec = (float)num_items / kernel_time * 1e-6f;
            printf("%u items %7.3f ms   |   %f GDbl/sec\n", num_items, kernel_time, (float)num_items/kernel_time*1e-6);
            if (d_temp_storage) cudaFree(d_temp_storage);
        }
        /////////////////////

        // Check results
        if (validate)
        {
            int compare;
            compare = (reinterpret_bits<unsigned long long>(h_reference[0]) != reinterpret_bits<unsigned long long>(h_out[0]));
            if (compare)
            {
                thrpt_gdbl_sec = -1.f;
                printf("\nREF, RES: %f %f | %g, %g [%016llX, %016llX]\n",
                    *h_reference,
                    *h_out,
                    *h_reference,
                    *h_out,
                    reinterpret_bits<unsigned long long>(*h_reference),
                    reinterpret_bits<unsigned long long>(*h_out));
            }
        }

        return thrpt_gdbl_sec;
    }
};

/*
 * Range of values to test for a configuration parameter
 */
template<int _BEGIN, int _END>
struct Range {
    enum {
        BEGIN = _BEGIN,
        END = _END
    };
};

/*
 * Measure accurate summation throughput for a setup that consists of configuration parameters
 *
 * Input:
 * TEST_NUM  - serial number of execution setup
 * SETUP     - contains a range of values for every input configuration parameter
 * testobj   - executes test with given parameters
 * num_items - number of items to sum
 * validate  - (flag) validate results with CPU
 * seed      - randomization seed
 */
template <int TEST_NUM, typename SETUP>
float RunTest(Int2Type<TEST_NUM> test_num, SETUP setup, AccusumBenchmark& testobj, int num_items = (1 << 25), bool validate = false, unsigned long long seed = 1234ULL)
{
    const int RANGE1 = (SETUP::WarpsPerBlock::END     - SETUP::WarpsPerBlock::BEGIN + 1)  ;
    const int RANGE2 = (SETUP::ItemsPerThread::END    - SETUP::ItemsPerThread::BEGIN + 1) ;
    const int RANGE3 = (SETUP::Expansions::END        - SETUP::Expansions::BEGIN + 1)     ;
    const int RANGE4 = (SETUP::RadixBits::END         - SETUP::RadixBits::BEGIN + 1)      ;
    const int RANGE5 = (SETUP::MinConcurrentBlocks::END - SETUP::MinConcurrentBlocks::BEGIN + 1)      ;
    const int RANGE6 = (SETUP::GenerationMode::END    - SETUP::GenerationMode::BEGIN + 1) ;

    const int TEMP1 = TEST_NUM;
    const int TEMP2 = TEMP1 / RANGE1;
    const int TEMP3 = TEMP2 / RANGE2;
    const int TEMP4 = TEMP3 / RANGE3;
    const int TEMP5 = TEMP4 / RANGE4;
    const int TEMP6 = TEMP5 / RANGE5;

    const int WARPS_PER_BLOCK =     (TEMP1 % RANGE1) + SETUP::WarpsPerBlock ::BEGIN;
    const int ITEMS_PER_THREAD =    (TEMP2 % RANGE2) + SETUP::ItemsPerThread::BEGIN;
    const int EXPANSIONS =          (TEMP3 % RANGE3) + SETUP::Expansions    ::BEGIN;
    const int RADIX_BITS =          (TEMP4 % RANGE4) + SETUP::RadixBits     ::BEGIN;
    const int MIN_CONCURRENT_BLOCKS = (TEMP5 % RANGE5) + SETUP::MinConcurrentBlocks::BEGIN;
    const int GENERATION_MODE =     (TEMP6 % RANGE6) + SETUP::GenerationMode::BEGIN;

    const int BLOCK_THREADS = WARPS_PER_BLOCK * CUB_PTX_WARP_THREADS;
//    const int GRID_SIZE = BLOCKS_PER_SM * 14;
    float thrpt_gdbl_sec = testobj.Test<SETUP::Method, BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS, (GenModeDbl)GENERATION_MODE >(num_items, validate, seed);
    printf ("%5d,%9g,%5d,%5d,%5d,%5d,%5d,%10s,%5d,%8d\n",
        TEST_NUM,
        thrpt_gdbl_sec       ,
        WARPS_PER_BLOCK      ,
        ITEMS_PER_THREAD     ,
        EXPANSIONS           ,
        RADIX_BITS           ,
        MIN_CONCURRENT_BLOCKS,
        GenModeDblNames[GENERATION_MODE],
        BLOCK_THREADS        ,
        num_items
        );

    g_test_thrpt_gdbl_sec[TEST_NUM] = thrpt_gdbl_sec;

    if (thrpt_gdbl_sec > 0.f && thrpt_gdbl_sec > g_max_thrpt)
    {
        g_max_thrpt = thrpt_gdbl_sec;
        g_max_thrpt_test = TEST_NUM;
        g_max_thrpt_params[0] = WARPS_PER_BLOCK;
        g_max_thrpt_params[1] = ITEMS_PER_THREAD;
        g_max_thrpt_params[2] = EXPANSIONS;
        g_max_thrpt_params[3] = RADIX_BITS;
        g_max_thrpt_params[4] = MIN_CONCURRENT_BLOCKS;
        g_max_thrpt_params[5] = GENERATION_MODE;
        g_max_thrpt_params[6] = BLOCK_THREADS;
        g_max_thrpt_params[7] = num_items;
    }

    return thrpt_gdbl_sec;
}

/*
 * Run a series of tests from SETUP and write the throughput values to an array
 */
template <typename SETUP>
void RunTests(AccusumBenchmark& testobj, int num_items, bool validate, unsigned long long seed)
{
    const int NUM_TESTS =
        (SETUP::WarpsPerBlock::END     - SETUP::WarpsPerBlock::BEGIN + 1)     *
        (SETUP::ItemsPerThread::END    - SETUP::ItemsPerThread::BEGIN + 1)    *
        (SETUP::Expansions::END        - SETUP::Expansions::BEGIN + 1)        *
        (SETUP::RadixBits::END         - SETUP::RadixBits::BEGIN + 1)         *
        (SETUP::MinConcurrentBlocks::END - SETUP::MinConcurrentBlocks::BEGIN + 1) *
        (SETUP::GenerationMode::END    - SETUP::GenerationMode::BEGIN + 1);
    RunTests<0,NUM_TESTS,SETUP>(testobj, num_items, validate, seed);
}

template <int FIRST, int COUNT, typename SETUP>
void RunTests(AccusumBenchmark& testobj, int num_items, bool validate, unsigned long long seed)
{
    assert(FIRST + COUNT <= (sizeof(g_test_thrpt_gdbl_sec) / sizeof(g_test_thrpt_gdbl_sec[0])) );

    printf ("%5s,%5s,%5s,%5s,%5s,%5s,%5s,%10s,%5s,%8s\n",
        "TEST_NUM",
        "thrpt_gdbl_sec",
        "WARPS_PER_BLOCK",
        "ITEMS_PER_THREAD",
        "EXPANSIONS",
        "RADIX_BITS",
        "MIN_CONCURRENT_BLOCKS",
        "GENERATION_MODE",
        "BLOCK_THREADS",
        "num_items"
        );

    memset(&g_test_thrpt_gdbl_sec[FIRST], 0, COUNT * sizeof(float));
    g_max_thrpt = -1.f;
    g_max_thrpt_test = -1;
    memset(g_max_thrpt_params, 0, sizeof(g_max_thrpt_params));
    RunTests(Int2Type<FIRST>(), Int2Type<FIRST + COUNT>(), SETUP(), testobj, num_items, validate, seed);
    printf(",,,,,,,,,,,BEST_TEST,%d,MAXTHRPT=,%9g\n", g_max_thrpt_test, g_max_thrpt);
    printf("\nBest setup:\n\n");
    printf (
        "%10s = %d\n%10s = %d\n%10s = %d\n%10s = %d\n%10s = %d\n%10s = %d\n%10s = %d\n%10s = %d\n",
        "WARPS_PER_BLOCK",       g_max_thrpt_params[0],
        "ITEMS_PER_THREAD",      g_max_thrpt_params[1],
        "EXPANSIONS",            g_max_thrpt_params[2],
        "RADIX_BITS",            g_max_thrpt_params[3],
        "MIN_CONCURRENT_BLOCKS", g_max_thrpt_params[4],
        "GENERATION_MODE",       g_max_thrpt_params[5],
        "BLOCK_THREADS",         g_max_thrpt_params[6],
        "num_items",             g_max_thrpt_params[7]);
}

template <int TEST_NUM, int NUM_TESTS, typename SETUP>
void RunTests(Int2Type<TEST_NUM> test_num, Int2Type<NUM_TESTS> num_tests, SETUP setup, AccusumBenchmark& testobj, int num_items = (1 << 25), bool validate = false, unsigned long long seed = 1234ULL)
{
    RunTest(Int2Type<TEST_NUM>(), setup, testobj, num_items, validate, seed);
    RunTests(Int2Type<TEST_NUM + 1>(), num_tests, setup, testobj, num_items, validate, seed);
}

template <int NUM_TESTS, typename SETUP>
void RunTests(Int2Type<NUM_TESTS> test_num, Int2Type<NUM_TESTS> num_tests, SETUP setup, AccusumBenchmark& testobj, int num_items = (1 << 25), bool validate = false, unsigned long long seed = 1234ULL)
{
}

///////////////////////////////////////////////////////////////////////////////

/*
 * Default configuration parameters setup.
 * Other setups inherit from it and override specific parameters
 */
struct SetupDefault
{
    enum { Method = AccusumBenchmark::ACCUSUM_SORT_REDUCE};  /*ACCUSUM_SMEM_ATOMIC ACCUSUM_SORT_REDUCE*/
    typedef Range<4,4>      WarpsPerBlock;       //< for BLOCK_SIZE multiple by warp size (32)
    typedef Range<5,5>      ItemsPerThread;
    typedef Range<2,2>      Expansions;
    typedef Range<3,3>      RadixBits;
    typedef Range<5,5>      MinConcurrentBlocks;    //< to be used as a parameter to __launch_bounds__
    typedef Range<5,5>      GenerationMode;
};

/* Test parameter: warps per thread-block */
struct SetupWarpsPerBlock : public SetupDefault
{
    typedef Range<1,6>      WarpsPerBlock;
};

/* Test parameter: items per thread */
struct SetupItemsPerThread : public SetupDefault
{
    typedef Range<4,8>      ItemsPerThread;
};

/* Test parameter: accumulator expansion size (number of words per bin) */
struct SetupExpansions : public SetupDefault
{
    typedef Range<2,8>      Expansions;
};

/* Test parameter: number of radix bits used in sorting */
struct SetupRadixBits : public SetupDefault
{
    typedef Range<3,6>      RadixBits;
};

/* Test parameter: minimum blocks per SM (parameter to __launch_bounds__) */
struct SetupMinConcurrentBlocks : public SetupDefault
{
    typedef Range<2,16>      MinConcurrentBlocks;
};

/* Test parameter: input generation mode (enum GenModeDbl) */
struct SetupGenMode : public SetupDefault
{
    typedef Range<1,9>      GenerationMode;
};

/* Test parameter: accumulation method (sort-reduce / smem-atomic) */
template<typename Setup, int METHOD>
struct SetupMethod : public Setup
{
    enum { Method = METHOD};
};

/*
 * setups for different numbers of items
 */
template<int NUM_ITEMS> struct SetupDefaultByNumItems {};
template<> struct SetupDefaultByNumItems< (1 << 16) > : public SetupDefault
{
};
template<> struct SetupDefaultByNumItems< (1 << 28) > : public SetupDefault
{
};

/*
 * custom setup
 */
struct SetupCustom : public SetupDefault
{
    enum { Method = AccusumBenchmark::ACCUSUM_SMEM_ATOMIC };
//    typedef Range<1,1>      GenerationMode;
};

void TestCustom()
{


    AccusumBenchmark testobj;

    int num_items = (1 << 25);
    bool validate = true;
    unsigned long long seed = 1234ULL;


//    RunTests<SetupDefaultByNumItems< 1<<16 > >(testobj, 1 << 25, validate, seed);
//    RunTests<SetupDefaultByNumItems< 1<<28 > >(testobj, 1 << 28, validate, seed);

//    RunTests<SetupDefault>(testobj, 1 << 10, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 12, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 14, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 16, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 18, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 20, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 22, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 24, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 26, validate, seed);
//    RunTests<SetupDefault>(testobj, 1 << 28, validate, seed);

//    RunTests<SetupMethod<SetupDefault, AccusumBenchmark::ACCUSUM_SORT_REDUCE> >(testobj, 1<<25, true, seed);
//    RunTests<SetupMethod<SetupDefault, AccusumBenchmark::ACCUSUM_SMEM_ATOMIC> >(testobj, 1<<25, true, seed);

//    RunTests<SetupGenMode>(testobj, num_items, validate, seed);
    RunTests<SetupDefault>(testobj, num_items, validate, seed);
//    RunTests<SetupCustom>(testobj, num_items, validate, seed);
//    RunTests<SetupCustom>(testobj, num_items, validate, seed);

//    int num_items_options[] = {
//        1<<16,1<<17, 1<<18, 1<<19,
//        1<<20, 1<<21, 1<<22, 1<<23,
//        1<<24, 1<<25, 1<<26, 1<<27,
//        1<<28,
//        1<<29
//    };
//    for (int i = 0; i < sizeof(num_items_options) / sizeof(int); i++)
//    {
//        num_items = num_items_options[i];
//        RunTests<SetupDefault>(testobj, num_items, true, seed);
////        RunTests<SetupWarpsPerBlock>(testobj, num_items, validate, seed);
////        RunTests<SetupItemsPerThread>(testobj, num_items, validate, seed);
////        RunTests<SetupExpansions>(testobj, num_items, validate, seed);
////        RunTests<SetupRadixBits>(testobj, num_items, validate, seed);
////        RunTests<SetupMinConcurrentBlocks>(testobj, num_items, validate, seed);
////        RunTests<SetupGenMode>(testobj, num_items, validate, seed);
//    }

//    RunTests<SetupWarpsPerBlock>(testobj, num_items, validate, seed);
//    RunTests<SetupItemsPerThread>(testobj, num_items, validate, seed);
//    RunTests<SetupExpansions>(testobj, num_items, validate, seed);
//    RunTests<SetupRadixBits>(testobj, num_items, validate, seed);
//    RunTests<SetupMinConcurrentBlocks>(testobj, num_items, validate, seed);
    RunTests<SetupGenMode>(testobj, num_items, validate, seed);
}

template<int>
void Tune(int num_items = (1 << 25), bool validate = true, unsigned long long seed = 1234ULL)
{
    AccusumBenchmark testobj;
    float max_thrpt = -1.f;
    int max_thrpt_test = -1;
    bool default_setup_is_optimal = true;

    enum {
        TEST_WARPS_PER_BLOCK,
        TEST_ITEMS_PER_THREAD,
        TEST_RADIX_BITS,
        TEST_MIN_CONCURRENT_BLOCKS,
        TEST_EXPANSIONS
    } max_thrpt_param;

    RunTests<SetupWarpsPerBlock>(testobj, num_items, validate, seed);
    if (max_thrpt < g_max_thrpt)
    {
        max_thrpt = g_max_thrpt;
        max_thrpt_test = g_max_thrpt_test;
        max_thrpt_param = TEST_WARPS_PER_BLOCK;
    }

    RunTests<SetupItemsPerThread>(testobj, num_items, validate, seed);
    if (max_thrpt < g_max_thrpt)
    {
        max_thrpt = g_max_thrpt;
        max_thrpt_test = g_max_thrpt_test;
        max_thrpt_param = TEST_ITEMS_PER_THREAD;
    }

    RunTests<SetupRadixBits>(testobj, num_items, validate, seed);
    if (max_thrpt < g_max_thrpt)
    {
        max_thrpt = g_max_thrpt;
        max_thrpt_test = g_max_thrpt_test;
        max_thrpt_param = TEST_RADIX_BITS;
    }

    RunTests<SetupMinConcurrentBlocks>(testobj, num_items, validate, seed);
    if (max_thrpt < g_max_thrpt)
    {
        max_thrpt = g_max_thrpt;
        max_thrpt_test = g_max_thrpt_test;
        max_thrpt_param = TEST_MIN_CONCURRENT_BLOCKS;
    }

//    RunTests<SetupExpansions>(testobj, num_items, validate, seed);.
//    if (max_thrpt < g_max_thrpt)
//    {
//        max_thrpt = g_max_thrpt;
//        max_thrpt_test = g_max_thrpt_test;
//        max_thrpt_param = TEST_EXPANSIONS;
//    }

    if (max_thrpt_test == -1)
        return;

    switch(max_thrpt_param)
    {
    case TEST_WARPS_PER_BLOCK:
        if ( SetupDefault::WarpsPerBlock::BEGIN != g_max_thrpt_test + SetupWarpsPerBlock::WarpsPerBlock::BEGIN)
        {
            printf("Tune default setup: %s = %d\n", "WarpsPerBlock", g_max_thrpt_test + SetupWarpsPerBlock::WarpsPerBlock::BEGIN);
            default_setup_is_optimal = false;
        }
        break;
    case TEST_ITEMS_PER_THREAD:
        if ( SetupDefault::ItemsPerThread::BEGIN != g_max_thrpt_test + SetupItemsPerThread::ItemsPerThread::BEGIN)
        {
            printf("Tune default setup: %s = %d\n", "ItemsPerThread", g_max_thrpt_test + SetupItemsPerThread::ItemsPerThread::BEGIN);
            default_setup_is_optimal = false;
        }
        break;
    case TEST_RADIX_BITS:
        if ( SetupDefault::RadixBits::BEGIN != g_max_thrpt_test + SetupRadixBits::RadixBits::BEGIN)
        {
            printf("Tune default setup: %s = %d\n", "RadixBits", g_max_thrpt_test + SetupRadixBits::RadixBits::BEGIN);
            default_setup_is_optimal = false;
        }
        break;
    case TEST_MIN_CONCURRENT_BLOCKS:
        if ( SetupDefault::MinConcurrentBlocks::BEGIN != g_max_thrpt_test + SetupMinConcurrentBlocks::MinConcurrentBlocks::BEGIN)
        {
            printf("Tune default setup: %s = %d\n", "MinConcurrentBlocks", g_max_thrpt_test + SetupMinConcurrentBlocks::MinConcurrentBlocks::BEGIN);
            default_setup_is_optimal = false;
        }
        break;
    case TEST_EXPANSIONS:
        if ( SetupDefault::Expansions::BEGIN != g_max_thrpt_test + SetupExpansions::Expansions::BEGIN)
        {
            printf("Tune default setup: %s = %d\n", "Expansions", g_max_thrpt_test + SetupExpansions::Expansions::BEGIN);
            default_setup_is_optimal = false;
        }
        break;
    }

    if (default_setup_is_optimal)
        printf("Tuning result: default setup is optimal\n");

}

//struct TuningVariables {
//    enum { NVARS = 4 };
//    typedef SetupWarpsPerBlock          Test1;
//    typedef SetupItemsPerThread         Test2;
//    typedef SetupMinConcurrentBlocks    Test3;
//};
//
//template <typename Tuning>
//void OptimizeConfiguration()
//{
//    AccusumBenchmark testobj;
//
//    int num_items = (1 << 25);
//    bool validate = true;
//    unsigned long long seed = 1234ULL;
//
//
//
//    RunTests<SetupWarpsPerBlock>(testobj, num_items, validate, seed);
//    RunTests<SetupItemsPerThread>(testobj, num_items, validate, seed);
////    RunTests<SetupExpansions>(testobj, num_items, validate, seed);
////    RunTests<SetupRadixBits>(testobj, num_items, validate, seed);
//    RunTests<SetupMinConcurrentBlocks>(testobj, num_items, validate, seed);
//}

/*
 * Simple accurate summation test with custom configuration parameters (setup)
 */
void simple_test_with_setup()
{
    printf("Simple test with setup started\n");
    double* h_items = NULL;
    double* d_items = NULL;
    double* d_result = NULL;
    void*   h_temp  = NULL;
    void*   d_temp  = NULL;
    size_t d_temp_bytes = 0;
    double result = 0.;
    double reference = 0.;

    // performance tuning parameters
    enum SortReduceConfig_e{
        BLOCK_THREADS =             SetupDefault::WarpsPerBlock::BEGIN * CUB_PTX_WARP_THREADS,
        ITEMS_PER_THREAD =          SetupDefault::ItemsPerThread::BEGIN,
        EXPANSIONS =                SetupDefault::Expansions::BEGIN,
        RADIX_BITS =                SetupDefault::RadixBits::BEGIN,
        MIN_CONCURRENT_BLOCKS =     SetupDefault::MinConcurrentBlocks::BEGIN,
    };

    int num_items = 1 << 24;

    printf("Working on %d items.\n", num_items);

    h_items = (double*)malloc(num_items * sizeof(double));
    if (h_items == NULL)
    {
        printf("Cannot allocate host memory\n");
        exit(-1);
    }
    memset(h_items, 0, num_items * sizeof(double));
    std::fill(&h_items[0], &h_items[num_items], 1.0);   // initialize all items to 1.0

    // get required size of temporary space on device to d_temp_bytes
    DeviceAccurateFPSum::SumSortReduce<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS>
        (NULL, num_items, NULL, NULL, NULL, d_temp_bytes);

    // allocate and initialize device memory
    CubDebugExit(cudaMalloc((void**)&d_items, num_items * sizeof(double)));
    CubDebugExit(cudaMalloc((void**)&d_result, sizeof(double) ));
    CubDebugExit(cudaMalloc((void**)&d_temp, d_temp_bytes ));
    h_temp = malloc(d_temp_bytes);
    if (h_temp == NULL)
    {
        printf("Cannot allocate temporary buffer in host memory\n");
        exit(-1);
    }

    CubDebugExit(cudaMemcpy(d_items, h_items, num_items * sizeof(double), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_result, 0, sizeof(double)));
    CubDebugExit(cudaMemset(d_temp, 0, d_temp_bytes));
    memset(h_temp, 0, d_temp_bytes);

    DeviceAccurateFPSum::SumSortReduce<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS>
        (d_items, num_items, d_result, d_temp, h_temp, d_temp_bytes);
    cudaDeviceSynchronize();
    CubDebugExit(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    printf("Computed result: %g [%016llX]\n", result, reinterpret_bits<unsigned long long>(result));

    // compute reference sum
    reference = sum_mpfr(h_items, num_items);

    // validate result
    int compare;
    compare = (reinterpret_bits<unsigned long long>(reference) != reinterpret_bits<unsigned long long>(result));
    if (compare)
    {
        printf("\nREF, RES: %f %f | %g, %g [%016llX, %016llX]\n",
            reference,
            result,
            reference,
            result,
            reinterpret_bits<unsigned long long>(reference),
            reinterpret_bits<unsigned long long>(result));
    }
    else
    {
        printf("Validation OK.");
    }

    free(h_items);
    CubDebugExit(cudaFree((void*)d_items));
    CubDebugExit(cudaFree((void*)d_result));
    CubDebugExit(cudaFree((void*)d_temp));
    free(h_temp);
    printf("Simple test with setup finished\n");
}

/*
 * Simple accurate summation test
 */
void simple_test()
{
    printf("Simple test started\n");
    double* h_items = NULL;
    double* d_items = NULL;
    double* d_result = NULL;
    double* d_temp  = NULL;
    void*   h_temp  = NULL;
    size_t d_temp_bytes = 0;
    double result = 0.;
    double reference = 0.;

    int num_items = 1 << 24;

    printf("Working on %d items.\n", num_items);

    h_items = (double*)malloc(num_items * sizeof(double));
    if (h_items == NULL)
    {
        printf("Cannot allocate host memory\n");
        return;
    }
    memset(h_items, 0, num_items * sizeof(double));
    std::fill(&h_items[0], &h_items[num_items], 1.0);   // initialize all items to 1.0

    // get required size of temporary space on device to d_temp_bytes
    DeviceAccurateFPSum::Sum(NULL, num_items, NULL, NULL, NULL, d_temp_bytes);
    cudaDeviceSynchronize();

    // allocate and initialize device memory
    CubDebugExit(cudaMalloc((void**)&d_items, num_items * sizeof(double)));
    CubDebugExit(cudaMalloc((void**)&d_result, sizeof(double) ));
    CubDebugExit(cudaMalloc((void**)&d_temp, d_temp_bytes ));
    h_temp = malloc(d_temp_bytes);
    if (h_temp == NULL)
    {
        printf("Cannot allocate temporary buffer in host memory\n");
        exit(-1);
    }

    CubDebugExit(cudaMemcpy(d_items, h_items, num_items * sizeof(double), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_result, 0, sizeof(double)));
    CubDebugExit(cudaMemset(d_temp, 0, d_temp_bytes));
    memset(h_temp, 0, d_temp_bytes);

    DeviceAccurateFPSum::Sum(d_items, num_items, d_result, d_temp, h_temp, d_temp_bytes);
    cudaDeviceSynchronize();
    CubDebugExit(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    printf("Computed result: %g [%016llX]\n", result, reinterpret_bits<unsigned long long>(result));

    // compute reference sum
    reference = sum_mpfr(h_items, num_items);

    // validate result
    int compare;
    compare = (reinterpret_bits<unsigned long long>(reference) != reinterpret_bits<unsigned long long>(result));
    if (compare)
    {
        printf("\nREF, RES: %f %f | %g, %g [%016llX, %016llX]\n",
            reference,
            result,
            reference,
            result,
            reinterpret_bits<unsigned long long>(reference),
            reinterpret_bits<unsigned long long>(result));
    }
    else
    {
        printf("Validation OK.");
    }

    free(h_items);
    CubDebugExit(cudaFree((void*)d_items));
    CubDebugExit(cudaFree((void*)d_result));
    CubDebugExit(cudaFree((void*)d_temp));
    free(h_temp);
    printf("Simple test finished\n");
}

/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    size_t printf_buffer_size = 1 << 30;
    CubDebugExit(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printf_buffer_size));
    CubDebugExit(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

//    simple_test();
//    simple_test_with_setup();

    // Remove comments to tune default setup for the GPU in use. May increase build time.
//     Tune<0>();

    TestCustom();
    cudaDeviceReset();
    return 0;
}
