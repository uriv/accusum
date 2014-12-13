#define CUB_STDERR

#include <stdio.h>
#include <assert.h>
#include <string>
#include <sstream>
#include <limits>
#include <algorithm>
#include <math.h>

#include <device_functions.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <curand.h>

#include <mpfr.h>

#include <cub/device/device_accusum.cuh>

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose       = false;
int                     g_repeat        = 0;
CachingDeviceAllocator  g_allocator(true);
__device__ double     g_dbg_dummy;
float                   g_test_thrpt_gdbl_sec[1<<20] = { 0.f };

#define SERIALIZE(code) \
    if (blockIdx.x == 0) { \
      for (int __j_4921 = 0; __j_4921 < BLOCK_THREADS; ++__j_4921) { \
          if (threadIdx.x == __j_4921) { code } \
          __syncthreads(); \
    }}

#define SERIALIZE_BLOCK(code,b) \
    if (blockIdx.x == b) { \
      for (int __j_4921 = 0; __j_4921 < BLOCK_THREADS; ++__j_4921) { \
          if (threadIdx.x == __j_4921) { code } \
          __syncthreads(); \
    }}

#define STRINGIFY(x) #x

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

template<int Expansion>
std::string CoutCast(const AccumulatorDouble<Expansion>& val)
{
    std::stringstream string_build;
    double words[Expansion];
    val.Store(words);
//    string_build.precision(std::numeric_limits< double >::digits10);
//    string_build << std::fixed;
    string_build << std::hex;
    string_build.width(16);
    string_build << "[ ";
    if (Expansion > 0)
    {
        string_build << words[0];
    }

    for(int i=1; i<Expansion; i++)
    {
        string_build << ", " << words[i];
    }

    string_build << " ]";

    return string_build.str();
}



/////////////////////////////////////////////////////////////////

template<typename T>
struct abscmp {
    __host__ __device__ __forceinline__
    bool operator()(const double& d1, const double& d2) { return abs(d1) < abs(d2); }
};

extern "C" double sum_mpfr(double *data, int size) {
    mpfr_t tmp;
    int i;
    double result;
    mpfr_init2(tmp, 2098);
    mpfr_set_d(tmp, 0.0, MPFR_RNDN);

    for (i = 0; i < size; i++)
    {
        mpfr_add_d(tmp, tmp, data[i], MPFR_RNDN);
//        if (data[i] != 1.0 && data[i] != 0.0)
//        {
//            printf(">> (+%g) ", data[i]);
//            mpfr_out_str(stdout, 16, 0, tmp, MPFR_RNDD);
//            printf("\n");
//        }
    }

    result = mpfr_get_d(tmp, MPFR_RNDN);
//    printf("MPFR Sum is %g [%016llX]  ", result, reinterpret_bits<unsigned long long>(result));
//    mpfr_out_str(stdout, 16, 0, tmp, MPFR_RNDD);
//    printf("\n");
    return result;
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);}} while(0)

/**
 * modify input (random bits or constant 0) to match input generation mode
 */
__global__ void fix_input(double* items, long long int num_items, GenModeDbl INPUT_TYPE)
{
//    unsigned long long* ull_items = (unsigned long long*)items;
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

template <GenModeDbl GEN_MODE>
void InitializeExponentSort(
    double     *h_in,
    double     *d_in,
    double     *h_reference,
    int         num_items,
    unsigned long long seed = 1234ULL)
{
    switch (GEN_MODE)
    {
    case GEN_UNINITIALIZED:
        AssertEquals(1,-1);
        break;
    case GEN_C0NSTANT_1:
    case GEN_CONSTANT_FULL_MANTISSA:
    case GEN_TWO_VALS_FULL_MANTISSA:
    case GEN_TWO_VALS_LAST_MANTISSA_BIT:
        fix_input<<<14*2, 256>>>(d_in, num_items, GEN_MODE);
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
        fix_input<<<14*2, 256>>>(d_in, num_items, GEN_MODE);
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

struct TestExponentSort
{
    enum AccuSumMethod {
        ACCUSUM_SORT_REDUCE = 0,
        ACCUSUM_SMEM_ATOMIC
    };

//    static const int SM_COUNT = 13;
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
    void   *d_megabins ;
    double *h_megabins ;
    double *h_map_temp_reduce;
    ExtremeFlags* d_extreme_flags;
    ExtremeFlags* h_extreme_flags;

    int device_id;
    int sm_count;
    cudaEvent_t start;
    cudaEvent_t stop;

    GenModeDbl gen_mode;
    unsigned long long _seed;

    TestExponentSort() : gen_mode(GEN_UNINITIALIZED), _seed(1234ULL)
    {
        static const int MIN_EXPANSIONS         = 2;
        static const int MAX_BINS               = AccumulatorBinsMetadata<MAX_BLOCK_THREADS, MAX_ITEMS_PER_THREAD, MIN_EXPANSIONS>::NUM_BINS;
        static const int MAX_ITEMS_PER_BLOCK    = AccumulatorBinsMetadata<MAX_BLOCK_THREADS, MAX_ITEMS_PER_THREAD, MAX_EXPANSIONS>::BIN_CAPACITY;
        static const int MAX_BIN_SIZE           = AccumulatorBinsMetadata<MAX_BLOCK_THREADS, MAX_ITEMS_PER_THREAD, MAX_EXPANSIONS>::BIN_SIZE_BYTES;
//        static const int MAX_GRID_SIZE          = CUB_ROUND_UP_NEAREST(CUB_MAX(MAX_GRID_SIZE_LOWBOUND, MAX_ITEMS / MAX_ITEMS_PER_BLOCK), SM_COUNT);
        CubDebugExit(cudaGetDevice(&device_id));
        CubDebugExit(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
        int max_grid_size          = CUB_ROUND_UP_NEAREST(CUB_MAX(MAX_GRID_SIZE_LOWBOUND, MAX_ITEMS / MAX_ITEMS_PER_BLOCK), sm_count);

        h_in               = new double[MAX_ITEMS];
        h_out              = new double[1];
        h_reference        = new double[MAX_ITEMS];
        h_elapsed          = new clock_t[max_grid_size];

        h_megabins          = new double[(MAX_EXPANSIONS + 1) * MAX_BINS];
        h_extreme_flags     = new ExtremeFlags[1];

//        int num_iterations = CUB_QUOTIENT_CEILING(MAX_ITEMS, MAX_ITEMS_PER_BLOCK);
        size_t max_temp_reduce_size = max_grid_size * MAX_BINS * MAX_BIN_SIZE;     //< more memory than required, but simply computed
        max_temp_reduce_size *= 15;   //< smem-atomic method uses 15 bins per block

        CubDebugExit(cudaMalloc((void**)&d_in,          sizeof(double) * MAX_ITEMS));
        CubDebugExit(cudaMalloc((void**)&d_out,         sizeof(double)));
        CubDebugExit(cudaHostAlloc((void**)&h_map_temp_reduce, max_temp_reduce_size, cudaHostAllocMapped));

        CubDebugExit(cudaMalloc((void**)&d_megabins,         sizeof(double) * (MAX_EXPANSIONS + 1) * MAX_BINS ));
        CubDebugExit(cudaMalloc((void**)&d_extreme_flags,         sizeof(ExtremeFlags)));

        CubDebugExit(cudaEventCreate(&start));
        CubDebugExit(cudaEventCreate(&stop));
    }

    ~TestExponentSort()
    {
        if (h_in) delete[] h_in;
        if (h_out) delete[] h_out;
        if (h_reference) delete[] h_reference;
        if (h_elapsed) delete[] h_elapsed;
        if (d_in) CubDebugExit(cudaFree(d_in));
        if (d_out) CubDebugExit(cudaFree(d_out));
        if (h_map_temp_reduce) cudaFreeHost(h_map_temp_reduce);

        if (h_megabins) delete[] h_megabins;
        if (d_out) CubDebugExit(cudaFree(d_megabins));
        if (h_extreme_flags) delete[] h_extreme_flags;
        if (d_extreme_flags) CubDebugExit(cudaFree(d_extreme_flags));

        CubDebugExit(cudaEventDestroy(start));
        CubDebugExit(cudaEventDestroy(stop));
    }

    void Test()
    {
        Test<ACCUSUM_SORT_REDUCE, 128, 16*16*2, 2, 2, 4, 8, GEN_RANDOM>();
    }

    template<
        int METHOD,
        int BLOCK_THREADS,
        int MIN_GRID_SIZE,
        int ITEMS_PER_THREAD,
        int EXPANSIONS,
        int RADIX_BITS,
        int MIN_CONCURRENT_BLOCKS,
        GenModeDbl GEN_MODE
    >
    float Test(unsigned num_items = (1 << 25), bool validate = false, unsigned long long seed = 1234ULL)
    {
        float kernel_time = 0.f;
        float thrpt_gdbl_sec = 0.f;

        assert(num_items <= MAX_ITEMS);

        // generate input
        if (gen_mode != GEN_MODE || _seed != seed)
        {
            InitializeExponentSort<GEN_MODE>(h_in, d_in, h_reference, MAX_ITEMS, _seed);
            gen_mode = GEN_MODE;
            _seed = seed;
        }

        if (METHOD == ACCUSUM_SORT_REDUCE)
        {
            typedef AccumulatorBinsMetadata<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS> BinMeta;
            const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
            const int grid_size = CUB_ROUND_UP_NEAREST(CUB_MAX(MIN_GRID_SIZE, CUB_QUOTIENT_CEILING(num_items, BinMeta::BIN_CAPACITY)), sm_count);
            num_items = CUB_ROUND_UP_NEAREST(num_items, TILE_SIZE * grid_size);

            // compute reference sum
            if (validate)
            {
                *h_reference = sum_mpfr(h_in, num_items);
            }

            // allocate temporary storage
            void   *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            DeviceAccurateFPSum::SumSortReduce<BLOCK_THREADS, MIN_GRID_SIZE, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS>
                (d_in, num_items, d_out, d_temp_storage, temp_storage_bytes);
            cudaDeviceSynchronize();
            CubDebugExit(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes ));

            GpuTimer timer;
            timer.Start();
            DeviceAccurateFPSum::SumSortReduce<BLOCK_THREADS, MIN_GRID_SIZE, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS>
                (d_in, num_items, d_out, d_temp_storage, temp_storage_bytes);
            cudaDeviceSynchronize();
            timer.Stop();
            CubDebugExit(cudaMemcpy(h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
            kernel_time = timer.ElapsedMillis();
            thrpt_gdbl_sec = (float)num_items / kernel_time * 1e-6f;
            if (d_temp_storage) cudaFree(d_temp_storage);
        }
        else if (METHOD == ACCUSUM_SMEM_ATOMIC)
        {
            /////////////////////
            // Test accumulation in shared mem
            // Benchmark: K40c  random data    4.0GDbl/sec
            //                  constant data  0.2GDbl/sec

            const int ACCU_TILE_SIZE = 672;
            const int NUM_BIN_COPIES_SMEM = 15;
            const int accu_grid_size = 3 * sm_count * 8;
            num_items = CUB_ROUND_UP_NEAREST(num_items, ACCU_TILE_SIZE * accu_grid_size);

            // compute reference sum
            if (validate)
            {
                *h_reference = sum_mpfr(h_in, num_items);
            }

            // allocate temporary storage
            void   *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            CubDebugExit(DeviceAccurateFPSum::SumSmemAtomic(d_in, num_items, d_out, d_temp_storage, temp_storage_bytes));
            CubDebugExit(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes ));

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

template<int _BEGIN, int _END>
struct Range {
    enum {
        BEGIN = _BEGIN,
        END = _END
    };
};

template <int TEST_NUM, typename SETUP>
float RunTest(Int2Type<TEST_NUM> test_num, SETUP setup, TestExponentSort& testobj, unsigned num_items = (1 << 25), bool validate = false, unsigned long long seed = 1234ULL)
{
    const int RANGE1 = (SETUP::WarpsPerBlock::END     - SETUP::WarpsPerBlock::BEGIN + 1)  ;
    const int RANGE2 = (SETUP::BlocksPerSm::END       - SETUP::BlocksPerSm::BEGIN + 1) ;
    const int RANGE3 = (SETUP::ItemsPerThread::END    - SETUP::ItemsPerThread::BEGIN + 1) ;
    const int RANGE4 = (SETUP::Expansions::END        - SETUP::Expansions::BEGIN + 1)     ;
    const int RANGE5 = (SETUP::RadixBits::END         - SETUP::RadixBits::BEGIN + 1)      ;
    const int RANGE6 = (SETUP::MinConcurrentBlocks::END - SETUP::MinConcurrentBlocks::BEGIN + 1)      ;
    const int RANGE7 = (SETUP::GenerationMode::END    - SETUP::GenerationMode::BEGIN + 1) ;

    const int TEMP1 = TEST_NUM;
    const int TEMP2 = TEMP1 / RANGE1;
    const int TEMP3 = TEMP2 / RANGE2;
    const int TEMP4 = TEMP3 / RANGE3;
    const int TEMP5 = TEMP4 / RANGE4;
    const int TEMP6 = TEMP5 / RANGE5;
    const int TEMP7 = TEMP6 / RANGE6;

    const int WARPS_PER_BLOCK =     (TEMP1 % RANGE1) + SETUP::WarpsPerBlock ::BEGIN;
    const int BLOCKS_PER_SM =       (TEMP2 % RANGE2) + SETUP::BlocksPerSm::BEGIN;
    const int ITEMS_PER_THREAD =    (TEMP3 % RANGE3) + SETUP::ItemsPerThread::BEGIN;
    const int EXPANSIONS =          (TEMP4 % RANGE4) + SETUP::Expansions    ::BEGIN;
    const int RADIX_BITS =          (TEMP5 % RANGE5) + SETUP::RadixBits     ::BEGIN;
    const int MIN_CONCURRENT_BLOCKS = (TEMP6 % RANGE6) + SETUP::MinConcurrentBlocks::BEGIN;
    const int GENERATION_MODE =     (TEMP7 % RANGE7) + SETUP::GenerationMode::BEGIN;

    const int BLOCK_THREADS = WARPS_PER_BLOCK * 32;
    const int GRID_SIZE = BLOCKS_PER_SM * 14;
    float thrpt_gdbl_sec = testobj.Test<SETUP::Method, BLOCK_THREADS, GRID_SIZE, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS, (GenModeDbl)GENERATION_MODE >(num_items, validate, seed);
    printf ("%5d,%9g,%5d,%5d,%5d,%5d,%5d,%5d,%10s,%5d,%5d,%8d\n",
        TEST_NUM,
        thrpt_gdbl_sec       ,
        WARPS_PER_BLOCK      ,
        BLOCKS_PER_SM        ,
        ITEMS_PER_THREAD     ,
        EXPANSIONS           ,
        RADIX_BITS           ,
        MIN_CONCURRENT_BLOCKS,
        GenModeDblNames[GENERATION_MODE],
        BLOCK_THREADS        ,
        GRID_SIZE            ,
        num_items
        );

    return thrpt_gdbl_sec;
}

template <int TEST_NUM, int NUM_TESTS, typename SETUP>
void RunTests(Int2Type<TEST_NUM> test_num, Int2Type<NUM_TESTS> num_tests, SETUP setup, TestExponentSort& testobj, unsigned num_items = (1 << 25), bool validate = false, unsigned long long seed = 1234ULL)
{
    g_test_thrpt_gdbl_sec[TEST_NUM] = RunTest(Int2Type<TEST_NUM>(), setup, testobj, num_items, validate, seed);
    RunTests(Int2Type<TEST_NUM + 1>(), num_tests, setup, testobj, num_items, validate, seed);
}

template <int NUM_TESTS, typename SETUP>
void RunTests(Int2Type<NUM_TESTS> test_num, Int2Type<NUM_TESTS> num_tests, SETUP setup, TestExponentSort& testobj, unsigned num_items = (1 << 25), bool validate = false, unsigned long long seed = 1234ULL)
{
}

template <int FIRST, int COUNT, typename SETUP>
void RunTests(TestExponentSort& testobj, unsigned num_items, bool validate, unsigned long long seed)
{
    assert(FIRST + COUNT <= (sizeof(g_test_thrpt_gdbl_sec) / sizeof(g_test_thrpt_gdbl_sec[0])) );

    printf ("%5s,%5s,%5s,%5s,%5s,%5s,%5s,%5s,%10s,%5s,%5s,%8s\n",
        "TEST_NUM",
        "thrpt_gdbl_sec",
        "WARPS_PER_BLOCK",
        "BLOCKS_PER_SM",
        "ITEMS_PER_THREAD",
        "EXPANSIONS",
        "RADIX_BITS",
        "MIN_CONCURRENT_BLOCKS",
        "GENERATION_MODE",
        "BLOCK_THREADS",
        "GRID_SIZE",
        "num_items"
        );

    memset(&g_test_thrpt_gdbl_sec[FIRST], 0, COUNT * sizeof(float));
    RunTests(Int2Type<FIRST>(), Int2Type<FIRST + COUNT>(), SETUP(), testobj, num_items, validate, seed);
    float max_thrpt = -1.f;
    int test_max_thrpt = -1;
    for (int i = FIRST; i < FIRST + COUNT; i++)
    {
        if (g_test_thrpt_gdbl_sec[i] > 0.f && g_test_thrpt_gdbl_sec[i] > max_thrpt)
        {
            max_thrpt = g_test_thrpt_gdbl_sec[i];
            test_max_thrpt = i;
        }
    }
    printf(",,,,,,,,,,,BEST_TEST,%d,MAXTHRPT=,%9g\n", test_max_thrpt, max_thrpt);
}

template <typename SETUP>
void RunTests(TestExponentSort& testobj, unsigned num_items, bool validate, unsigned long long seed)
{
    const int NUM_TESTS =
        (SETUP::WarpsPerBlock::END     - SETUP::WarpsPerBlock::BEGIN + 1)     *
        (SETUP::BlocksPerSm::END       - SETUP::BlocksPerSm::BEGIN + 1)       *
        (SETUP::ItemsPerThread::END    - SETUP::ItemsPerThread::BEGIN + 1)    *
        (SETUP::Expansions::END        - SETUP::Expansions::BEGIN + 1)        *
        (SETUP::RadixBits::END         - SETUP::RadixBits::BEGIN + 1)         *
        (SETUP::MinConcurrentBlocks::END - SETUP::MinConcurrentBlocks::BEGIN + 1) *
        (SETUP::GenerationMode::END    - SETUP::GenerationMode::BEGIN + 1);
    RunTests<0,NUM_TESTS,SETUP>(testobj, num_items, validate, seed);
}

struct SetupMax
{
    enum { Method = TestExponentSort::ACCUSUM_SORT_REDUCE };
    typedef Range<3,5>      WarpsPerBlock;       //< for BLOCK_SIZE multiple by 32
    typedef Range<2,24>     BlocksPerSm;         //< for MIN_GRID_SIZE multiple by 14
    typedef Range<1,8>      ItemsPerThread;
    typedef Range<2,8>      Expansions;
    typedef Range<2,6>      RadixBits;
    typedef Range<2,16>     MinConcurrentBlocks;    //< to be used as a parameter to __launch_bounds__
    typedef Range<1,9>      GenerationMode;
};

struct SetupDefault
{
    enum { Method = TestExponentSort::ACCUSUM_SORT_REDUCE};  /*ACCUSUM_SMEM_ATOMIC ACCUSUM_SORT_REDUCE*/
    typedef Range<4,4>      WarpsPerBlock;       //< for BLOCK_SIZE multiple by 32
    typedef Range<16,16>    BlocksPerSm;         //< for MIN_GRID_SIZE multiple by 14
    typedef Range<3,3>      ItemsPerThread;
    typedef Range<2,2>      Expansions;
    typedef Range<3,3>      RadixBits;
    typedef Range<9,9>      MinConcurrentBlocks;    //< to be used as a parameter to __launch_bounds__
    typedef Range<5,5>      GenerationMode;
};

struct SetupWarpsPerBlock : public SetupDefault
{
    typedef Range<1,6>      WarpsPerBlock;       //< for BLOCK_SIZE multiple by 32
};

struct SetupBlocksPerSm : public SetupDefault
{
    typedef Range<8,24>      BlocksPerSm;         //< for MIN_GRID_SIZE multiple by 14
};

struct SetupItemsPerThread : public SetupDefault
{
    typedef Range<2,4>      ItemsPerThread;
};

struct SetupExpansions : public SetupDefault
{
    typedef Range<2,8>      Expansions;
};

struct SetupRadixBits : public SetupDefault
{
    typedef Range<3,6>      RadixBits;
};

struct SetupMinConcurrentBlocks : public SetupDefault
{
    typedef Range<2,16>      MinConcurrentBlocks;
};

struct SetupGenMode : public SetupDefault
{
    typedef Range<1,9>      GenerationMode;
};

template<typename Setup, int METHOD>
struct SetupMethod : public Setup
{
    enum { Method = METHOD};
};

template<int NUM_ITEMS> struct SetupDefaultByNumItems {};

template<>
struct SetupDefaultByNumItems< (1 << 16) > : public SetupDefault
{
    typedef Range<8,8>    BlocksPerSm;         //< for MIN_GRID_SIZE multiple by 14
};

template<>
struct SetupDefaultByNumItems< (1 << 28) > : public SetupDefault
{
    typedef Range<24,24>    BlocksPerSm;         //< for MIN_GRID_SIZE multiple by 14
};

struct SetupCustom : public SetupDefault
{
    enum { Method = TestExponentSort::ACCUSUM_SMEM_ATOMIC };
};

void TestExponentAll()
{


    TestExponentSort testobj;

//    const int NUM_TESTS =
//        (SetupMax::WarpsPerBlock::END     - SetupMax::WarpsPerBlock::BEGIN + 1)     *
//        (SetupMax::BlocksPerSm::END       - SetupMax::BlocksPerSm::BEGIN + 1)       *
//        (SetupMax::ItemsPerThread::END    - SetupMax::ItemsPerThread::BEGIN + 1)    *
//        (SetupMax::Expansions::END        - SetupMax::Expansions::BEGIN + 1)        *
//        (SetupMax::RadixBits::END         - SetupMax::RadixBits::BEGIN + 1)         *
//        (SetupMax::MinConcurrentBlocks::END - SetupMax::MinConcurrentBlocks::BEGIN + 1) *
//        (SetupMax::GenerationMode::END    - SetupMax::GenerationMode::BEGIN + 1);

//    assert(NUM_TESTS <= (sizeof(g_test_thrpt_gdbl_sec) / sizeof(g_test_thrpt_gdbl_sec[0])) );

    int num_items = (1 << 25);
    bool validate = true;
    unsigned long long seed = 1234ULL;


//    RunTests<SetupDefaultByNumItems< 1<<16 > >(testobj, 1 << 25, validate, seed);
//    RunTests<SetupDefaultByNumItems< 1<<28 > >(testobj, 1 << 28, validate, seed);

//    RunTests<0,2,SetupMax>(testobj, num_items, validate, seed);
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

//    RunTests<SetupMethod<SetupDefault, TestExponentSort::ACCUSUM_SORT_REDUCE> >(testobj, 1<<25, true, seed);
//    RunTests<SetupMethod<SetupDefault, TestExponentSort::ACCUSUM_SMEM_ATOMIC> >(testobj, 1<<25, true, seed);

//    RunTests<SetupGenMode>(testobj, num_items, validate, seed);
    RunTests<SetupDefault>(testobj, num_items, validate, seed);
    RunTests<SetupCustom>(testobj, num_items, validate, seed);

    int num_items_options[] = {
//        1<<16,1<<17, 1<<18, 1<<19,
//        1<<20, 1<<21, 1<<22, 1<<23,
//        1<<24, 1<<25, 1<<26, 1<<27,
//        1<<28,
//        1<<29
    };
//    for (int i = 0; i < sizeof(num_items_options) / sizeof(int); i++)
//    {
//        num_items = num_items_options[i];
//        RunTests<SetupDefault>(testobj, num_items, true, seed);
////        RunTests<SetupWarpsPerBlock>(testobj, num_items, validate, seed);
////        RunTests<SetupBlocksPerSm>(testobj, num_items, validate, seed);
////        RunTests<SetupItemsPerThread>(testobj, num_items, validate, seed);
////        RunTests<SetupExpansions>(testobj, num_items, validate, seed);
////        RunTests<SetupRadixBits>(testobj, num_items, validate, seed);
////        RunTests<SetupMinConcurrentBlocks>(testobj, num_items, validate, seed);
////        RunTests<SetupGenMode>(testobj, num_items, validate, seed);
//    }

//    RunTests<SetupWarpsPerBlock>(testobj, num_items, validate, seed);
//    RunTests<SetupBlocksPerSm>(testobj, num_items, validate, seed);
//    RunTests<SetupItemsPerThread>(testobj, num_items, validate, seed);
//    RunTests<SetupExpansions>(testobj, num_items, validate, seed);
//    RunTests<SetupRadixBits>(testobj, num_items, validate, seed);
//    RunTests<SetupMinConcurrentBlocks>(testobj, num_items, validate, seed);
//    RunTests<SetupGenMode>(testobj, num_items, validate, seed);
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

    //TestFullTileDBG<BLOCK_REDUCE_RAKING,                   512, 1, 1, 8>(RANDOM, 200, CUB_TYPE_STRING(int));
    //TestFullTileDBG<BLOCK_REDUCE_RAKING,                   128, 1, 1, 4>(RANDOM, 1, CUB_TYPE_STRING(int));

//    test_block_sort();
    TestExponentAll();
//    TestExponentCustom();
    cudaDeviceReset();
    return 0;
}
