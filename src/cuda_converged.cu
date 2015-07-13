#include "precompiled.h"
#include <cstdint>
#include <memory>
#include <algorithm>

#include "freeform_patch.h"

#include <thrust/transform_reduce.h>

namespace freeform
{
    struct norm_patches
    {
        __host__ __device__ float operator()( const thrust::tuple<patch, patch>& t ) const
        {
            auto p0 = thrust::get<0>(t);
            auto p1 = thrust::get<1>(t);

            auto x0 = abs(p0.x0 - p1.x0);
            auto x1 = abs(p0.x1 - p1.x1);
            auto x2 = abs(p0.x2 - p1.x2);
            auto x3 = abs(p0.x3 - p1.x3);

            auto y0 = abs(p0.y0 - p1.y0);
            auto y1 = abs(p0.y1 - p1.y1);
            auto y2 = abs(p0.y2 - p1.y2);
            auto y3 = abs(p0.y3 - p1.y3);

            auto m = max(x0, x1);
            m = max(m, x2);
            m = max(m, x3);

            auto n = max(y0, y1);
            n = max(n, y2);
            n = max(n, y3);

            return max(m, n);
        }
    };

    uint32_t iterations = 0;
    static uint32_t max_iterations = 160;
    bool converged( thrust::device_vector<uint32_t>& stop )
    {
        using namespace thrust;
        auto norm = reduce(stop.begin(), stop.end(), 0, plus<uint32_t>());

        float percent = static_cast<float>(norm) / static_cast<float>(stop.size());
        auto r = percent > 0.99f;
        
        if (stop.size() > 2000)
        {
            return true;
        }

        if (iterations++ < max_iterations)
        {
            return r;
        }
        else
        {
            return true;
        }

    }
}


