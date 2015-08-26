#pragma once

#include "cuda_patches.h"

#include <math/math_vector.h>

namespace freeform
{
    struct scatter_normals_kernel
    {
        //scatter the samples into different points, so we can get them more parallel
        template <typename t> __device__ void operator()(t& t)
        {
            sample p = thrust::get < 0 >(t);

            point p0;
            point p1;
            point p2;
            point p3;

            p0.x = p.x0;
            p1.x = p.x1;
            p2.x = p.x2;
            p3.x = p.x3;

            p0.y = p.y0;
            p1.y = p.y1;
            p2.y = p.y2;
            p3.y = p.y3;

            thrust::get<1>(t) = p0;
            thrust::get<2>(t) = p1;
            thrust::get<3>(t) = p2;
            thrust::get<4>(t) = p3;
        }
    };
}