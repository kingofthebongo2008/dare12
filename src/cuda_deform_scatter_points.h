#pragma once

#include "cuda_patches.h"

#include <math/math_vector.h>

namespace freeform
{
    struct scatter_points_kernel
    {
        //scatter the samples into different points, so we can get them more parallel
        template <typename t> __device__ void operator()(t& tp)
        {
            patch p = thrust::get < 0 >(tp);

            float4 t = math::set(0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 3.0f);
            sample s = multi_eval_patch_3(p, t);             //sample the bezier curve

            point p0;
            point p1;
            point p2;
            point p3;

            p0.x = s.x0;
            p1.x = s.x1;
            p2.x = s.x2;
            p3.x = s.x3;

            p0.y = s.y0;
            p1.y = s.y1;
            p2.y = s.y2;
            p3.y = s.y3;

            thrust::get<1>(tp) = p0;
            thrust::get<2>(tp) = p1;
            thrust::get<3>(tp) = p2;
            thrust::get<4>(tp) = p3;
        }
    };
}