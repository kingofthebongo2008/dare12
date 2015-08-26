#pragma once

#include "cuda_patches.h"
#include <math/math_vector.h>

namespace freeform
{
    struct gather_samples_kernel
    {
        __device__ patch operator() (const thrust::tuple< point, point, point, point, point, point, point, point, patch > & pt)
        {
            sample s;

            //resampled points
            point r0 = thrust::get<0>(pt);
            point r1 = thrust::get<1>(pt);
            point r2 = thrust::get<2>(pt);
            point r3 = thrust::get<3>(pt);

            point p0 = thrust::get<4>(pt);
            point p1 = thrust::get<5>(pt);
            point p2 = thrust::get<6>(pt);
            point p3 = thrust::get<7>(pt);

            //form delta of moved points
            s.x0 = r0.x - p0.x;
            s.x1 = r1.x - p1.x;
            s.x2 = r2.x - p2.x;
            s.x3 = r3.x - p3.x;

            s.y0 = r0.y - p0.y;
            s.y1 = r1.y - p1.y;
            s.y2 = r2.y - p2.y;
            s.y3 = r3.y - p3.y;

            //obtain delta of moved control points
            patch r = interpolate_curve(s);

            patch p = thrust::get<8>(pt);

            patch res;

            res.x0 = p.x0 + r.x0;
            res.x1 = p.x1 + r.x1;
            res.x2 = p.x2 + r.x2;
            res.x3 = p.x3 + r.x3;

            res.y0 = p.y0 + r.y0;
            res.y1 = p.y1 + r.y1;
            res.y2 = p.y2 + r.y2;
            res.y3 = p.y3 + r.y3;


            return res;
        }
    };
}