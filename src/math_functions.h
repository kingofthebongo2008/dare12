#pragma once

#include "math_vector.h"

namespace math
{
    __device__ __host__ inline float distance( float x1, float y1, float x2, float y2 )
    {
        //calcul la distance entre deux points.
        auto x = (x2 - x1) * (x2 - x1);
        auto y = (y2 - y1) * (y2 - y1);

        return sqrtf( x + y );
    }


    __device__  inline float lerp(float a, float b, float t)
    {
        return a * t + (1 - t) * b;
    }

    __device__ inline float decaste_casteljau(float4 points, float t)
    {
        auto b0_0 = lerp(points.y, points.x, t);
        auto b0_1 = lerp(points.z, points.y, t);
        auto b0_2 = lerp(points.w, points.z, t);

        auto b1_0 = lerp(b0_1, b0_0, t);
        auto b1_1 = lerp(b0_2, b0_1, t);

        auto b2_0 = lerp(b1_1, b1_0, t);

        return b2_0;
    }
}
