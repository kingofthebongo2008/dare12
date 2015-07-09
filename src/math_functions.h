#pragma once

#include <math/math_vector.h>

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

    __device__ inline float decasteljau(float4 points, float t)
    {
        auto b0_0 = lerp( math::get_y(points), math::get_x(points), t);
        auto b0_1 = lerp( math::get_z(points), math::get_y(points), t);
        auto b0_2 = lerp( math::get_w(points), math::get_z(points), t);

        auto b1_0 = lerp(b0_1, b0_0, t);
        auto b1_1 = lerp(b0_2, b0_1, t);

        auto b2_0 = lerp(b1_1, b1_0, t);

        return b2_0;
    }
}
