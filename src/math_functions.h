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

    //decasteljau algorith for cubic bezier, for evaluating the curves
    __device__ inline float decasteljau_3(float4 points, float t)
    {
        auto b0_0 = lerp( math::get_y(points), math::get_x(points), t);
        auto b0_1 = lerp( math::get_z(points), math::get_y(points), t);
        auto b0_2 = lerp( math::get_w(points), math::get_z(points), t);

        auto b1_0 = lerp(b0_1, b0_0, t);
        auto b1_1 = lerp(b0_2, b0_1, t);

        auto b2_0 = lerp(b1_1, b1_0, t);

        return b2_0;
    }

    //decasteljau algorith for quadratic bezier, for evaluating the derivatives
    __device__ inline float decasteljau_2(float4 points, float t)
    {
        auto b0_0 = math::get_x(points);
        auto b0_1 = math::get_y(points);
        auto b0_2 = math::get_z(points);

        auto b1_0 = lerp(b0_1, b0_0, t);
        auto b1_1 = lerp(b0_2, b0_1, t);

        auto b2_0 = lerp(b1_1, b1_0, t);

        return b2_0;
    }

    //returns the control points of the quadratic curve, which is a derivative of a cubic curve
    __device__ inline float4 cubic_bezier_derivative( float4 points )
    {
        //make vector p1, p2, p3
        auto a = swizzle<y, z, w, w >(points);
        auto m = math::set(3.0f, 3.0f, 3.0f, 0.0f);

        //make curve which is

        // 3 * ( p1 - p0), 3 * ( p2 - p1),3 * ( p3 - p2)
        auto r = math::mul(m, math::sub(a, points));

        //.w is undefined

        return r;
    }
}
