#pragma once

#include <math/math_vector.h>
#include <math/math_functions.h>

#include "cuda_patches.h"

namespace freeform
{
    __device__ inline float4 make_x(const patch& p)
    {
        return math::set(p.x0, p.x1, p.x2, p.x3);
    }

    __device__ inline float4 make_y(const patch& p)
    {
        return math::set(p.y0, p.y1, p.y2, p.y3);
    }

    __device__ __host__ inline point add(point a, point b)
    {
        point c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        return c;
    }

    __device__ __host__ inline point sub(point a, point b)
    {
        point c;

        c.x = a.x - b.x;
        c.y = a.y - b.y;
        return c;
    }

    __device__ __host__ inline point mul(point a, point b)
    {
        point c;

        c.x = a.x * b.x;
        c.y = a.y * b.y;
        return c;
    }


    __device__ __host__ inline point mul(float s, point b)
    {
        point c;

        c.x = s * b.x;
        c.y = s * b.y;
        return c;
    }

    __device__ __host__ inline point mad(point a, point b, point c)
    {
        point d;

        d.x = a.x * b.x + c.x;
        d.y = a.y * b.y + c.y;
        return d;
    }

    __device__ __host__ inline point normalize(point a)
    {
        float magnitude = a.x * a.x + a.y * a.y;
        float s = 1.0f / sqrtf(magnitude);

        return mul(s, a);
    }
}
