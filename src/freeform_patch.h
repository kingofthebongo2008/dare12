#pragma once

#include <thrust/device_vector.h>

#include <sstream>

#include "math_vector.h"
#include "math_matrix.h"
#include "math_functions.h"

namespace freeform
{
    struct patch
    {
        float x0;
        float x1;
        float x2;
        float x3;

        float y0;
        float y1;
        float y2;
        float y3;

    };

    typedef thrust::device_vector< patch > patches;


    inline std::ostream& operator<<(std::ostream& s, const patch& p)
    {
        s << "x: " << p.x0 << " " << p.x1 << " " << p.x2 << " " << p.x3 << std::endl;
        s << "y: " << p.y0 << " " << p.y1 << " " << p.y2 << " " << p.y3 << std::endl;
        return s;
    }


    __device__ inline patch cub_bezier_interpol(patch p)
    {
        using namespace math;

        float4  v0 = math::identity_r0();
        float4  v1 = set(-5.0f / 6.0f, 3.0f, -3.0f / 2.0f, 1.0f / 3.0f);
        float4  v2 = swizzle<w, z, y, x>(v1);
        float4  v3 = math::identity_r3();

        float4x4 m = set(v0, v1, v2, v3);

        float4  p0_x = set(p.x0, p.x1, p.x2, p.x3);
        float4  p0_y = set(p.y0, p.y1, p.y2, p.y3);

        float4  x = mul(m, p0_x);
        float4  y = mul(m, p0_y);

        patch   r = {
            math::get_x(x), math::get_y(x), math::get_z(x), math::get_w(x),
            math::get_x(y), math::get_y(y), math::get_z(y), math::get_w(y),
        };

        return r;
    }

    __device__ inline float2 eval_patches_3(patch p, float t)
    {
        auto ncp = 4;
        auto n = ncp / 4.0f;

        auto k = floorf(n * t);

        if (t == 1.0f)
        {
            return make_float2(p.x3, p.y3);
        }
        else if (k == n * t)
        {
            return make_float2(p.x0, p.y0);
        }
        else
        {
            float4  x4 = math::set(p.x0, p.x1, p.x2, p.x3);
            float4  y4 = math::set(p.y0, p.y1, p.y2, p.y3);

            auto x = math::decaste_casteljau(x4, n * t - k);
            auto y = math::decaste_casteljau(y4, n * t - k);
            return make_float2(x, y);
        }
    }

    __device__ inline patch multi_eval_patch(patch p, float4 u)
    {

        auto r0 = eval_patches_3(p, u.x);
        auto r1 = eval_patches_3(p, u.y);
        auto r2 = eval_patches_3(p, u.z);
        auto r3 = eval_patches_3(p, u.w);

        patch r = { r0.x, r1.x, r2.x, r3.x, r0.y, r1.y, r2.y, r3.y };

        return r;
    }
}
