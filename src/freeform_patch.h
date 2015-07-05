#pragma once

#include <thrust/device_vector.h>

#include <sstream>

#include <math/math_vector.h>
#include <math/math_matrix.h>
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

    struct point
    {
        float x;
        float y;
    };

#if defined( __CUDACC__ )
    struct tab
    {
        uint32_t        m_index;
        math::float4    m_aabb;

        __device__ __host__ tab()
        {

        }

        __device__ __host__ tab( uint32_t index, math::float4 aabb ) :
            m_index(index)
            , m_aabb( aabb )
        {
        }
    };

    __device__ inline bool operator== (const tab& a, const tab& b)
    {
        return a.m_index == b.m_index && a.m_aabb.x == b.m_aabb.x && a.m_aabb.y == b.m_aabb.y && a.m_aabb.z == b.m_aabb.z && a.m_aabb.w == b.m_aabb.w;
    }

    typedef thrust::device_vector< patch >          patches;
    typedef thrust::device_vector< point >          points;
    typedef thrust::device_vector< tab>             tabs;


    inline std::ostream& operator<<(std::ostream& s, const patch& p)
    {
        s << "x: " << p.x0 << " " << p.x1 << " " << p.x2 << " " << p.x3 << std::endl;
        s << "y: " << p.y0 << " " << p.y1 << " " << p.y2 << " " << p.y3 << std::endl;
        return s;
    }

    inline std::ostream& operator<<(std::ostream& s, const point& p)
    {
        s << "x: " << p.x << " " << p.y << std::endl;
        return s;
    }

    inline std::ostream& operator<<(std::ostream& s, const tab& p)
    {
        s << p.m_index << "\t" << p.m_aabb.x << "\t" << p.m_aabb.y << "\t" << p.m_aabb.z << "\t" << p.m_aabb.w <<  std::endl;
        return s;
    }

    __device__ inline patch cub_bezier_interpol(patch p)
    {
        using namespace math;

        math::float4  v0 = math::identity_r0();
        math::float4  v1 = set(-5.0f / 6.0f, 3.0f, -3.0f / 2.0f, 1.0f / 3.0f);
        math::float4  v2 = swizzle<w, z, y, x>(v1);
        math::float4  v3 = math::identity_r3();

        math::float4x4 m = math::set(v0, v1, v2, v3);

        math::float4  p0_x = set(p.x0, p.x1, p.x2, p.x3);
        math::float4  p0_y = set(p.y0, p.y1, p.y2, p.y3);

        math::float4  x = mul(m, p0_x);
        math::float4  y = mul(m, p0_y);

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
            math::float4  x4 = math::set(p.x0, p.x1, p.x2, p.x3);
            math::float4  y4 = math::set(p.y0, p.y1, p.y2, p.y3);

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

    __device__ inline float min4(float x0, float x1, float x2, float x3)
    {
        auto x = min(x0, x1);

        x = min(x, x2);
        x = min(x, x3);
        return x;
    }

    __device__ inline float max4(float x0, float x1, float x2, float x3)
    {
        auto x = max(x0, x1);

        x = max(x, x2);
        x = max(x, x3);
        return x;
    }

    __device__ inline bool intersect_bounding_boxes(math::float4 a, math::float4 b )
    {
        //a and b contain: min_x, max_x, min_y, max_y for an aabb

        float x1 = math::get_x(a);
        float y1 = math::get_z(a);

        float x2 = math::get_y(a);
        float y2 = math::get_w(a);

        float x3 = math::get_x(b);
        float y3 = math::get_z(b);

        float x4 = math::get_y(b);
        float y4 = math::get_w(b);

        if (x2 < x3)
        {
            return false;
        }
        
        if ((x2 == x3) && (y2 < y3 || y4 < y1))
        {
            return false;
        }

        if (x4 < x1)
        {
            return false;
        }

        if ((x4 == x1) && (y4 < y1 || y2 < y3))
        {
            return false;
        }

        return true;
    }
#endif

}
