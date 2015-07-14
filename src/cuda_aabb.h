#pragma once
#include "math_functions.h"

namespace freeform
{
    struct aabb
    {
        float m_min_x;
        float m_max_x;

        float m_min_y;
        float m_max_y;

        __device__ __host__ aabb()
        {

        }
    };

    __device__ inline aabb make_aabb(const patch& p)
    {
        aabb r;

        r.m_min_x = min4(p.x0, p.x1, p.x2, p.x3);
        r.m_max_x = max4(p.x0, p.x1, p.x2, p.x3);

        r.m_min_y = min4(p.y0, p.y1, p.y2, p.y3);
        r.m_max_y = max4(p.y0, p.y1, p.y2, p.y3);

        return r;
    }

    struct lexicographical_sorter
    {
        __device__ bool operator()(const patch& p0, const patch& p1) const
        {
            aabb a0 = make_aabb(p0);
            aabb b0 = make_aabb(p1);

            float4 a = math::set(a0.m_min_x, a0.m_min_y, a0.m_max_x, a0.m_max_y);
            float4 b = math::set(b0.m_min_x, b0.m_min_y, b0.m_max_x, b0.m_max_y);

            return  a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && (a.z < b.z || (a.z == b.z && a.w < b.w)))));
        }
    };
}
