#pragma once

#include "cuda_patches.h"
#include "cuda_points.h"

#include <math/math_vector.h>

namespace freeform
{
    //ensure c0 continuity of the contour
    struct average_points_kernel
    {
        thrust::device_ptr< point >     m_points_in;
        thrust::device_ptr< point >     m_points_out;
        uint32_t                        m_count;


        average_points_kernel(thrust::device_ptr<point> points_in, thrust::device_ptr<point> points_out, uint32_t count) :
            m_points_in(points_in)
            , m_points_out(points_out)
            , m_count(count)
        {

        }

        __device__ void operator() (uint32_t i) const
        {

            auto i0 = 4 * i + 3;
            auto i1 = 4 * i + 4;

            if (i1 == 4 * m_count)
            {
                i1 = 0;
            }

            point n0 = m_points_in[i0];
            point n1 = m_points_in[i1];

            point n = mul(0.5, add(n0, n1));

            m_points_out[i0] = n;
            m_points_out[i1] = n;

        }
    };
}