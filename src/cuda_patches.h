#pragma once

#include <cstdint>
#include <thrust/device_ptr.h>

#include "freeform_patch.h"

namespace freeform
{
    struct average_patches
    {
        thrust::device_ptr< patch > m_patches_in;
        thrust::device_ptr< patch > m_patches_out;
        uint32_t                    m_count;

        average_patches(thrust::device_ptr<patch> patches_in, thrust::device_ptr<patch> patches_out, uint32_t count) :
            m_patches_in(patches_in)
            , m_patches_out(patches_out)
            , m_count(count)
        {

        }

        __device__ void operator() (uint32_t i) const
        {
            auto i0 = i;
            auto i1 = i + 1;


            if (i1 == m_count)
            {
                i1 = 0;
            }

            patch p0 = m_patches_in[i0];
            patch p1 = m_patches_in[i1];

            patch r0 = p0;
            patch r1 = p1;

            r0.x3 = r1.x0;
            r0.y3 = r1.y0;

            //copy the modified patch to stick the control points together
            m_patches_out[i0] = r0;
        }
    };

    /*
    patches n3;
    n3.resize(n2.size());
    {
    auto b = make_counting_iterator(0);
    auto e = b + n2.size();
    thrust::for_each(b, e, average_patches(&n2[0], &n3[0], n2.size()));
    }
    */
}
