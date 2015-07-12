#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>

#include <math/math_vector.h>

#include "math_functions.h"


#include <algorithm>

namespace freeform
{
    struct multi_eval_patches2_kernel
    {
        float                          m_seuil;
        thrust::device_ptr<patch>      m_patches;
        thrust::device_ptr<uint32_t>   m_element_count;

        multi_eval_patches2_kernel(float seuil, thrust::device_ptr<patch> patches, thrust::device_ptr<uint32_t> element_count) :
            m_seuil(seuil)
            , m_patches(patches)
            , m_element_count(element_count)
        {

        }

        //split procedure
        __device__ void operator() (const patch & p) const
        {
            auto d_0 = math::distance(p.x0, p.y0, p.x1, p.y1);
            auto d_1 = math::distance(p.x1, p.y1, p.x2, p.y2);
            auto d_2 = math::distance(p.x2, p.y2, p.x3, p.y3);

            auto m = fmax(fmax(d_0, d_1), d_2);

            if (m > m_seuil)
            {
                float4 g1u = math::set(0.0f, 1.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f);
                float4 g2u = math::set(3.0f / 6.0f, 4.0f / 6.0f, 5.0f / 6.0f, 6.0f / 6.0f);

                auto   g1 = multi_eval_patch_3(p, g1u);
                auto   g2 = multi_eval_patch_3(p, g2u);

                auto   g4 = interpolate_curve(g1);
                auto   g5 = interpolate_curve(g2);

                auto old = atomicAdd(m_element_count.get(), 2);

                m_patches[old] = g4;
                m_patches[old + 1] = g5;
            }
            else
            {
                auto old = atomicAdd(m_element_count.get(), 1);
                m_patches[old] = p;
            }
        }
    };

    //thrust::copy(g.begin(), g.end(), std::ostream_iterator< float >(std::cout, " "));

    //split procedure, generates new patches if they are too close
    patches split(const patches& p)
    {
        patches                      n2;
        thrust::device_vector<uint32_t> element_count;
        thrust::device_vector<uint32_t> host_element_count;

        auto maxi = p.size();

        n2.resize( p.size() * 2 );

        element_count.resize(1);
        host_element_count.resize(1);

        auto b = p.begin();
        auto e = p.end();

        thrust::for_each(b, e, multi_eval_patches2_kernel(45, &n2[0], &element_count[0]));

        auto elements = element_count.front();

        thrust::copy_n(element_count.begin(), 1, host_element_count.begin());

        uint32_t new_size = host_element_count[0];
        n2.resize(new_size);
        return n2;
    }
}


