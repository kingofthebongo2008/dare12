#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>
    struct generate_patch
    {
        float m_center_x;
        float m_center_y;
        float m_radius;
        float m_step;

        generate_patch(float center_x, float center_y, float radius, float step) :
            m_center_x(center_x)
            , m_center_y(center_y)
            , m_radius(radius)
            , m_step(step)
        {

        }

        __device__ float x(uint32_t i, uint32_t step) const
        {
            return m_center_x + m_radius * cosf((i + step) * m_step);
        }

        __device__ float y(uint32_t i, uint32_t step) const
        {
            return m_center_y + m_radius * sinf((i + step) * m_step);
        }

        __device__ freeform::patch operator() (uint32_t i) const
        {
            float x0 = x(3 * i, 0);
            float x1 = x(3 * i, 1);
            float x2 = x(3 * i, 2);
            float x3 = x(3 * i, 3);

            float y0 = y(3 * i, 0);
            float y1 = y(3 * i, 1);
            float y2 = y(3 * i, 2);
            float y3 = y(3 * i, 3);


            freeform::patch p = { x0, x1, x2, x3, y0, y1, y2, y3 };

            return p;
        }
    };

namespace cuda
{
    void inititialize_free_form( uint32_t center_image_x, uint32_t center_image_y, float radius, uint32_t patch_count )
    {
        thrust::device_vector<float> x;
        thrust::device_vector<float> y;

        thrust::device_vector<freeform::patch> patches;

        auto pi = 3.1415926535f;
        auto pas = 2 * pi / patch_count;
        auto pas_pt_patch = pas / 3.0f;

        auto iterations = static_cast<uint32_t> (ceilf(2 * pi / pas_pt_patch));

    
        patches.resize( iterations / 3 );


        auto begin  = thrust::make_counting_iterator(0);
        auto end    = begin + iterations / 3;

        thrust::transform(begin, end, patches.begin(), generate_patch(static_cast<float> (center_image_x), static_cast<float> (center_image_y), radius, pas_pt_patch));


        thrust::host_vector<freeform::patch> r;
        r.resize(iterations / 3 );

        thrust::copy(patches.begin(), patches.end(), r.begin());
        thrust::copy(r.begin(), r.end(), std::ostream_iterator< freeform::patch >(std::cout, " "));
    }
}


