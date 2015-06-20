#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>


inline std::ostream& operator<<(std::ostream& s, const float4& p)
{
    s << "x: " << p.x << " " << p.y << " " << p.z << " " << p.w << std::endl;
    return s;
}

namespace freeform
{
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

        __device__ thrust::tuple< freeform::patch, freeform::patch, math::float4 > operator() (uint32_t i) const
        {
            float x0 = x(3 * i, 0);
            float x1 = x(3 * i, 1);
            float x2 = x(3 * i, 2);
            float x3 = x(3 * i, 3);

            float y0 = y(3 * i, 0);
            float y1 = y(3 * i, 1);
            float y2 = y(3 * i, 2);
            float y3 = y(3 * i, 3);

            freeform::patch p0 = { x0, x1, x2, x3, y0, y1, y2, y3 };
            freeform::patch p1 = cub_bezier_interpol( p0 );


            float min_0 = min4(p1.x0, p1.x1, p1.x2, p1.x3);
            float min_1 = min4(p1.y0, p1.y1, p1.y2, p1.y3);
            float max_0 = max4(p1.x0, p1.x1, p1.x2, p1.x3);
            float max_1 = max4(p1.y0, p1.y1, p1.y2, p1.y3);


            float4  tab = math::set(min_0, max_0, min_1, max_1 );

            return thrust::make_tuple ( p0, p1, tab );
        }
    };

    thrust::tuple< patches, patches, thrust::device_vector<math::float4> > inititialize_free_form(uint32_t center_image_x, uint32_t center_image_y, float radius, uint32_t patch_count)
    {
        thrust::device_vector<float> x;
        thrust::device_vector<float> y;

        thrust::device_vector<freeform::patch> n;
        thrust::device_vector<freeform::patch> np;
        thrust::device_vector<math::float4>    tabs;

        auto pi = 3.1415926535f;
        auto pas = 2 * pi / patch_count;
        auto pas_pt_patch = pas / 3.0f;

        auto iterations = static_cast<uint32_t> (ceilf(2 * pi / pas_pt_patch));

    
        n.resize( iterations / 3 );
        np.resize(iterations / 3);
        tabs.resize(iterations / 3);


        auto begin  = thrust::make_counting_iterator(0);
        auto end    = begin + iterations / 3;
        auto o      = thrust::make_zip_iterator(thrust::make_tuple(n.begin(), np.begin(), tabs.begin()));

        thrust::transform(begin, end, o, generate_patch(static_cast<float> (center_image_x), static_cast<float> (center_image_y), radius, pas_pt_patch));


        return std::move(thrust::make_tuple(std::move(n), std::move(np), std::move(tabs)));
    }
}


