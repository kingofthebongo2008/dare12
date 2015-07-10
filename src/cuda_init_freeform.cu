#include "precompiled.h"

#include <tuple>
#include <thrust/transform.h>

#include "freeform_patch.h"






inline std::ostream& operator<<(std::ostream& s, const float4& p)
{
    s << "x: " << p.x << " " << p.y << " " << p.z << " " << p.w << std::endl;
    return s;
}

namespace freeform
{
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

        __device__ float sample_x(uint32_t i, uint32_t step) const
        {
            return m_center_x + m_radius * cosf((i + step) * m_step);
        }

        __device__ float sample_y(uint32_t i, uint32_t step) const
        {
            return m_center_y + m_radius * sinf((i + step) * m_step);
        }

        __device__ thrust::tuple< sample, patch, tab > operator() (uint32_t i) const
        {
            float x0 = sample_x(3 * i, 0);
            float x1 = sample_x(3 * i, 1);
            float x2 = sample_x(3 * i, 2);
            float x3 = sample_x(3 * i, 3);

            float y0 = sample_y(3 * i, 0);
            float y1 = sample_y(3 * i, 1);
            float y2 = sample_y(3 * i, 2);
            float y3 = sample_y(3 * i, 3);

            freeform::sample p0 = { x0, x1, x2, x3, y0, y1, y2, y3 };

            //obtain patch control points from sampled points
            auto p1 = interpolate_curve( p0 );

            float min_x = min4(p1.x0, p1.x1, p1.x2, p1.x3);
            float min_y = min4(p1.y0, p1.y1, p1.y2, p1.y3);
            float max_x = max4(p1.x0, p1.x1, p1.x2, p1.x3);
            float max_y = max4(p1.y0, p1.y1, p1.y2, p1.y3);

            float4  tb = math::set(min_x, max_x, min_y, max_y );
            
            tab     t(i, tb);


            return thrust::make_tuple ( p0, p1, t );
        }
    };

    //sample the curve and obtain patches through curve interpolation as in the paper
    std::tuple< samples, patches  > inititialize_free_form(uint32_t center_image_x, uint32_t center_image_y, float radius, uint32_t patch_count)
    {
        thrust::device_vector<float> x;
        thrust::device_vector<float> y;

        samples n;
        patches np;
        tabs    tabs;

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
        return std::move(std::make_tuple(std::move(n), std::move(np) )); 
    }
}


