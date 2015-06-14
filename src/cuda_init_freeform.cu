#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>

#include "math_vector.h"
#include "math_matrix.h"



namespace freeform
{
    __device__ patch cub_bezier_interpol(patch p)
    {
        using namespace math;

        float4  v0 = math::identity_r0();
        float4  v1 = set( -5.0f / 6.0f  , 3.0f, -3.0f / 2.0f, 1.0f / 3.0f );
        float4  v2 = swizzle<w, z, y, x> ( v1 );
        float4  v3 = math::identity_r3();

        float4x4 m = set(v0, v1, v2, v3);

        float4  p0_x = set(p.x0, p.x1, p.x2, p.x3);
        float4  p0_y = set(p.y0, p.y1, p.y2, p.y3);

        float4  x = mul( m, p0_x);
        float4  y = mul( m, p0_y);

        patch   r = { 
                        math::get_x(x), math::get_y(x), math::get_z(x), math::get_w(x),
                        math::get_x(y), math::get_y(y), math::get_z(y), math::get_w(y),
                    };

        return r;
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

        __device__ thrust::tuple< freeform::patch, freeform::patch > operator() (uint32_t i) const
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

            return thrust::make_tuple ( p0, p1 );
        }
    };

    void inititialize_free_form( uint32_t center_image_x, uint32_t center_image_y, float radius, uint32_t patch_count )
    {
        thrust::device_vector<float> x;
        thrust::device_vector<float> y;

        thrust::device_vector<freeform::patch> patches;
        thrust::device_vector<freeform::patch> patches_n;

        auto pi = 3.1415926535f;
        auto pas = 2 * pi / patch_count;
        auto pas_pt_patch = pas / 3.0f;

        auto iterations = static_cast<uint32_t> (ceilf(2 * pi / pas_pt_patch));

    
        patches.resize( iterations / 3 );
        patches_n.resize(iterations / 3);


        auto begin  = thrust::make_counting_iterator(0);
        auto end    = begin + iterations / 3;
        auto o      = thrust::make_zip_iterator(thrust::make_tuple(patches.begin(), patches_n.begin()));

        thrust::transform(begin, end, o, generate_patch(static_cast<float> (center_image_x), static_cast<float> (center_image_y), radius, pas_pt_patch));


        thrust::host_vector<freeform::patch> r;

        r.resize(iterations / 3 );

        thrust::copy(patches_n.begin(), patches_n.end(), r.begin());
        thrust::copy(r.begin(), r.end(), std::ostream_iterator< freeform::patch >(std::cout, " "));
    }
}


