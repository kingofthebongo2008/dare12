#include "precompiled.h"

#include <thrust/transform.h>
#include <thrust/merge.h>

#include "imaging_utils.h"
#include "cuda_imaging.h"

#include "freeform_patch.h"
#include <math_functions.h>

#include "math_functions.h"


namespace freeform
{
    typedef thrust::device_vector<point> points;

    struct to_points_kernel
    {
        thrust::device_ptr<point> m_pt;

        to_points_kernel(thrust::device_ptr<point> pt) : m_pt(pt)
        {

        }

        __device__ void operator() (const thrust::tuple< patch, uint32_t > & t) const
        {
            auto p = thrust::get<0>(t);
            auto i = thrust::get<1>(t);

            point* pt = m_pt.get();

            pt[4 * i].x = p.x0;
            pt[4 * i].y = p.y0;

            pt[4 * i + 1].x = p.x1;
            pt[4 * i + 1].y = p.y1;

            pt[4 * i + 2].x = p.x2;
            pt[4 * i + 2].y = p.y2;

            pt[4 * i + 3].x = p.x3;
            pt[4 * i + 3].y = p.y3;
        }
    };

    static inline points to_points(const patches& p)
    {
        points pt;

        pt.resize( p.size() * 4 );

        auto cb = thrust::make_counting_iterator(0);
        auto ce = cb + p.size();

        auto b = thrust::make_zip_iterator(thrust::make_tuple(p.begin(), cb));
        auto e = thrust::make_zip_iterator(thrust::make_tuple(p.end(), ce));

        thrust::for_each(b, e, to_points_kernel(&pt[0]));

        return std::move(pt);
    }

    //thrust::copy(g.begin(), g.end(), std::ostream_iterator< float >(std::cout, " "));

    //displacement of the Bezier points in the normal direction ( if the points are not on an edge ), otherwise the are stopped

    __device__ inline void voisinage(float x, float y, int32_t v1[], int32_t v2[])
    {
        int32_t x1 = static_cast<int32_t> (floorf(x));  //todo cast to int
        int32_t y1 = static_cast<int32_t> (floorf(y));  //todo cast to int

        //% Returns the "pixelique" coordinates of the point neighborhood( its size is 9 * 9 )? 8x8?

        const int32_t indices_v1[8] = { -1, -1, -1, 0, 0, +1, +1, +1 };
        const int32_t indices_v2[8] = { -1, 0, 1, -1, 1, -1, 0, +1 };

        for (uint32_t i = 0; i < 8; ++i)
        {
            v1[i] = x1 + indices_v1[i];
        }

        for (uint32_t i = 0; i < 8; ++i)
        {
            v2[i] = y1 + indices_v2[i];
        }
    }

    
    struct displace_points_kernel
    {
        const  cuda::image_kernel_info  m_sampler;
        const   uint8_t*                m_grad;

        thrust::device_ptr< patch >     m_nor;

        //out
        thrust::device_ptr< uint8_t>    m_stop;

        displace_points_kernel( const cuda::image_kernel_info& sampler , const uint8_t* grad) : m_sampler(sampler), m_grad(grad)
        {

        }

        template <typename tuple>
        __device__ void operator() ( tuple t ) //thrust::tuple<point, point, point, uint8_t > t) 
        {
            using namespace cuda;
            auto m      = thrust::get<0>(t);
            auto nor    = thrust::get<1>(t);

            int32_t     x_offset[8];
            int32_t     y_offset[8];

            voisinage(m.x, m.y, x_offset, y_offset);

            auto s0 = sample_2d<uint8_t, border_type::clamp>(m_grad, m_sampler, x_offset[0], y_offset[0]);
            auto s1 = sample_2d<uint8_t, border_type::clamp>(m_grad, m_sampler, x_offset[1], y_offset[1]);
            auto s2 = sample_2d<uint8_t, border_type::clamp>(m_grad, m_sampler, x_offset[2], y_offset[2]);
            auto s3 = sample_2d<uint8_t, border_type::clamp>(m_grad, m_sampler, x_offset[3], y_offset[3]);

            auto s4 = sample_2d<uint8_t, border_type::clamp>(m_grad, m_sampler, x_offset[4], y_offset[4]);
            auto s5 = sample_2d<uint8_t, border_type::clamp>(m_grad, m_sampler, x_offset[5], y_offset[5]);
            auto s6 = sample_2d<uint8_t, border_type::clamp>(m_grad, m_sampler, x_offset[6], y_offset[6]);
            auto s7 = sample_2d<uint8_t, border_type::clamp>(m_grad, m_sampler, x_offset[7], y_offset[7]);

            auto p0 = *s0;
            auto p1 = *s1;
            auto p2 = *s2;
            auto p3 = *s3;

            auto p4 = *s4;
            auto p5 = *s5;
            auto p6 = *s6;
            auto p7 = *s7;
            
            auto mx = max(p0, p1);
            mx = max(mx, p2);
            mx = max(mx, p3);
            mx = max(mx, p4);
            mx = max(mx, p5);
            mx = max(mx, p6);
            mx = max(mx, p7);
            
            //if we are on an edge or we are approaching the image boundary
            if (mx == 1 || m.x < 4 || m.y < 4 || m.x >(m_sampler.width() - 4) || m.y >(m_sampler.height() - 4))
            {
                thrust::get<2>(t) = m;
                thrust::get<3>(t) = 1;
            }
            else
            {
                thrust::get<2>(t) = nor;
                thrust::get<3>(t) = 0;
            }
        }
    };
    
    thrust::tuple<points, thrust::device_vector<uint8_t> >  displace_points(const patches& m, const patches& nor, const imaging::cuda_texture& grad )
    {
        using namespace cuda;
        
        auto pt_m = to_points(m);
        auto pt_nor = to_points(nor);

        points pt_n;
        thrust::device_vector<uint8_t> stop;

        pt_n.resize( m.size() * 4 );
        stop.resize( m.size() * 4 );

        auto b = thrust::make_zip_iterator(thrust::make_tuple(pt_m.begin(), pt_nor.begin(), pt_n.begin(), stop.begin()));
        auto e = thrust::make_zip_iterator(thrust::make_tuple(pt_m.end(), pt_nor.end(), pt_n.end(), stop.end()));
        
        auto info = create_image_kernel_info(grad);
        
        thrust::for_each(b, e, displace_points_kernel( info, grad.get_gpu_pixels() ));

        /*
        thrust::copy(pt_n.begin(), pt_n.end(), std::ostream_iterator< point >(std::cout, " "));
        thrust::copy(stop.begin(), stop.end(), std::ostream_iterator< uint8_t >(std::cout, " "));
        */
        
        return thrust::make_tuple(pt_n, stop);
    }
}


