#include "precompiled.h"

#include <algorithm>
#include <thrust/transform.h>
#include <thrust/sort.h>


#include <math/math_vector.h>
#include "freeform_patch.h"


#include "math_functions.h"


#include "cuda_imaging.h"
#include "cuda_strided_range.h"
#include "cuda_print_utils.h"
#include "cuda_aabb.h"
#include "cuda_patches.h"
#include "cuda_points.h"

#include "cuda_deform_scatter_normals.h"
#include "cuda_deform_scatter_points.h"
#include "cuda_deform_stitch_patches.h"
#include "cuda_deform_gather_samples.h"
#include "cuda_deform_normal_curve_points.h"


namespace freeform
{
    __device__ inline sample mad(float4 gradient, const sample& s0, const sample& s1)
    {
        float4 x  = math::set(s0.x0, s0.x1, s0.x2, s0.x3);
        float4 y  = math::set(s0.y0, s0.y1, s0.y2, s0.y3);
        float4 sx = math::set(s1.x0, s1.x1, s1.x2, s1.x3);
        float4 sy = math::set(s1.y0, s1.y1, s1.y2, s1.y3);

        float4 mx = math::mul(gradient, x);
        float4 my = math::mul(gradient, y);

        float4 dx = math::add(mx, sx);
        float4 dy = math::add(my, sy);

        sample r;

        r.x0 = math::get_x(dx);
        r.x1 = math::get_y(dx);
        r.x2 = math::get_z(dx);
        r.x3 = math::get_w(dx);

        r.y0 = math::get_x(dy);
        r.y1 = math::get_y(dy);
        r.y2 = math::get_z(dy);
        r.y3 = math::get_w(dy);
        return r;
    }

    struct deform_points_kernel2
    {
        const   cuda::image_kernel_info m_sampler;
        const   uint8_t*                m_grad;


        deform_points_kernel2(const cuda::image_kernel_info& sampler, const uint8_t* grad) : m_sampler(sampler), m_grad(grad)
        {

        }

        __device__ inline float    compute_sobel_dx(
            float ul, // upper left
            float um, // upper middle
            float ur, // upper right
            float ml, // middle left
            float mm, // middle (unused)
            float mr, // middle right
            float ll, // lower left
            float lm, // lower middle
            float lr // lower right
            )
        {
            return ur + 2 * mr + lr - ul - 2 * ml - ll;
        }

        __device__ inline float    compute_sobel_dy(
            float ul, // upper left
            float um, // upper middle
            float ur, // upper right
            float ml, // middle left
            float mm, // middle (unused)
            float mr, // middle right
            float ll, // lower left
            float lm, // lower middle
            float lr // lower right
            )
        {
            return  ul + 2 * um + ur - ll - 2 * lm - lr;
        }

    
        __device__ thrust::tuple<point, uint8_t> operator() (const thrust::tuple < point, point> p)
        {
            using namespace cuda;

            auto pt = thrust::get<0>(p);
            auto normal = thrust::get<1>(p);

            auto w = m_sampler.width();
            auto h = m_sampler.height();

            auto pixel_size = max(1.0f / w, 1.0f / h);
            
            auto scale = 1.5f;
            auto k1 = make_point(scale * pixel_size, scale * pixel_size);   //adjust this for faster convergence

            auto d0 = mad(k1, normal, pt);
             
            auto x = d0.x * w;
            auto y = d0.y * h;

            const uint8_t* pix00 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 1, y - 1);
            const uint8_t* pix01 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 0, y - 1);
            const uint8_t* pix02 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x + 1, y - 1);

            const uint8_t* pix10 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x - 1, y);
            const uint8_t* pix11 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x - 0, y);
            const uint8_t* pix12 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x + 1, y);

            const uint8_t* pix20 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 1, y + 1);
            const uint8_t* pix21 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 0, y + 1);
            const uint8_t* pix22 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x + 1, y + 1);
             
            auto  u00 = *pix00;
            auto  u01 = *pix01;
            auto  u02 = *pix02;

            auto  u10 = *pix10;
            auto  u11 = *pix11;
            auto  u12 = *pix12;

            auto  u20 = *pix20;
            auto  u21 = *pix21;
            auto  u22 = *pix22;

            auto mx = max(u00, u01);
            mx = max(mx, u02);
            mx = max(mx, u10);
            mx = max(mx, u11);
            mx = max(mx, u12);
            mx = max(mx, u20);
            mx = max(mx, u21);
            mx = max(mx, u22);

            //mx = *pix11;
            uint32_t stop = 0;

            if ( mx > 250  || x > (w - 4) || y > ( h - 4) || x < 4 || y < 4 ) 
            {  
                d0   = pt;
                stop = 1;
            }
            else
            {
                auto muls = 1.0f / 255.0f;

                auto u10f  = u10 * muls;
                auto u12f  = u12 * muls;

                auto u21f  = u21 * muls;
                auto u01f  = u01 * muls;

                auto gradx = u12f - u10f;
                auto grady = u21f - u01f;

                auto m = max( abs(gradx), abs(grady) );
                auto n = 1.0f / ( m + 0.001f);

                gradx = gradx * n;
                grady = grady * n;

                point up = make_point( gradx, grady );

                float scale_x = 2.6f;
                float scale_y = 2.6f;

                auto k1      = make_point(scale * pixel_size, scale * pixel_size);   //adjust this for faster convergence
                auto k2      = make_point(scale_x * pixel_size, scale_y * pixel_size);
                auto grad_pt = make_point(gradx, grady);
                
                d0 = mad(k1, normal, pt);
                d0 = mad(k2, grad_pt, d0);

                /*
                float s = 0.002f;
                d0 = add( mul( s,up ), d0 );
                */
            }

            return thrust::make_tuple(d0, stop);
        }
    };

    

    //sample the curve and obtain patches through curve interpolation as in the paper
    void deform( const patches& p, const imaging::cuda_texture& grad, patches& deformed, thrust::device_vector<uint32_t>& stop)
    {
        using namespace thrust;

        samples s;
        s.resize(p.size());

        //get normals that we want to transfer along
        thrust::transform(p.begin(), p.end(), s.begin(), normal_curve_points_kernel() );

        //convert to points for easier gradient sampling
        points  normal_vectors;
        normal_vectors.resize(s.size() * 4);

        {
            auto r0 = make_strided_range(normal_vectors.begin() + 0, normal_vectors.end(), 4);
            auto r1 = make_strided_range(normal_vectors.begin() + 1, normal_vectors.end(), 4);
            auto r2 = make_strided_range(normal_vectors.begin() + 2, normal_vectors.end(), 4);
            auto r3 = make_strided_range(normal_vectors.begin() + 3, normal_vectors.end(), 4);

            auto b = make_zip_iterator(make_tuple(s.begin(), r0.begin(), r1.begin(), r2.begin(), r3.begin()));
            auto e = make_zip_iterator(make_tuple(s.end(), r0.end(), r1.end(), r2.end(), r3.end()));
            thrust::for_each(b, e, scatter_normals_kernel());
        }

        //convert patches to points for easier gradient sampling
        points  pts;
        pts.resize(s.size() * 4);
        {
            auto r0 = make_strided_range(pts.begin() + 0, pts.end(), 4);
            auto r1 = make_strided_range(pts.begin() + 1, pts.end(), 4);
            auto r2 = make_strided_range(pts.begin() + 2, pts.end(), 4);
            auto r3 = make_strided_range(pts.begin() + 3, pts.end(), 4);

            auto b = make_zip_iterator(make_tuple(p.begin(), r0.begin(), r1.begin(), r2.begin(), r3.begin()));
            auto e = make_zip_iterator(make_tuple(p.end(), r0.end(), r1.end(), r2.end(), r3.end()));
            thrust::for_each(b, e, scatter_points_kernel());
        }

        //deform samples with the image gradient
        points resampled_points;
        resampled_points.resize(s.size() * 4);
        stop.resize( s.size() * 4);
        {
            auto info = ::cuda::create_image_kernel_info(grad);
            auto pixels = grad.get_gpu_pixels();
            
            auto b = make_zip_iterator(make_tuple(pts.begin(), normal_vectors.begin()));
            auto e = make_zip_iterator(make_tuple(pts.end(),   normal_vectors.end()));

            auto tb = make_zip_iterator(make_tuple(resampled_points.begin(),    stop.begin()));
            auto te = make_zip_iterator(make_tuple(resampled_points.end(),      stop.end()));

            thrust::transform(b, e, tb, deform_points_kernel2(info, pixels) );
        }

        //patch moved samples to ensure c0 continuity again
        points avg_points;
        avg_points.resize(s.size() * 4);
        {
            auto cb = make_counting_iterator(0);
            auto ce = cb + s.size();

            copy(resampled_points.begin(), resampled_points.end(), avg_points.begin());

            for_each(cb, ce, average_points_kernel(&resampled_points[0], &avg_points[0], s.size() ));
        }

        //gather transformed samples again
        deformed.resize(s.size());
        {
            //resampled points
            //auto b = resampled_points.begin();
            //auto e = resampled_points.end();

            auto b = avg_points.begin();
            auto e = avg_points.end();

            auto r0 = make_strided_range(b + 0, e, 4);
            auto r1 = make_strided_range(b + 1, e, 4);
            auto r2 = make_strided_range(b + 2, e, 4);
            auto r3 = make_strided_range(b + 3, e, 4);

            auto b0 = pts.begin();
            auto e0 = pts.end();

            auto p0 = make_strided_range(b0 + 0, e0, 4);
            auto p1 = make_strided_range(b0 + 1, e0, 4);
            auto p2 = make_strided_range(b0 + 2, e0, 4);
            auto p3 = make_strided_range(b0 + 3, e0, 4);

            auto zb = make_zip_iterator(make_tuple(r0.begin(), r1.begin(), r2.begin(), r3.begin(), p0.begin(), p1.begin(), p2.begin(), p3.begin(), p.begin()));
            auto ze = make_zip_iterator(make_tuple(r0.end(), r1.end(), r2.end(), r3.end(), p0.end(), p1.end(), p2.end(), p3.end(), p.end()));

            thrust::transform(zb, ze, deformed.begin(), gather_samples_kernel());
        }

        return;

    }
    
}


