#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>

#include <math/math_vector.h>

#include "math_functions.h"

#include "cuda_imaging.h"
#include "cuda_strided_range.h"
#include "cuda_print_utils.h"

#include <algorithm>

namespace freeform
{
    __device__ inline float4 make_x(const patch& p)
    {
        return math::set(p.x0, p.x1, p.x2, p.x3);
    }

    __device__ inline float4 make_y(const patch& p)
    {
        return math::set(p.y0, p.y1, p.y2, p.y3);
    }

    __device__ inline point add(point a, point b)
    {
        point c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        return c;
    }

    __device__ inline point mul(point a, point b)
    {
        point c;

        c.x = a.x * b.x;
        c.y = a.y * b.y;
        return c;
    }

    __device__ inline point mad(point a, point b, point c)
    {
        point d;

        d.x = a.x * b.x + c.x;
        d.y = a.y * b.y + c.y;
        return d;
    }

    __device__ inline patch compute_derivatives(const patch& p)
    {
        float4 x = make_x(p);
        float4 y = make_y(p);

        //compute the control points of the derivative curve
        float4 dx = math::cubic_bezier_derivative(x);
        float4 dy = math::cubic_bezier_derivative(y);

        patch r;

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

    __device__ inline float norm_l2(float x, float y)
    {
        return  sqrtf(x * x + y * y);
    }

    //calculate normal vector of q cubic bezier poin, assumes in s are the tangent (derivative) values of the bezier curve
    __device__ inline sample normal_vector(const sample& s)
    {
        float d0 = norm_l2(s.x0, s.y0);
        float d1 = norm_l2(s.x1, s.y1);
        float d2 = norm_l2(s.x2, s.y2);
        float d3 = norm_l2(s.x3, s.y3);

        //convert to union tangent vector
        float x0 = s.x0 / d0;
        float x1 = s.x1 / d1;
        float x2 = s.x2 / d2;
        float x3 = s.x3 / d3;

        float y0 = s.y0 / d0;
        float y1 = s.y1 / d1;
        float y2 = s.y2 / d2;
        float y3 = s.y3 / d3;

        //obtain normal components as a rotation in the plane of the tangent components of 90 degrees
        float n_x_0 = y0;
        float n_y_0 = -x0;

        float n_x_1 = y1;
        float n_y_1 = -x1;

        float n_x_2 = y2;
        float n_y_2 = -x2;
        
        float n_x_3 = y3;
        float n_y_3 = -x3;

        sample r;

        r.x0 = n_x_0;
        r.x1 = n_x_1;
        r.x2 = n_x_2;
        r.x3 = n_x_3;

        r.y0 = n_y_0;
        r.y1 = n_y_1;
        r.y2 = n_y_2;
        r.y3 = n_y_3;

        return r;

    }

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

    struct normal_curve_points_kernel
    {
        __device__ sample  operator() ( const patch& p )
        {
            float4 t            = math::set(0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 3.0f);

            patch  derivative   = compute_derivatives(p);

            sample s            = multi_eval_patch_3(p, t);             //sample the bezier curve
            sample s_derivative = multi_eval_patch_2(derivative, t);    //sample the derivative
            
            sample normal       = normal_vector(s_derivative);

            //float4 gradient     = math::set(100.0f,100.0f, 100.0f, 100.0f);

            //displace points along the normals
            //sample displaced    = mad(gradient, normal, s);

            //sample the derivative of a patch and output 4 points along the normal
            return normal;
        }
    };

    struct scatter_normals_kernel
    {
        //scatter the samples into different points, so we can get them more parallel
        template <typename t> __device__ void operator()( t& t)
        {
            sample p = thrust::get < 0 >(t);

            point p0;
            point p1;
            point p2;
            point p3;

            p0.x = p.x0;
            p1.x = p.x1;
            p2.x = p.x2;
            p3.x = p.x3;

            p0.y = p.y0;
            p1.y = p.y1;
            p2.y = p.y2;
            p3.y = p.y3;

            thrust::get<1>(t) = p0;
            thrust::get<2>(t) = p1;
            thrust::get<3>(t) = p2;
            thrust::get<4>(t) = p3;
        }
    };

    struct scatter_points_kernel
    {
        //scatter the samples into different points, so we can get them more parallel
        template <typename t> __device__ void operator()(t& tp)
        {
            patch p  = thrust::get < 0 >(tp);

            float4 t = math::set(0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 3.0f);
            sample s = multi_eval_patch_3(p, t);             //sample the bezier curve

            point p0;
            point p1;
            point p2;
            point p3;

            p0.x = s.x0;
            p1.x = s.x1;
            p2.x = s.x2;
            p3.x = s.x3;

            p0.y = s.y0;
            p1.y = s.y1;
            p2.y = s.y2;
            p3.y = s.y3;

            thrust::get<1>(tp) = p0;
            thrust::get<2>(tp) = p1;
            thrust::get<3>(tp) = p2;
            thrust::get<4>(tp) = p3;
        }
    };

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


    struct deform_points_kernel
    {
        const   cuda::image_kernel_info m_sampler;
        const   uint8_t*                m_grad;

        deform_points_kernel(   const cuda::image_kernel_info& sampler, const uint8_t* grad ) : m_sampler(sampler), m_grad(grad)
        {

        }

        __device__ static inline float    compute_sobel(
            float ul, // upper left
            float um, // upper middle
            float ur, // upper right
            float ml, // middle left
            float mm, // middle (unused)
            float mr, // middle right
            float ll, // lower left
            float lm, // lower middle
            float lr, // lower right
            float& dx,
            float& dy
            )
        {
            dx = ur + 2 * mr + lr - ul - 2 * ml - ll;
            dy   = ul + 2 * um + ur - ll - 2 * lm - lr;

            float  sum = static_cast<float> (abs(dx) + abs(dy));
            return sum;
        }

        __device__ point operator() (const thrust::tuple < point, point> p)
        {
            using namespace cuda;

            auto pt = thrust::get<0>(p);
            auto normal = thrust::get<1>(p);

            auto x = pt.x;
            auto y = pt.y;

            const uint8_t* pix00 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 1, y - 1);
            const uint8_t* pix01 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 0, y - 1);
            const uint8_t* pix02 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x + 1, y - 1);


            const uint8_t* pix10 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x - 1, y);
            const uint8_t* pix11 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x - 0, y);
            const uint8_t* pix12 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x + 1, y);

            const uint8_t* pix20 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 1, y + 1);
            const uint8_t* pix21 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 0, y + 1);
            const uint8_t* pix22 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x + 1, y + 1);

            float c   = 1.0f / 255.0f;

            auto  u00 = *pix00 * c;
            auto  u01 = *pix01 * c;
            auto  u02 = *pix02 * c;

            auto  u10 = *pix10 * c;
            auto  u11 = *pix11 * c;
            auto  u12 = *pix12 * c;

            auto  u20 = *pix20 * c;
            auto  u21 = *pix21 * c;
            auto  u22 = *pix22 * c;

            float dx = 0.0f;
            float dy = 0.0f;

            auto  r = compute_sobel(
                u00, u01, u02,
                u10, u11, u12,
                u20, u21, u22, dx, dy
                );

            //normalize the gradient
            float g = 1.0f / (r + 0.0001f);
            dx = dx * g;
            dy = dy * g;

            dx = 22.0f; //test to see if the gradient works

            float scale      = 2.0f;
            float pixel_size = max( 2.0f / m_sampler.width(), 2.0f / m_sampler.height() );

            point k1         = make_point(scale, scale);
            point k          = make_point(-scale * 1.1f, -scale * 1.1f);

            point grad       = make_point(dx, dy);
            point d0         = mad(k1, normal, pt);
            point d1         = mad(k, grad, d0);

            return d1;
        }
    };

    struct gather_samples_kernel
    {
        __device__ patch operator() (const thrust::tuple< point, point, point, point> & pt)
        {
            sample s;

            point p0 = thrust::get<0>(pt);
            point p1 = thrust::get<1>(pt);
            point p2 = thrust::get<2>(pt);
            point p3 = thrust::get<3>(pt);

            s.x0 = p0.x;
            s.x1 = p1.x;
            s.x2 = p2.x;
            s.x3 = p3.x;

            s.y0 = p0.y;
            s.y1 = p1.y;
            s.y2 = p2.y;
            s.y3 = p3.y;

            patch p = interpolate_curve(s);

            return p;
        }
    };

    //sample the curve and obtain patches through curve interpolation as in the paper
    patches deform(const patches& p, const imaging::cuda_texture& grad )
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
        {
            auto info = ::cuda::create_image_kernel_info(grad);
            auto pixels = grad.get_gpu_pixels();
            
            auto b = make_zip_iterator(make_tuple(pts.begin(), normal_vectors.begin()));
            auto e = make_zip_iterator(make_tuple(pts.end(), normal_vectors.end()));
            thrust::transform(b, e, resampled_points.begin(), deform_points_kernel(info, pixels) );
        }

        //gather transformed samples again
        patches patches;
        patches.resize(s.size());
        {
            auto b = resampled_points.begin();
            auto e = resampled_points.end();

            auto zb = make_zip_iterator(make_tuple(b,   b + 1,  b + 2,  b + 3));
            auto ze = make_zip_iterator(make_tuple(e,   e,      e,      e));

            auto s  = make_strided_range(zb, ze, 4 );

            thrust::transform(s.begin(), s.end(), patches.begin(), gather_samples_kernel());
        }

        return patches;
    }
    
}


