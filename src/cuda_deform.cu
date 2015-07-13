#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>
#include <thrust/sort.h>

#include <math/math_vector.h>

#include "math_functions.h"

#include "cuda_imaging.h"
#include "cuda_strided_range.h"
#include "cuda_print_utils.h"
#include "cuda_aabb.h"

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

    __device__ inline point sub(point a, point b)
    {
        point c;

        c.x = a.x - b.x;
        c.y = a.y - b.y;
        return c;
    }

    __device__ inline point mul(point a, point b)
    {
        point c;

        c.x = a.x * b.x;
        c.y = a.y * b.y;
        return c;
    }


    __device__ inline point mul(float s, point b)
    {
        point c;

        c.x = s * b.x;
        c.y = s * b.y;
        return c;
    }

    __device__ inline point mad(point a, point b, point c)
    {
        point d;

        d.x = a.x * b.x + c.x;
        d.y = a.y * b.y + c.y;
        return d;
    }

    __device__ inline point normalize(point a)
    {
        float magnitude = a.x * a.x + a.y * a.y;
        float s = 1.0f / sqrtf(magnitude);

        return mul(s, a);
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

    struct deform_points_kernel2
    {
        const   cuda::image_kernel_info m_sampler;
        const   uint8_t*                m_grad;

        deform_points_kernel2(const cuda::image_kernel_info& sampler, const uint8_t* grad) : m_sampler(sampler), m_grad(grad)
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
            dy = ul + 2 * um + ur - ll - 2 * lm - lr;

            float  sum = static_cast<float> (abs(dx) + abs(dy));
            return sum;
        }

        __device__ thrust::tuple<point, uint8_t> operator() (const thrust::tuple < point, point> p)
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

            point d0;
            auto w = m_sampler.width();
            auto h = m_sampler.height();
            uint32_t stop = 0;

            if ( false ) //mx > 250  || pt.x > (w - 4) || pt.y > ( h - 4) ) 
            {  
                d0 = pt;
                stop = 1;
            }
            else
            {
                auto k1 = make_point(2.0f, 2.0f);   //adjust this for faster convergence
                d0 = mad(k1, normal, pt);
                
                point a = make_point(0.0011111f, 0.0011111f);
                d0 = add(d0, a);
                d0 = sub(d0, a);
            }

            return thrust::make_tuple(d0, stop);
        }
    };

    struct gather_samples_kernel
    {
        __device__ patch operator() (const thrust::tuple< point, point, point, point, point, point, point, point, patch > & pt )
        {
            sample s;

            //resampled points
            point r0 = thrust::get<0>(pt);
            point r1 = thrust::get<1>(pt);
            point r2 = thrust::get<2>(pt);
            point r3 = thrust::get<3>(pt);

            point p0 = thrust::get<4>(pt);
            point p1 = thrust::get<5>(pt);
            point p2 = thrust::get<6>(pt);
            point p3 = thrust::get<7>(pt);

            //form delta of moved points
            s.x0 = r0.x - p0.x;
            s.x1 = r1.x - p1.x;
            s.x2 = r2.x - p2.x;
            s.x3 = r3.x - p3.x;

            s.y0 = r0.y - p0.y;
            s.y1 = r1.y - p1.y;
            s.y2 = r2.y - p2.y;
            s.y3 = r3.y - p3.y;

            //obtain delta of moved control points
            patch r = interpolate_curve(s);

            patch p = thrust::get<8>(pt);

            patch res;

            res.x0 = p.x0 + r.x0;
            res.x1 = p.x1 + r.x1;
            res.x2 = p.x2 + r.x2;
            res.x3 = p.x3 + r.x3;

            res.y0 = p.y0 + r.y0;
            res.y1 = p.y1 + r.y1;
            res.y2 = p.y2 + r.y2;
            res.y3 = p.y3 + r.y3;


            return res;
        }
    };

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

    struct average_normals
    {
        thrust::device_ptr< point > m_normals_in;
        thrust::device_ptr< point > m_normals_out;
        uint32_t                    m_count;

        average_normals(thrust::device_ptr<point> normals_in, thrust::device_ptr<point> normals_out, uint32_t count) :
            m_normals_in(normals_in)
            , m_normals_out(normals_out)
            , m_count(count)
        {

        }

        __device__ void operator() (uint32_t i) const
        {
            auto i0 = 4 * i + 3;
            auto i1 = 4 * i + 4;

            if (i == m_count - 1)
            {
                i1 = 0;
            }

            point n0 = m_normals_in[i0];
            point n1 = m_normals_in[i1];

            point n = mul(0.5f, add(n0, n1));

            n = normalize(n);

            m_normals_out[i0] = n;
            m_normals_out[i1] = n;
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

        //make the normal vectors to be the same for the same control points
        points  normal_vectors_avg;
        normal_vectors_avg.resize(s.size() * 4);
        thrust::copy(normal_vectors.begin(), normal_vectors.end(), normal_vectors_avg.begin());

        {
            auto b = make_counting_iterator(0);
            auto e = b + s.size();
            thrust::for_each(b, e, average_normals(&normal_vectors[0], &normal_vectors_avg[0], s.size()));
        }
        thrust::copy(normal_vectors_avg.begin(), normal_vectors_avg.end(), normal_vectors.begin());

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

        //gather transformed samples again
        patches control_points;
        control_points.resize(s.size());
        {
            //resampled points
            auto b = resampled_points.begin();
            auto e = resampled_points.end();

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

            thrust::transform(zb, ze, control_points.begin(), gather_samples_kernel());
        }

        extern uint32_t iterations;


        deformed.resize( s.size() );
        copy(control_points.begin(), control_points.end(), deformed.begin());
        
        //print<patches, patch>(control_points);
        //average points on the boundaries between patches, since they point in different directions
        


        {
            auto b = make_counting_iterator(0);
            auto e = b + s.size();
            thrust::for_each(b, e, average_patches(&control_points[0], &deformed[0], s.size()));
        }

        return;

    }
    
}


