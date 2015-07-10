#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>

#include <math/math_vector.h>

#include "math_functions.h"

#include "cuda_strided_range.h"
#include "cuda_print_utils.h"

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

    __device__ inline sample madd(float4 gradient, const sample& s0, const sample& s1)
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
            //sample displaced    = madd(gradient, normal, s);

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

    //sample the curve and obtain patches through curve interpolation as in the paper
    samples deform( const patches& p )
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
        points  points;
        points.resize(s.size() * 4);
        {
            
            auto r0 = make_strided_range(points.begin() + 0, points.end(), 4);
            auto r1 = make_strided_range(points.begin() + 1, points.end(), 4);
            auto r2 = make_strided_range(points.begin() + 2, points.end(), 4);
            auto r3 = make_strided_range(points.begin() + 3, points.end(), 4);

            auto b = make_zip_iterator(make_tuple(p.begin(), r0.begin(), r1.begin(), r2.begin(), r3.begin()));
            auto e = make_zip_iterator(make_tuple(p.end(), r0.end(), r1.end(), r2.end(), r3.end()));
            thrust::for_each(b, e, scatter_points_kernel());
            
        }

        return s;
    }
    
}


