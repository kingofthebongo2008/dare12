#pragma once

#include <math/math_vector.h>

#include "cuda_patches.h"
#include "cuda_points.h"


namespace freeform
{
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

    __device__ __host__ inline float norm_l2(float x, float y)
    {
        return  sqrtf(x * x + y * y);
    }

    //calculate normal vector of q cubic bezier point, assumes in s are the tangent (derivative) values of the bezier curve
    __device__ inline sample normal_vector(const sample& s)
    {
        float d0 = norm_l2(s.x0, s.y0);
        float d1 = norm_l2(s.x1, s.y1);
        float d2 = norm_l2(s.x2, s.y2);
        float d3 = norm_l2(s.x3, s.y3);


        float mul0 = 1.0f / d0;
        float mul1 = 1.0f / d1;
        float mul2 = 1.0f / d2;
        float mul3 = 1.0f / d3;


        //convert to union tangent vector
        float x0 = s.x0 * mul0;
        float x1 = s.x1 * mul1;
        float x2 = s.x2 * mul2;
        float x3 = s.x3 * mul3;

        float y0 = s.y0 * mul0;
        float y1 = s.y1 * mul1;
        float y2 = s.y2 * mul2;
        float y3 = s.y3 * mul3;

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

    struct normal_curve_points_kernel
    {
        __device__ sample  operator() (const patch& p)
        {
            float4 t = math::set(0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 3.0f);

            patch  derivative = compute_derivatives(p);

            sample s = multi_eval_patch_3(p, t);             //sample the bezier curve
            sample s_derivative = multi_eval_patch_2(derivative, t);    //sample the derivative

            sample normal = normal_vector(s_derivative);

            return normal;
        }
    };
}