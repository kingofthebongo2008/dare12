#pragma once

#include "freeform_patch.h"

namespace freeform
{
    struct segment
    {
        point a;
        point b;
    };

    __device__ inline float area(point a, point b, point c)
    {
        return ( b.x - a.x ) * ( c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    }

    __device__ inline bool left(point a, point b, point c)
    {
        return area(a, b, c) > 0.0f;
    }

    __device__ inline bool colinear(point a, point b, point c)
    {
        //todo: epsilon
        return area(a, b, c) == 0.0f;
    }

    //exclusive or: returns true exactly one argument is true
    __device__ inline bool xor(bool x, bool y)
    {
        //return !x ^ !y;
        return x ^ y;
    }

    __device__ inline bool intersect_segments(point a, point b, point c, point d)
    {
        uint32_t a0 = colinear(a, b, c) ? 1 : 0 ;
        uint32_t a1 = colinear(a, b, d) ? 1 : 0;
        uint32_t a2 = colinear(c, d, a) ? 1 : 0;
        uint32_t a3 = colinear(c, d, b) ? 1 : 0;

        uint32_t co = a0 + a1 + a2 + a3;

        if (co == 0 )
        {
            //c - > d are left of the segment ab or vice versa
            return xor( left(a, b, c), left(a, b, d) ) && xor(left(c, d, a), left(c, d, b));
        }
        else
        {
            return false;
        }
    }

    __device__ inline bool intersect_segments(segment a, segment b)
    {
        return intersect_segments(a.a, a.b, b.a, b.b);
    }

    template < uint32_t c > __device__  point make_point(patch a)
    {
        point r;
        switch (c)
        {
            case 0: r = make_point(a.x0, a.y0); break;
            case 1: r = make_point(a.x1, a.y1); break;
            case 2: r = make_point(a.x2, a.y2); break;
            case 3: r = make_point(a.x3, a.y3); break;
            default: break;
        }
        return r;
    }

    __device__ inline segment make_segment(point a, point b)
    {
        segment s;
        s.a = a;
        s.b = b;
        return s;
    }

    __device__ inline bool intersect_patches(patch a, patch b)
    {
        segment s0 = make_segment(make_point<0>(a), make_point<1>(a));
        segment s1 = make_segment(make_point<1>(a), make_point<2>(a));
        segment s2 = make_segment(make_point<2>(a), make_point<3>(a));

        segment d0 = make_segment(make_point<0>(b), make_point<1>(b));
        segment d1 = make_segment(make_point<1>(b), make_point<2>(b));
        segment d2 = make_segment(make_point<2>(b), make_point<3>(b));

        uint32_t a00 = intersect_segments(s0, d0) ? 1 : 0;
        uint32_t a01 = intersect_segments(s0, d1) ? 1 : 0;
        uint32_t a02 = intersect_segments(s0, d2) ? 1 : 0;

        uint32_t a10 = intersect_segments(s1, d0) ? 1 : 0;
        uint32_t a11 = intersect_segments(s1, d1) ? 1 : 0;
        uint32_t a12 = intersect_segments(s1, d2) ? 1 : 0;

        uint32_t a20 = intersect_segments(s2, d0) ? 1 : 0;
        uint32_t a21 = intersect_segments(s2, d1) ? 1 : 0;
        uint32_t a22 = intersect_segments(s2, d2) ? 1 : 0;

        uint32_t co = a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22;

        return co != 0;
    
    }
}
