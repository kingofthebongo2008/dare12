#pragma once

namespace freeform
{
    struct vector2
    {
        float x;
        float y;
    };

    __device__ __host__ inline vector2 make_vector2(point a, point b)
    {
        vector2 r;

        r.x = b.x - a.x;
        r.y = b.y - a.y;

        return r;
    }

    __device__ __host__ inline vector2 mul(float s, vector2 v)
    {
        vector2 r;
        r.x = v.x * s;
        r.y = v.y * s;
        return r;
    }

    __device__ __host__ inline vector2 add(vector2 v1, vector2 v2)
    {
        vector2 r;
        r.x = v1.x + v2.x;
        r.y = v1.y + v2.y;
        return r;
    }

    __device__ __host__ inline vector2 sub(vector2 v1, vector2 v2)
    {
        vector2 r;
        r.x = v1.x + v2.x;
        r.y = v1.y + v2.y;
        return r;
    }

    __device__ __host__ inline float dot(vector2 v1, vector2 v2)
    {
        return v1.x * v2.x + v1.y * v2.y;
    }

    __device__ __host__ inline float norm(vector2 v1)
    {
        return sqrtf(dot(v1, v1));
    }
}
