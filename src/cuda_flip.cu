#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>
#include <thrust/sort.h>

#include <math/math_vector.h>

#include "math_functions.h"
#include <algorithm>

#include "cuda_aabb.h"

namespace freeform
{
    struct collide_kernel_aabb
    {
        uint32_t                    m_n;
        thrust::device_ptr< patch>  m_patches;

        collide_kernel_aabb(uint32_t n, thrust::device_ptr< patch>  patches) :m_n(n), m_patches(patches)
        {

        }


        __device__ static inline bool collide(const patch& p0, const patch& p1 )
        {
            aabb a0 = make_aabb(p0);
            aabb b0 = make_aabb(p1);

            float4 a = math::set(a0.m_min_x, a0.m_min_y, a0.m_max_x, a0.m_max_y);
            float4 b = math::set(b0.m_min_x, b0.m_min_y, b0.m_max_x, b0.m_max_y);

            return intersect_bounding_boxes(a, b) ? true : false;
        }

        __device__ bool operator() (uint32_t i) const
        {
            patch p0 = m_patches[i];
            patch p1 = m_patches[i + 1];
            return collide(p0, p1);
        }
    };

    struct collide_kernel
    {
        uint32_t                    m_n;
        thrust::device_ptr< patch>  m_patches;

        collide_kernel(uint32_t n, thrust::device_ptr< patch>  patches) :m_n(n), m_patches(patches)
        {

        }


        __device__ static inline bool collide(const patch& p0, const patch& p1)
        {
            return false;
        }

        __device__ bool operator() (uint32_t i) const
        {
            patch p0 = m_patches[i];
            patch p1 = m_patches[i + 1];
            return collide(p0, p1);
        }
    };

    static inline std::tuple<patch, patch> reorder(const patch& p0, const patch& p1)
    {
        patch r0;
        patch r1;

        r0.x0 = p0.x0;
        r0.x1 = p0.x1;
        r0.x2 = p1.x2;
        r0.x3 = p1.x3;

        r1.x0 = p0.x2;
        r1.x1 = p0.x3;
        r1.x2 = p1.x0;
        r1.x3 = p1.x1;

        r0.y0 = p0.y0;
        r0.y1 = p0.y1;
        r0.y2 = p1.y2;
        r0.y3 = p1.y3;

        r1.y0 = p0.y2;
        r1.y1 = p0.y3;
        r1.y2 = p1.y0;
        r1.y3 = p1.y1;

        return std::make_tuple( r0, r1);
    }

    patches flip(   patches& p  )
    {
        return p;
        
        using namespace thrust;

        auto s = p.size();
        device_vector< bool > collision;
        

        sort(p.begin(), p.end(), lexicographical_sorter());

        collision.resize( p.size() );

        auto b = make_counting_iterator(0);
        auto e = b + s - 1;

        transform(b, e, collision.begin(), collide_kernel(s, &p[0]));

        host_vector<patch>  h_patches;
        host_vector< bool > h_collision;

        h_patches.resize( p.size() );
        h_collision.resize( collision.size() );

        host_vector<patch> outside;
        host_vector<patch> inside;

        outside.reserve(2 * s);
        inside.reserve (2 * s);

        copy( p.begin(), p.end(), h_patches.begin() );
        copy( collision.begin(),  collision.end(), h_collision.begin() );

        for (uint32_t i = 0; i < s ; ++i)
        {
            if ( h_collision[i])
            {
                auto t = reorder(h_patches[i], h_patches[i + 1]);

                auto t0 = std::get<0>(t);
                auto t1 = std::get<1>(t);
               
                outside.push_back(t0);
                inside.push_back(t1);
            }
            else
            {
                outside.push_back(h_patches[i]);
            }
        }

        patches r;
        r.resize(outside.size() + inside.size() );
        copy(outside.begin(), outside.end(), r.begin());
        copy(inside.begin(), inside.end(),  r.begin() + outside.size() );
        return r;
    }
}


