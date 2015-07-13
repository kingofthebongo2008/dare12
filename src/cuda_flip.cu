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
    /*


    struct lexicographical_sorter_tabs
    {
        __device__ bool operator()(const tab& p0, const tab& p1) const
        {
            aabb a0 = p0.m_aabb;
            aabb b0 = p1.m_aabb;

            float4 a = math::set(a0.m_min_x, a0.m_min_y, a0.m_max_x, a0.m_max_y);
            float4 b = math::set(b0.m_min_x, b0.m_min_y, b0.m_max_x, b0.m_max_y);

            return  a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && (a.z < b.z || (a.z == b.z && a.w < b.w)))));
        }
    };
    

    typedef thrust::device_vector<tab> tabs;
    typedef thrust::host_vector<tab>   htabs;

    struct collision_result
    {
        uint32_t m_index_0;
        uint32_t m_index_1;
    };

    struct collide_kernel_aabb
    {
        uint32_t                    m_n;
        thrust::device_ptr< tab>    m_tabs;

        collide_kernel_aabb(uint32_t n, thrust::device_ptr< patch>  patches) :m_n(n), m_patches(patches)
        {

        }

        __device__ static inline bool collide(const tab& p0, const tab& p1 )
        {
            aabb a0 = p0.m_aabb;
            aabb b0 = p1.m_aabb;

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

        transform(b, e, collision.begin(), collide_kernel_aabb(s, &p[0]));

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
            if ( h_collision[i] && (i == 0))
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
    */

    struct vector2
    {
        float x;
        float y;
    };

    __device__ inline vector2 make_vector2(point a, point b)
    {
        vector2 r;

        r.x = b.x - a.x;
        r.y = b.y - a.y;

        return r;
    }

    __device__ inline vector2 mul(float s, vector2 v)
    {
        vector2 r;
        r.x = v.x * s;
        r.y = v.y * s;
        return r;
    }

    __device__ inline vector2 add(vector2 v1, vector2 v2)
    {
        vector2 r;
        r.x = v1.x + v2.x;
        r.y = v1.y + v2.y;
        return r;
    }

    __device__ inline vector2 sub(vector2 v1, vector2 v2)
    {
        vector2 r;
        r.x = v1.x + v2.x;
        r.y = v1.y + v2.y;
        return r;
    }

    __device__ inline float dot(vector2 v1, vector2 v2)
    {
        return v1.x * v2.x + v1.y * v2.y;
    }

    struct tab
    {
        aabb     m_aabb;
        uint32_t m_index;
    };

    struct collision_result
    {
        uint32_t m_index_0;
        uint32_t m_index_1;
    };

    struct collision_detection_kernel
    {
        thrust::device_ptr< patch >           m_patches;
        thrust::device_ptr< tab >             m_tabs;
        thrust::device_ptr< collision_result> m_results;
        thrust::device_ptr<uint32_t>          m_element_count;
        
        

        collision_detection_kernel( thrust::device_ptr<patch>   patches, 
                                    thrust::device_ptr<tab>     tabs,
                                    thrust::device_ptr<collision_result> results,
                                    thrust::device_ptr<uint32_t> element_count ) : m_patches(patches), m_tabs(tabs), m_results(results), m_element_count(element_count)
        {

        }

        __device__ static inline bool collide(const patch& a, const patch& b)
        {
            point a0 = make_point(a.x0, a.y0);
            point a3 = make_point(a.x3, a.y3);

            point b0 = make_point(b.x0, b.y0);
            point b3 = make_point(b.x3, b.y3);

            vector2 ab   = make_vector2(a0, a3);

            vector2 a0b0 = make_vector2(a0, b0);
            vector2 a0b3 = make_vector2(a0, b3);

            float   d0 = dot(ab, a0b0);
            float   d1 = dot(ab, a0b3);

            float   d2 = dot(ab, ab);

            float   r0 = d0 / d2;
            float   r1 = d1 / d2;

            bool    ba = r0 < 0.0f && r1 < 0.0f;
            bool    bb = r0 > 1.0f && r1 > 1.0f;

            return !(ba || bb);
        }

        __device__ void operator() ( uint32_t i )
        {
            if (true)
            {
                tab t0 = m_tabs[i];
                tab t1 = m_tabs[i+1];

                auto i0 = t0.m_index;
                auto i1 = t1.m_index;

                patch p0 = m_patches[i0];
                patch p1 = m_patches[i1];
                auto  result = collide(p0, p1);
 
                if (result)
                {
                    auto id = atomicAdd(m_element_count.get(), 2);

                    collision_result r;

                    r.m_index_0 = i0;
                    r.m_index_1 = i1;

                    m_results[id] = r;
                }
            }
        }
    };

    struct lexicographical_sorter_tabs
    {
        __device__ bool operator()(const tab& p0, const tab& p1) const
        {
            aabb a0 = p0.m_aabb;
            aabb b0 = p1.m_aabb;

            float4 a = math::set(a0.m_min_x, a0.m_min_y, a0.m_max_x, a0.m_max_y);
            float4 b = math::set(b0.m_min_x, b0.m_min_y, b0.m_max_x, b0.m_max_y);

            return  a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && (a.z < b.z || (a.z == b.z && a.w < b.w)))));
        }
    };

    typedef thrust::device_vector<tab> tabs;
    typedef thrust::host_vector<tab>   htabs;

    struct build_tabs_kernel
    {
        __device__ tab operator() (const thrust::tuple< uint32_t, patch >& t)
        {
            auto i = thrust::get<0>(t);
            auto p = thrust::get<1>(t);

            tab ta;

            ta.m_index = i;
            ta.m_aabb = make_aabb(p);

            return ta;
        }
    };

    patches flip(patches& p)
    {
        using namespace thrust;

        tabs t;
        auto s = p.size();

        t.resize(s);

        {
            auto cb = make_counting_iterator(0);
            auto ce = cb + s;

            auto b = make_zip_iterator(make_tuple(cb, p.begin()));
            auto e = make_zip_iterator(make_tuple(ce, p.end()));

            transform(b, e, t.begin(), build_tabs_kernel());
        }

        sort(t.begin(), t.end(), lexicographical_sorter_tabs());

        device_vector< collision_result >     dresults;
        host_vector< collision_result >       results;
        

        {
            device_ptr<uint32_t>                  element_count = device_malloc<uint32_t>(1);
            dresults.resize(s);
            *element_count = 0;

            auto cb = make_counting_iterator(0);
            auto ce = cb + s - 1;

            for_each(cb, ce, collision_detection_kernel(&p[0], &t[0], &dresults[0], element_count));
            dresults.resize(*element_count);
           
            device_free(element_count);

            results.resize(dresults.size());
            copy(dresults.begin(), dresults.end(), results.begin());


        }


        


        

        return p;
    }
}


