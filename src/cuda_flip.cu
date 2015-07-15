#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>
#include <thrust/sort.h>

#include <math/math_vector.h>

#include "math_functions.h"
#include <algorithm>

#include "cuda_aabb.h"
#include "cuda_patches.h"
#include "cuda_print_utils.h"

#include "cuda_collision_detection.h"
#include "cuda_math_2d.h"

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

    
    inline uint32_t get_next(uint32_t index, uint32_t n)
    {
        if (index < n - 1)
        {
            return index + 1;
        }
        else
        {
            return 0;
        }
    }

    inline uint32_t get_prev(uint32_t index, uint32_t n)
    {
        if (index > 0 )
        {
            return index - 1;
        }
        else
        {
            return n - 1;
        }
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

    struct collision_result_sorter
    {
        __device__ bool operator() (const collision_result& a, const collision_result& b)
        {
            if (a.m_index_0 < b.m_index_0)
            {
                return true;
            }
            else
            {
                if (a.m_index_0 == b.m_index_0)
                {
                    return a.m_index_1 < b.m_index_1;
                }
                else
                {
                    return false;
                }
            }
            
        }
    };

    struct collision_control_polygon
    {
        //sweep algorithm
        __device__ inline bool operator()(const patch& a, const patch& b) const
        {
            point a0 = make_point(a.x0, a.y0);
            point a3 = make_point(a.x3, a.y3);

            point b0 = make_point(b.x0, b.y0);
            point b3 = make_point(b.x3, b.y3);

            vector2 ab = make_vector2(a0, a3);

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
    };


    struct collision_aabb
    {
        __device__ inline bool operator()(const patch& p0, const patch& p1) const
        {
            aabb a0 = make_aabb(p0);
            aabb b0 = make_aabb(p1);

            float4 a = math::set(a0.m_min_x, a0.m_min_y, a0.m_max_x, a0.m_max_y);
            float4 b = math::set(b0.m_min_x, b0.m_min_y, b0.m_max_x, b0.m_max_y);

            return intersect_bounding_boxes(a, b) ? true : false;
        }
    };

    struct collision_segments
    {
        //segment intersection of the control polygon
        __device__ inline bool operator()(const patch& p0, const patch& p1) const
        {
            return intersect_patches(p0, p1);
        }
    };

    template < typename t > 
    struct collision_detection_kernel
    {
        thrust::device_ptr< patch >           m_patches;
        thrust::device_ptr< tab >             m_tabs;
        thrust::device_ptr< collision_result> m_results;
        thrust::device_ptr<uint32_t>          m_element_count;
        t                                     m_collision_functor;
        
        

        collision_detection_kernel( thrust::device_ptr<patch>   patches, 
                                    thrust::device_ptr<tab>     tabs,
                                    thrust::device_ptr<collision_result> results,
                                    thrust::device_ptr<uint32_t> element_count, const t& collision_functor) : m_patches(patches), m_tabs(tabs), m_results(results), m_element_count(element_count), m_collision_functor(collision_functor)
        {

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
                auto  result = m_collision_functor(p0, p1);
 
                if (result)
                {
                    auto id = atomicAdd(m_element_count.get(), 1);

                    collision_result r;

                    r.m_index_0 = min(i0, i1);
                    r.m_index_1 = max(i0, i1);

                    m_results[id] = r;
                }
            }
        }
    };

    template < typename t >
    struct collision_detection_kernel2
    {
        thrust::device_ptr< patch >             m_patches;
        thrust::device_ptr< collision_result >  m_aabb_results;
        thrust::device_ptr< collision_result>   m_results;
        thrust::device_ptr<uint32_t>            m_element_count;
        t                                       m_collision_functor;



        collision_detection_kernel2(thrust::device_ptr<patch>   patches,
            thrust::device_ptr<collision_result>               aabb_results,
            thrust::device_ptr<collision_result> results,
            thrust::device_ptr<uint32_t> element_count, const t& collision_functor) : m_patches(patches), m_aabb_results(aabb_results), m_results(results), m_element_count(element_count), m_collision_functor(collision_functor)
        {

        }

        __device__ void operator() (uint32_t i)
        {
            if (true)
            {
                collision_result r = m_aabb_results[i];

                auto i0 = r.m_index_0;
                auto i1 = r.m_index_1;

                patch p0 = m_patches[i0];
                patch p1 = m_patches[i1];
                auto  result = m_collision_functor(p0, p1);

                if (result)
                {
                    auto id = atomicAdd(m_element_count.get(), 1);

                    collision_result r;

                    r.m_index_0 = min(i0, i1);
                    r.m_index_1 = max(i0, i1);

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

    struct x_min_sorter
    {
        __device__ bool operator()(const patch& p0, const patch& p1) const
        {
            aabb a0 = make_aabb(p0);
            aabb b0 = make_aabb(p1);

            return a0.m_min_x < b0.m_min_x;
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

    static inline float project(vector2 u, vector2 v)
    {
        float d = dot( u, v );
        float n = dot( u, u );

        return d / n;
    }

    namespace details
    {
        struct sorter
        {
            vector2 m_p03;
            point   m_p0;

            sorter(point p0, point p3) : m_p03(make_vector2( p0, p3 ) ) , m_p0(p0)
            {

            }

            inline bool operator() ( point a, point b ) const
            {
                //project 4 points on p03
                auto d0a = project(m_p03, make_vector2(m_p0, a));
                auto d0b = project(m_p03, make_vector2(m_p0, b));

                return abs(d0a) < abs(d0b);
            }

        };

        template <typename t> inline std::tuple<point, point, point, point> sort4(point a, point b, point c, point d, t f)
        {
            point m[4] = { a, b, c, d };
            
            auto n = 4U;

            do
            {
                auto new_n = 0;
                for (auto i = 1U; i < 4U; ++i)
                {
                    if ( f ( m[i], m[i  -1 ] ) )
                    {
                        std::swap( m[i - 1], m[i] );
                        new_n = i;
                    }
                }

                n = new_n;
            } while (n != 0);

            return std::make_tuple(m[0], m[1], m[2], m[3]);
        }

        template <typename t> inline std::tuple<point, point> sort2(point a, point b, t f)
        {
            auto r = f (a, b);

            if (r)
            {
                return std::make_tuple(a, b);
            }
            else
            {
                return std::make_tuple( b, a );
            }
        }
    }
    
    static inline std::tuple<patch, patch> reorder(const patch& p, const patch& q)
    {
        //flip end control points
        auto p0 = make_point(p.x0, p.y0);
        auto p1 = make_point(p.x1, p.y1);
        auto p2 = make_point(p.x2, p.y2);
        auto p3 = make_point(p.x3, p.y3);

        auto q0 = make_point(q.x0, q.y0);
        auto q1 = make_point(q.x1, q.y1);
        auto q2 = make_point(q.x2, q.y2);
        auto q3 = make_point(q.x3, q.y3);

        //now decide about the other points


        //
        auto r0 = p0;
        auto r3 = q3;

        auto s0 = q0;
        auto s3 = p3;

        //first phase
        //project 4 points on p03
        auto sd = details::sort4( p1, p2, q1, q2, details::sorter(p0, p3) );

        //go to the first point
        auto r_a = std::get<0>(sd);
        auto r_b = std::get<1>(sd);

        //go to the second point
        auto q_a = std::get<2>(sd);
        auto q_b = std::get<3>(sd);

        //second phase
        vector2 p03 = make_vector2(p0, q3);
        auto sd1 = details::sort2(r_a, r_b, details::sorter(p0, q3));
        auto sd2 = details::sort2(q_a, q_b, details::sorter(q0, p3));
        
        auto r1 = std::get<0>(sd1);
        auto r2 = std::get<1>(sd1);

        auto s1 = std::get<0>(sd2);
        auto s2 = std::get<1>(sd2);

        patch res0;
        patch res1;

        res0.x0 = r0.x;
        res0.x1 = r1.x;
        res0.x2 = r2.x;
        res0.x3 = r3.x;

        res0.y0 = r0.y;
        res0.y1 = r1.y;
        res0.y2 = r2.y;
        res0.y3 = r3.y;

        res1.x0 = s0.x;
        res1.x1 = s1.x;
        res1.x2 = s2.x;
        res1.x3 = s3.x;

        res1.y0 = s0.y;
        res1.y1 = s1.y;
        res1.y2 = s2.y;
        res1.y3 = s3.y;
        
        return std::make_tuple(res0, res1);
    }

    patches flip(patches& p)
    {
        using namespace thrust;
       
        tabs t;
        auto s = p.size();

        t.resize( p.size() );

        {
            auto cb = make_counting_iterator(0);
            auto ce = cb + s;

            auto b = make_zip_iterator(make_tuple(cb, p.begin()));
            auto e = make_zip_iterator(make_tuple(ce, p.end()));

            transform(b, e, t.begin(), build_tabs_kernel());
        }

        sort(t.begin(), t.end(), lexicographical_sorter_tabs());

        device_vector< collision_result >     dresults;
        device_vector< collision_result >     dresults_segment;
        host_vector< collision_result >       results_segment;
        
        {
            device_ptr<uint32_t>                  element_count = device_malloc<uint32_t>(1);
            dresults.resize(s);
            *element_count = 0;

            auto cb = make_counting_iterator(0);
            auto ce = cb + s - 1;

            //test aabb 
            for_each(cb, ce, collision_detection_kernel<collision_aabb>(&p[0], &t[0], &dresults[0], element_count, collision_aabb()));
            dresults.resize(*element_count);
           
            //if aabb returns results
            if ( !dresults.empty() )
            {
                device_ptr<uint32_t>                  element_count2 = device_malloc<uint32_t>(1);

                dresults_segment.resize(dresults.size());
                cb = make_counting_iterator(0);
                ce = cb + dresults.size();

                *element_count2 = 0;
                //test segments of the aabb
                for_each(cb, ce, collision_detection_kernel2<collision_segments>(&p[0], &dresults[0], &dresults_segment[0], element_count2, collision_segments()));

                dresults_segment.resize(*element_count2);

                if (!dresults_segment.empty())
                {
                    __debugbreak();

                    results_segment.resize(dresults_segment.size());
                    sort(dresults_segment.begin(), dresults_segment.end(), collision_result_sorter());
                    copy(dresults_segment.begin(), dresults_segment.end(), results_segment.begin());
                }

                device_free(element_count2);
            }
          

            device_free(element_count);
        }

        //if there are collisions
        if (!results_segment.empty())
        {
            host_vector<patch> h_patches;
            {
                h_patches.resize(s);
                copy(p.begin(), p.end(), h_patches.begin());
            }


            std::vector<collision_result> test;
            std::vector<patch> h_results2;

            //copy for debuggin
            {
                test.resize(results_segment.size());

                std::copy(results_segment.begin(), results_segment.end(), test.begin());
                h_results2.resize(h_patches.size());
                std::copy(h_patches.begin(), h_patches.end(), h_results2.begin());
            }

            for (auto i = 0U; i < test.size(); ++i)
            {
                auto r = test[i];
                auto t = reorder(h_patches[ r.m_index_0], h_patches[r.m_index_1]);

                

                


            }

            return p;
        }
        else
        {
            return p;
        }
    }
}


