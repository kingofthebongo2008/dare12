#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>
#include <thrust/sort.h>

#include <algorithm>


inline std::ostream& operator<<(std::ostream& s, const float4& p)
{
    s << "x: " << p.x << " " << p.y << " " << p.z << " " << p.w << std::endl;
    return s;
}

namespace freeform
{
    namespace lower_triangular
    {
        __device__ __host__ uint32_t size(uint32_t dimension)
        {
            return dimension * (dimension + 1) / 2;
        }

        //lower triangular matrix row from index
        __device__ __host__ uint32_t row(uint32_t index)
        {
            return (uint32_t)((sqrtf(static_cast<float>( 8 * index + 1) )) + 1) / 2;
        }

        //lower triangular matrix col from index
        __device__ __host__ uint32_t col(uint32_t index)
        {
            auto r = row(index);
            return index - r * (r - 1) / 2;
        }
    }

    namespace upper_triangular
    {
        __device__ __host__ uint32_t size(uint32_t dimension)
        {
            return dimension * (dimension + 1) / 2;
        }

        //lower triangular matrix row from index
        __device__ __host__ uint32_t col(uint32_t index)
        {
            return (uint32_t)((uint32_t)(sqrtf(static_cast<float>(8 * (index)+1))) + 1) / 2;
        }

        //lower triangular matrix col from index
        __device__ __host__ uint32_t row(uint32_t index)
        {
            auto r = col(index );
            return index - r * (r - 1) / 2;
        }
    }

    struct sort_by_abscissa_asc
    {
        __device__ __host__ bool operator()( const point& a, const point& b ) const
        {
            return a.x < b.x;
        }
    };

    struct sort_by_ordinate_asc
    {
        __device__ __host__ bool operator()( const point& a, const point& b ) const
        {
            return a.y < b.y;
        }
    };

    __device__ __host__ inline thrust::tuple<patch, patch> disconnect_patches(const patch& p0, const patch& p1)
    {
        point pt[8];
        
        
        pt[0] = { p0.x0, p0.y0 };
        pt[1] = { p0.x1, p0.y1 };
        pt[2] = { p0.x2, p0.y2 };
        pt[3] = { p0.x3, p0.y3 };

        pt[4] = { p1.x0, p1.y0 };
        pt[5] = { p1.x1, p1.y1 };
        pt[6] = { p1.x2, p1.y2 };
        pt[7] = { p1.x3, p1.y3 };
        

        thrust::sort(&pt[0], &pt[0] + 8, sort_by_abscissa_asc());

        thrust::sort(&pt[0], &pt[0] + 4, sort_by_ordinate_asc());
        thrust::sort(&pt[0] + 4, &pt[0] + 8, sort_by_ordinate_asc());

        patch r0 =
        {
            pt[0].x,
            pt[1].x,
            pt[2].x,
            pt[3].x,

            pt[0].y,
            pt[1].y,
            pt[2].y,
            pt[3].y
        };

        patch r1 =
        {
            pt[4 + 0].x,
            pt[4 + 1].x,
            pt[4 + 2].x,
            pt[4 + 3].x,

            pt[4 + 0].y,
            pt[4 + 1].y,
            pt[4 + 2].y,
            pt[4 + 3].y
        };

        return thrust::make_tuple(r0, r1);

    }

    struct disconnect_patches
    {
        /*
        __device__ thrust::tuple< patch, patch> operator() (const patch& p1, const patch& p2)
        {
        }
        */

    };

    struct test_boxes
    {
        thrust::device_ptr< tab > m_tabs;

        test_boxes( thrust::device_ptr< tab > t) : m_tabs(t)
        {

        }

        __device__ bool operator() ( uint32_t i ) const
        {
            using namespace upper_triangular;

            tab* t = m_tabs.get();

            auto r = row(i);
            auto c = col(i);

            tab t0 = t[r];
            tab t1 = t[c];

            return intersect_bounding_boxes(t0.m_aabb, t1.m_aabb) ? 1 : 0 ;
        }
    };



    patches flip(const patches& p, tabs& t )
    {
        auto triangular_size = (p.size() * p.size() + 1) / 2;

        thrust::device_vector < bool> tests;
        tests.resize( triangular_size );

        auto cb = thrust::make_counting_iterator(0);
        auto ce = cb + triangular_size;

        thrust::transform(cb, ce, tests.begin(), test_boxes(&t[0]));

        thrust::copy(tests.begin(), tests.end(), std::ostream_iterator< bool >(std::cout, " "));

        return p;
    }
}


