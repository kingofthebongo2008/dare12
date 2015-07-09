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
        __device__ __host__ sort_by_abscissa_asc()
        {

        }

        __device__ __host__ bool operator()( const point& a, const point& b ) const
        {
            return a.x < b.x;
        }
    };

    struct sort_by_ordinate_asc
    {
        __device__ __host__ sort_by_ordinate_asc()
        {

        }

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

            return intersect_bounding_boxes(t0.m_aabb, t1.m_aabb) ? 1 : 0;
        }
    };

    struct disconnect_patches_kernel
    {
        thrust::device_ptr< patch >      m_patches;
        thrust::device_ptr< patch >      m_patches_in;
        thrust::device_ptr< patch >      m_patches_out;

        thrust::device_ptr<uint32_t>     m_patches_in_element_count;
        thrust::device_ptr<uint32_t>     m_patches_out_element_count;
        

        disconnect_patches_kernel
            ( 
                thrust::device_ptr< patch >      patches,
                thrust::device_ptr< patch >      patches_in,
                thrust::device_ptr< patch >      patches_out,
        
                thrust::device_ptr<uint32_t>     patches_in_element_count,
                thrust::device_ptr<uint32_t>     patches_out_element_count
            ) :
                m_patches(patches)
                , m_patches_in(patches_in)
                , m_patches_out(patches_out)
                , m_patches_in_element_count(patches_in_element_count)
                , m_patches_out_element_count(patches_out_element_count)
            {

            }

        __device__ void operator() ( thrust::tuple<uint32_t, bool> t )
        {
            auto p           = thrust::get<0>(t);
            auto intersected = thrust::get<1>(t);

            if (intersected)
            {
                using namespace upper_triangular;

                //get indices of patches i and j
                auto i = row(p);
                auto j = col(p);

                auto pi = m_patches[i];
                auto pj = m_patches[j];

                auto new_patches = disconnect_patches(pi, pj);

                auto index_in  = atomicAdd(m_patches_in_element_count.get(), 1);
                auto index_out = atomicAdd(m_patches_out_element_count.get(), 1);

                m_patches_out[index_out] = thrust::get<0>(new_patches);
                m_patches_in[index_in]   = thrust::get<1>(new_patches);
            }
        }
    };


    struct cub_bezier_interpol_kernel
    {
        __device__ patch operator() (const patch& p) const
        {
            return cub_bezier_interpol(p);
        }
    };

    thrust::tuple< patches, patches > flip(patches& p, tabs& t)
    {
        auto triangular_size = ( p.size() * ( p.size() - 1) / 2 ) ;

        thrust::device_vector < bool> tests;
        tests.resize( triangular_size );

        auto cb = thrust::make_counting_iterator(0);
        auto ce = cb + triangular_size;

        thrust::transform(cb, ce, tests.begin(), test_boxes(&t[0]));

        thrust::copy(tests.begin(), tests.end(), std::ostream_iterator< bool >(std::cout, " "));

        patches outside;
        patches inside;

        //reserve space
        outside.resize( triangular_size );
        inside.resize(  triangular_size );

        auto zb = thrust::make_zip_iterator(thrust::make_tuple(cb, tests.begin()));
        auto ze = thrust::make_zip_iterator(thrust::make_tuple(ce, tests.end()));

        thrust::device_vector<uint32_t> element_count_in;
        thrust::device_vector<uint32_t> element_count_out;

        element_count_in.resize(1);
        element_count_out.resize(1);

        thrust::for_each(zb, ze, disconnect_patches_kernel(&p[0], &inside[0], &outside[0], &element_count_in[0], &element_count_out[0]));

        auto out_size   = static_cast<uint32_t> (element_count_out[0]);
        auto in_size    = static_cast<uint32_t> (element_count_in[0]);

        
        outside.resize(out_size);
        inside.resize(in_size);

        //thrust::copy(tests.begin(), tests.end(), std::ostream_iterator< bool >(std::cout, " "));
        //thrust::copy(outside.begin(), outside.end(), std::ostream_iterator< const patch& >(std::cout, " "));

        patches displaced;
        displaced.resize(p.size());

        thrust::transform(p.begin(), p.end(), displaced.begin(), cub_bezier_interpol_kernel());


        return thrust::make_tuple( p, displaced );
    }
}


