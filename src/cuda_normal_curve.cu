#include "precompiled.h"

#include <thrust/transform.h>
#include <thrust/adjacent_difference.h>

#include "freeform_patch.h"
#include <math_functions.h>

#include "math_functions.h"




inline std::ostream& operator<<(std::ostream& s, const float4& p)
{
    s << "x: " << p.x << " " << p.y << " " << p.z << " " << p.w << std::endl;
    return s;
}

namespace freeform
{

    //thrust::copy(g.begin(), g.end(), std::ostream_iterator< float >(std::cout, " "));

    struct normal_patch
    {
        float x;
        float y;
    };

    __device__ inline normal_patch operator+( const normal_patch& a, const normal_patch& b)
    {
        normal_patch r = { a.x + b.x, a.y + b.y };
        return r;
    }

    struct normal_patch_difference
    {
        __device__ normal_patch operator() (const normal_patch& a, const normal_patch& b) const
        {
            normal_patch r = { a.x - b.x, -a.y - b.y };
            return r;
        }

        __device__ normal_patch operator() (const thrust::tuple< normal_patch, normal_patch> & t  ) const
        {
            auto a = thrust::get<0>(t);
            auto b = thrust::get<1>(t);

            normal_patch r = { a.y - b.y, - ( a.x - b.x )  };

            return r;
        }
    };


    inline std::ostream& operator<<(std::ostream& s, const normal_patch& p)
    {
        s << "x: " << p.x << " " << p.y <<  std::endl;
        return s;
    }


    typedef thrust::device_vector< normal_patch > normal_patches;

    struct generate_normals
    {
        thrust::device_ptr<normal_patch> m_np;

        generate_normals(thrust::device_ptr<normal_patch> np) : m_np(np)
        {

        }

        __device__ void operator() ( const thrust::tuple< patch, uint32_t > & t  ) const
        {
            auto p = thrust::get<0>(t);
            auto i = thrust::get<1>(t);

            normal_patch* np = m_np.get();

            np[ 3 * i ].x = p.x0;
            np[ 3 * i ].y = p.y0;

            np[3 * i + 1].x = p.x1;
            np[3 * i + 1].y = p.y1;

            np[3 * i + 2].x = p.x2;
            np[3 * i + 2].y = p.y2;
        }
    };

    struct normalize
    {
        __device__ normal_patch operator() (const normal_patch & p) const
        {
            auto norm = sqrtf(p.x * p.x + p.y * p.y);
            normal_patch r = { p.x / norm, p.y / norm };
            return r;
        }
    };

    struct generate_normals2
    {
        thrust::device_ptr<normal_patch> m_np;
        thrust::device_ptr<patch>        m_p;

        generate_normals2(thrust::device_ptr<normal_patch> np, thrust::device_ptr<patch> p) : m_np(np), m_p(p)
        {

        }

        __device__ void operator() ( uint32_t i )  const
        {
            normal_patch* np = m_np.get();
            patch*        p  = m_p.get();

            auto x0 = np[3 * i].x;
            auto y0 = np[3 * i].y;

            auto x1 = np[3 * i + 1].x;
            auto y1 = np[3 * i + 1].y;

            auto x2 = np[3 * i + 2].x;
            auto y2 = np[3 * i + 2].y;

            auto x3 = np[3 * i + 3].x;
            auto y3 = np[3 * i + 3].y;

            patch r = { x0, x1, x2, x3, y0, y1, y2, y3 };

            p[i] = r;
        }
    };


    patches normal_curve(const patches& n)
    {
        normal_patches n2;
        n2.resize( n.size() * 3 );

        auto cb = thrust::make_counting_iterator(0);
        auto ce = cb + n.size();

        auto b = thrust::make_zip_iterator( thrust::make_tuple(n.begin(), cb) );
        auto e = thrust::make_zip_iterator( thrust::make_tuple(n.end(), ce) );

        thrust::for_each(b, e, generate_normals(&n2[0]));

        normal_patches n2a;
        n2a.resize( n2.size() + 2 );

        thrust::copy( n2.begin(), n2.end(), n2a.begin() + 1 );
        thrust::copy( n2.begin(), n2.begin() + 1, n2a.end() - 1  );
        thrust::copy( n2.end() - 1, n2.end(), n2a.begin() );

        normal_patches norm;
        norm.resize( n2.size() );

        auto na = thrust::make_zip_iterator(thrust::make_tuple(n2a.begin() + 2, n2a.begin()));
        auto ne = thrust::make_zip_iterator(thrust::make_tuple(n2a.begin() + 2 + n2.size(), n2a.begin() + n2.size()));

        thrust::transform(na, ne, norm.begin(), normal_patch_difference() );

        normal_patches normals;
        normals.resize(n2.size());
        thrust::transform(norm.begin(), norm.end(), normals.begin(), normalize());

        normal_patches n3;
        n3.resize(n2.size() + 1 );

        thrust::transform(normals.begin(), normals.end(), n2.begin(), n3.begin(), thrust::plus< normal_patch>());
        thrust::copy(n3.begin(), n3.begin() + 1, n3.end() - 1 );
        
      
        patches p;
        p.resize( n.size() );


        thrust::for_each(cb, cb + n3.size() / 3 , generate_normals2( &n3[0], &p[0] ) );

        //thrust::copy(p.begin(), p.end(), std::ostream_iterator< patch >(std::cout, " "));

        return p;
    }
}


