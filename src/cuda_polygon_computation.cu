#include "precompiled.h"

#include <thrust/transform.h>
#include <thrust/merge.h>

#include "imaging_utils.h"
#include "cuda_imaging.h"

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

    //polygon computation + tab building with Ximin, Ximax, ..... for bounding box intersection tests

    struct compute_polygon
    {
        const thrust::device_ptr< point > m_n;

        compute_polygon(const thrust::device_ptr< point > n) : m_n(n)
        {

        }
        __device__ thrust::tuple< patch, tab > operator() (uint32_t i) const
        {
            const point* n = m_n.get();

            point  n0 = n[4 * i];
            point  n1 = n[4 * i + 1];
            point  n2 = n[4 * i + 2];
            point  n3 = n[4 * i + 3];

            float x0 = n0.x;
            float x1 = n1.x;
            float x2 = n2.x;
            float x3 = n3.x;

            float y0 = n0.y;
            float y1 = n1.y;
            float y2 = n2.y;
            float y3 = n3.y;

            freeform::patch p0 = { x0, x1, x2, x3, y0, y1, y2, y3 };
            freeform::patch p1 = cub_bezier_interpol(p0);

            float min_x = min4(p1.x0, p1.x1, p1.x2, p1.x3);
            float min_y = min4(p1.y0, p1.y1, p1.y2, p1.y3);
            float max_x = max4(p1.x0, p1.x1, p1.x2, p1.x3);
            float max_y = max4(p1.y0, p1.y1, p1.y2, p1.y3);

            float4  tb = math::set(min_x, max_x, min_y, max_y);
            tab     t(i, tb);

            return thrust::make_tuple( p1, t );
        }
    };

    thrust::tuple<patches, tabs > polygon_computation(points& n )
    {
        patches p;
        tabs    tabs;

        p.resize(n.size() / 4);
        tabs.resize(n.size() / 4);

        auto cb = thrust::make_counting_iterator(0);
        auto ce = cb + n.size() / 4 ;

        auto o = thrust::make_zip_iterator(thrust::make_tuple(p.begin(), tabs.begin()));

        thrust::transform(cb, ce, o, compute_polygon( &n[0] ));

        //thrust::copy(p.begin(), p.end(), std::ostream_iterator< patch >(std::cout, " "));
       
        return thrust::make_tuple(p, tabs);
    }
}


