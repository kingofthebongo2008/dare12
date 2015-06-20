#include "precompiled.h"

#include <thrust/transform.h>
#include <thrust/merge.h>

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
    struct multi_eval_patches2
    {
       float                          m_seuil;
       thrust::device_ptr<patch>      m_patches;
       thrust::device_ptr<uint32_t>   m_element_count;

       multi_eval_patches2(float seuil, thrust::device_ptr<patch> patches, thrust::device_ptr<uint32_t> element_count) :
            m_seuil(seuil)
            , m_patches(patches)
            , m_element_count(element_count)
        {

        }

        __device__ void operator() ( const thrust::tuple<patch, patch> & t ) const
        {
            auto p = thrust::get<0>(t);
            auto pn = thrust::get<1>(t);

            auto d_0 = math::distance(p.x0, p.y0, p.x1, p.y1);
            auto d_1 = math::distance(p.x1, p.y1, p.x2, p.y2);
            auto d_2 = math::distance(p.x2, p.y2, p.x3, p.y3);

            auto m = fmax(fmax(d_0, d_1), d_2);

            if ( m > m_seuil )
            {
                float4 g1u = math::set(0.0f, 1.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f);
                float4 g2u = math::set(3.0f / 6.0f, 4.0f / 6.0f, 5.0f / 6.0f, 6.0f / 6.0f);
                
                auto   g1  = multi_eval_patch(p, g1u);
                auto   g2  = multi_eval_patch(p, g2u);

                auto old = atomicAdd(m_element_count.get(), 2);

                m_patches[old] = g1;
                m_patches[old + 1] = g2;

            }
            else
            {
                auto old = atomicAdd(m_element_count.get(), 1);
                m_patches[old] = pn;

            }
        }
    };

    //thrust::copy(g.begin(), g.end(), std::ostream_iterator< float >(std::cout, " "));

    patches test_distances(const patches& n, const patches& np)
    {
        patches                      n2;
        thrust::device_vector<uint32_t> element_count;
        thrust::device_vector<uint32_t> host_element_count;

        auto maxi = np.size(); 

        n2.resize(maxi * 2);
        element_count.resize(1);
        host_element_count.resize(1);

        auto b = thrust::make_zip_iterator( thrust::make_tuple ( n.begin(), np.begin() ) );
        auto e = thrust::make_zip_iterator( thrust::make_tuple(  n.end(),   np.end()   ) );

        thrust::for_each(b, e, multi_eval_patches2(13, &n2[0], &element_count[0]));

        auto elements = element_count.front();

        thrust::copy_n(element_count.begin(), 1, host_element_count.begin());

        uint32_t new_size = host_element_count[0];
            
        n2.resize(new_size);
        return n2;
    }
}


