#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>
#include <thrust/sort.h>

#include <math/math_vector.h>

#include "math_functions.h"


#include "cuda_print_utils.h"
#include "cuda_patches.h"

#include <algorithm>

namespace freeform
{
    struct multi_eval_patches2_kernel
    {
        float                          m_seuil;
        thrust::device_ptr<patch>      m_patches;
        thrust::device_ptr<uint32_t>   m_element_count;
        thrust::device_ptr<uint32_t>   m_keys;

        multi_eval_patches2_kernel(float seuil, thrust::device_ptr<patch> patches, thrust::device_ptr<uint32_t> element_count, thrust::device_ptr<uint32_t> key) :
            m_seuil(seuil)
            , m_patches(patches)
            , m_element_count(element_count)
            , m_keys(key)
        {

        }

        //split procedure
        __device__ void operator() (const thrust::tuple<patch, uint32_t>& t) const
        {
            auto p = thrust::get<0>(t);
            auto key = thrust::get<1>(t);

            auto d_0 = math::distance(p.x0, p.y0, p.x1, p.y1);
            auto d_1 = math::distance(p.x1, p.y1, p.x2, p.y2);
            auto d_2 = math::distance(p.x2, p.y2, p.x3, p.y3);

            auto m = fmax(fmax(d_0, d_1), d_2);

            if (m > m_seuil  )
            {
                float4 g1u = math::set(0.0f, 1.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f);
                float4 g2u = math::set(3.0f / 6.0f, 4.0f / 6.0f, 5.0f / 6.0f, 6.0f / 6.0f);

                auto   g1 = multi_eval_patch_3(p, g1u);
                auto   g2 = multi_eval_patch_3(p, g2u);

                auto   g4 = interpolate_curve(g1);
                auto   g5 = interpolate_curve(g2);

                auto old = atomicAdd(m_element_count.get(), 2);

                m_patches[old] = g4;
                m_patches[old + 1] = g5;

                //store the index number, so we can sort later to maintain the order
                m_keys[old] =  key *10 + 1;
                m_keys[old + 1] = key* 10 + 2;
            }
            else
            {
                auto old = atomicAdd(m_element_count.get(), 1);
                m_patches[old] = p;
                m_keys[old] = key *10;
            }
        }
    };

    

    //split procedure, generates new patches if they are too close
    patches split(const patches& p, float pixel_size)
    {
        using namespace thrust;

        patches                 n2;
        device_vector<uint32_t> element_count;
        device_vector<uint32_t> host_element_count;
        device_vector<uint32_t> keys;

        auto maxi = p.size();
        auto s = p.size();

        n2.resize( s * 2 );
        keys.resize(s * 2);

        element_count.resize(1);
        host_element_count.resize(1);

        auto cb = make_counting_iterator(0);
        auto ce = cb + s;

        auto b = make_zip_iterator(make_tuple(p.begin(), cb));
        auto e = make_zip_iterator(make_tuple(p.end(), ce));

        for_each(b, e, multi_eval_patches2_kernel(64 * pixel_size, &n2[0], &element_count[0], &keys[0]));

        //fetch the number of new patches that were added
        auto elements = element_count.front();
        copy_n(element_count.begin(), 1, host_element_count.begin());
        uint32_t new_size = host_element_count[0];
        
        //resize the arrays
        n2.resize(new_size);
        keys.resize(new_size);

        //the patches in the free form countour have order which must be maintained
        sort_by_key(keys.begin(), keys.end(), n2.begin());

        
        patches n3;
        n3.resize(n2.size());
        {
            auto b = make_counting_iterator(0);
            auto e = b + n2.size();
            thrust::for_each(b, e, average_patches(&n2[0], &n3[0], n2.size()));
        }
        

        return n3;
    }
}


