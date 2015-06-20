#include "precompiled.h"

#include <thrust/transform.h>

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


    struct distance_control_points
    {
        float m_seuil;

        distance_control_points(float seuil) : m_seuil(seuil)
        {

        }

        __device__ float operator() (const patch& p) const
        {
            auto d_0 = math::distance(p.x0, p.y0, p.x1, p.y1);
            auto d_1 = math::distance(p.x1, p.y1, p.x2, p.y2);
            auto d_2 = math::distance(p.x2, p.y2, p.x3, p.y3);

            auto m   = fmax(fmax(d_0, d_1), d_2);

            return m;
        }
    };

    template <typename uint32_t patch_type>
    struct multi_eval_patches
    {
        multi_eval_patches()
        {

        }

        __device__ patch operator() (const patch& p) const
        {
            if (patch_type == 0)
            {
                float4 u = math::set(0.0f, 1.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f);
                return multi_eval_patch(p, u);
            }
            else
            {
                float4 u = math::set(3.0f / 6.0f, 4.0f / 6.0f, 5.0f / 6.0f, 6.0f / 6.0f);
                return multi_eval_patch(p, u);
            }
        }
    };

    //thrust::copy(g.begin(), g.end(), std::ostream_iterator< float >(std::cout, " "));

    patches test_distances(patches& n, patches& np)
    {
        thrust::device_vector<float> distances;
        thrust::device_vector<float> g;

        patches                      g1;
        patches                      g2;
        patches                      n2;

        auto maxi = np.size(); 

        distances.resize(maxi);

        g1.resize(maxi);
        g2.resize(maxi);
        n2.resize(maxi);

        thrust::transform(n.begin(), n.end(), distances.begin(), distance_control_points(13) );
        thrust::transform(n.begin(), n.end(), g1.begin(),        multi_eval_patches<0>());
        thrust::transform(n.begin(), n.end(), g2.begin(),        multi_eval_patches<1>());
        
        return g1;
    }
}


