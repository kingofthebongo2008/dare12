#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>

#include <math/math_vector.h>

#include "math_functions.h"


#include <algorithm>

namespace freeform
{
    struct sample_patches_kernel
    {
        __device__ sample operator() (const patch& p)
        {
            float4 t = math::set(0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 3.0f);
            sample s = multi_eval_patch_3(p, t);             //sample the bezier curve

            return s;
        }
    };

    samples sample_patches(const patches& p)
    {
        using namespace thrust;
        samples s;
        s.resize(p.size());

        transform(p.begin(), p.end(), s.begin(), sample_patches_kernel());

        return s;
    }

    
}


