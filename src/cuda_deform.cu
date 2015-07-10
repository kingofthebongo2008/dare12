#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>

namespace freeform
{
    struct deform_kernel
    {

        patch operator() (const patch& p )
        {
            return p;
        }
    };



    //sample the curve and obtain patches through curve interpolation as in the paper
    thrust::tuple< patches  > deform( const patches& p )
    {
        return p;
    }
}


