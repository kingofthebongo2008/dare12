#pragma once

#include <cmath>

namespace gpu_deformation
{
    inline float distance(float x1, float y1, float x2, float y2)
    {

        //calcul la distance entre deux points.
        auto x = (x2 - x1) * (x2 - x1);
        auto y = (y2 - y1) * (y2 - y1);

        return sqrtf( x + y );
    }
}
