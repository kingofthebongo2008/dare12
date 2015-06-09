#pragma once

#include <cmath>
#include <cstdint>

namespace gpu_deformation
{
    inline void voisinage(float x, float y, int32_t v1[], int32_t v2[])
    {
        int32_t x1 = static_cast<int32_t> (floorf(x));
        int32_t y1 = static_cast<int32_t> (floorf(y));

        //% Returns the "pixelique" coordinates of the point neighborhood( its size is 9 * 9 )? 8x8?

        const int32_t indices_v1[] = { -1,  -1, -1, 0,  0,  +1, +1, +1 };
        const int32_t indices_v2[] = { -1,  0,  1,  -1, 1,  -1, 0,  +1 };

        for ( uint32_t i = 0; i < 8; ++i )
        {
            v1[i] = x1 + indices_v1[i];
        }

        for (uint32_t i = 0; i < 8; ++i)
        {
            v2[i] = y1 + indices_v2[i];
        }

    }
}
