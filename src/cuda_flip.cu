#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>


inline std::ostream& operator<<(std::ostream& s, const float4& p)
{
    s << "x: " << p.x << " " << p.y << " " << p.z << " " << p.w << std::endl;
    return s;
}

namespace freeform
{
    patches flip(const patches& p, const tabs& t )
    {
        return p;
    }
}


