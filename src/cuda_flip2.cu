#include "precompiled.h"

#include "freeform_patch.h"

#include <thrust/transform.h>
#include <thrust/sort.h>

#include <math/math_vector.h>

#include "math_functions.h"
#include <algorithm>

#include "cuda_aabb.h"
#include "cuda_patches.h"
#include "cuda_print_utils.h"

#include "cuda_collision_detection.h"
#include "cuda_math_2d.h"

namespace freeform
{
    patches flip2(patches& p)
    {
        return p;
    }
}


