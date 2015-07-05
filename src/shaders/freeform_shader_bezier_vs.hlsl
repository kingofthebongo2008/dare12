
#include "freeform_shaders_struct.h"


//vertex shader just passes through
vs_bezier_output main(in vs_bezier_input i)
{
    vs_bezier_output o;
    o.m_position = i.m_position;

    return o;
}

