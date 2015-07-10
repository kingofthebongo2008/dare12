
#include "freeform_shaders_struct.h"


cbuffer transform_info : register(b0)
{
    float m_center_x;
    float m_center_y;

    float m_image_width;
    float m_image_height;
};


//vertex shader just passes through
vs_bezier_output main(in vs_bezier_input i)
{
    vs_bezier_output o;
    o.m_position = i.m_position;

    return o;
}

