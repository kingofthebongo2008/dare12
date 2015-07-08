
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

    float2 transform = float2 (1.0f / m_image_width, 1.0f / m_image_height);
    float2 center    = float2(-m_center_x, m_center_y );
    o.m_position     = (i.m_position) * transform;

    o.m_position = o.m_position *  2.0f - 1.0f;

    return o;
}

