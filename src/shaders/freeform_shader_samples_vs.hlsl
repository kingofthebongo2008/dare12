
#include "freeform_shaders_struct.h"


cbuffer transform_info : register(b0)
{
    float m_center_x;
    float m_center_y;

    float m_image_width;
    float m_image_height;
};

struct vs_samples_input
{
    float2  m_position  : samples_position;
};

struct vs_samples_output
{
    float4 m_position : SV_Position;
};


//vertex shader just passes through
vs_samples_output main(in vs_samples_input i)
{
    float2 o;

    float2 transform = float2 (1.0f / m_image_width, 1.0f / m_image_height);

        o = i.m_position;// *transform;

    //invert y
    o.y   = 1.0f - o.y;
    o     = o *  2.0f - 1.0f;   //transform into perspective space

    float x = o.x;
    
    vs_samples_output r;

    r.m_position = float4(o, 0.0f, 1.0f);

    return r;
}

