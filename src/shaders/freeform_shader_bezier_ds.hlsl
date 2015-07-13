
#include "freeform_shaders_struct.h"


cbuffer transform_info : register(b0)
{
    float m_center_x;
    float m_center_y;

    float m_image_width;
    float m_image_height;
};


[domain("isoline")]
ds_output main(hs_constant_data_output i, OutputPatch<hs_bezier_output, 4> op, float2 uv : SV_DomainLocation)
{
    ds_output o;

    float t = uv.x;

    float2 pos = pow(1.0f - t, 3.0f) * op[0].m_position + 3.0f * pow(1.0f - t, 2.0f) * t * op[1].m_position + 3.0f * (1.0f - t) * pow(t, 2.0f) * op[2].m_position + pow(t, 3.0f) * op[3].m_position;

    //change x and y, since it appears that otherwise the display is not correct
    float2 transform = float2 (1.0f / m_image_width, 1.0f / m_image_height);

    //pos = pos * transform.xy;
    pos.y = 1.0f - pos.y;
    pos = pos *  2.0f - 1.0f;

    float x = pos.x;

    //pos.x = pos.y;
    //pos.y = x;

    o.m_position = float4( pos, 0.0f, 1.0f );


    
    return o;
}



