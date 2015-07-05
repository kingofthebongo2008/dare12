#ifndef __freeform_shaders_struct_h__
#define __freeform_shaders_struct_h__

struct vs_bezier_input
{
    float2  m_position  : bezier_position;
};

struct vs_bezier_output
{
    float2  m_position  : bezier_position;
};

struct hs_bezier_output
{
    float2  m_position  : bezier_position;
};

struct hs_constant_data_output
{
    float edges[2] : SV_TessFactor;
};


struct ds_output
{
    float4 m_position : SV_Position;
};


#endif
