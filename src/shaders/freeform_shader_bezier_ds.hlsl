
#include "freeform_shaders_struct.h"


[domain("isoline")]
ds_output main(hs_constant_data_output i, OutputPatch<hs_bezier_output, 4> op, float2 uv : SV_DomainLocation)
{
    ds_output o;

    float t = uv.x;

    float2 pos = pow(1.0f - t, 3.0f) * op[0].m_position + 3.0f * pow(1.0f - t, 2.0f) * t * op[1].m_position + 3.0f * (1.0f - t) * pow(t, 2.0f) * op[2].m_position + pow(t, 3.0f) * op[3].m_position;
    
    o.m_position = float4( pos, 0.0f, 1.0f );
    
    return o;
}



