
#include "freeform_shaders_struct.h"


hs_constant_data_output bezier_constant_hs( )
{
    hs_constant_data_output o;
    o.edges[0] = 1.0;
    o.edges[1] = 64.0;

    return o;
}

[domain("isoline")]
[partitioning("integer")]
[outputtopology("line")]
[patchconstantfunc("bezier_constant_hs")]
[outputcontrolpoints(4)]
hs_bezier_output main(InputPatch<vs_bezier_output, 4> ip, uint id : SV_OutputControlPointID)
{
    hs_bezier_output o;

    o.m_position = ip[id].m_position;
    return o;
}



