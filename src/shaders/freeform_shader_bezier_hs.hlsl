
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


//vertex shader just passes through
vs_bezier_output vs_main(vs_bezier_input i)
{
    vs_bezier_output o;
    o.m_position = i.m_position;

    return o;
}

struct hs_constant_data_output
{
    float edges[2] : SV_TessFactor;
};

hs_constant_data_output bezier_constant_hs( InputPatch< vs_bezier_output, 4 > ip, uint patch_id : SV_PrimitiveID )
{
    hs_constant_data_output o;
    o.edges[0] = 1.0;
    o.edges[1] = 8.0;

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
