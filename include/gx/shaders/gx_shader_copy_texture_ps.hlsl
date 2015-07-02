struct vs_output
{
    float4	position_ps : sv_position;
    float2	uv          : texcoord; 
};


Texture2D<float>     sampled_texture;
SamplerState         default_sampler;

float4 main( in  vs_output input) : sv_target
{
    //read (sample) what is in a texture and output it on the render target
    float data = sampled_texture.Sample(default_sampler, input.uv).r;
    return float4(data, data, data, 1.0f);
}
