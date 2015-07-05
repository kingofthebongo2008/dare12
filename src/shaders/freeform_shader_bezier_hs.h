#pragma once

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>


namespace freeform
{
    namespace details
    {
        inline d3d11::ihullshader_ptr   create_shader_bezier_hs(ID3D11Device* device)
        {
            using namespace d3d11;
            d3d11::ihullshader_ptr   shader;

            using namespace os::windows;

            //strange? see in the hlsl file
            static
            #include "freeform_shader_bezier_hs_compiled.hlsl"

                //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
            throw_if_failed<create_hull_shader>(device->CreateHullShader(freeform_shader_bezier_hs, sizeof(freeform_shader_bezier_hs), nullptr, &shader));
            return shader;
        }
    }

    class shader_bezier_hs final
    {

    public:
        shader_bezier_hs()
        {

        }

        explicit shader_bezier_hs(d3d11::ihullshader_ptr shader) : m_shader(shader)
        {

        }


        shader_bezier_hs(shader_bezier_hs&&  o) : m_shader(std::move(o.m_shader))
        {

        }

        operator ID3D11HullShader* () const
        {
            return m_shader.get();
        }

        shader_bezier_hs& operator=(shader_bezier_hs&& o)
        {
            m_shader = std::move(o.m_shader);
            return *this;
        }

        d3d11::ihullshader_ptr     m_shader;
    };

    inline shader_bezier_hs   create_shader_samples_hs(ID3D11Device* device)
    {
        return shader_bezier_hs(details::create_shader_bezier_hs(device));
    }

    inline std::future< shader_bezier_hs> create_shader_samples_hs_async(ID3D11Device* device)
    {
        return std::async(std::launch::async, create_shader_samples_hs, device);
    }
}


