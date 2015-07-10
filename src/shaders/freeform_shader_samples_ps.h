#pragma once

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>


namespace freeform
{
    namespace details
    {
        inline d3d11::ipixelshader_ptr   create_shader_samples_ps(ID3D11Device* device)
        {
            using namespace d3d11;
            d3d11::ipixelshader_ptr   shader;

            using namespace os::windows;

            //strange? see in the hlsl file
            static
            #include "freeform_shader_samples_ps_compiled.hlsl"

                //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
            throw_if_failed<create_pixel_shader>(device->CreatePixelShader(freeform_shader_samples_ps, sizeof(freeform_shader_samples_ps), nullptr, &shader));
            return shader;
        }
    }

    class shader_samples_ps final
    {

    public:
        shader_samples_ps()
        {

        }

        explicit shader_samples_ps(d3d11::ipixelshader_ptr shader) : m_shader(shader)
        {

        }


        shader_samples_ps(shader_samples_ps&&  o) : m_shader(std::move(o.m_shader))
        {

        }

        operator ID3D11PixelShader* () const
        {
            return m_shader.get();
        }

        shader_samples_ps& operator=(shader_samples_ps&& o)
        {
            m_shader = std::move(o.m_shader);
            return *this;
        }

        d3d11::ipixelshader_ptr     m_shader;
    };

    inline shader_samples_ps   create_shader_samples_ps(ID3D11Device* device)
    {
        return shader_samples_ps(details::create_shader_samples_ps(device));
    }

    inline std::future< shader_samples_ps> create_shader_samples_ps_async(ID3D11Device* device)
    {
        return std::async(std::launch::async, create_shader_samples_ps, device);
    }
}


