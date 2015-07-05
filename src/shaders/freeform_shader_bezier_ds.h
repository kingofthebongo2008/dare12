#pragma once

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>


namespace freeform
{
    namespace details
    {
        inline d3d11::idomainshader_ptr   create_shader_bezier_ds(ID3D11Device* device)
        {
            using namespace d3d11;
            d3d11::idomainshader_ptr   shader;

            using namespace os::windows;

            //strange? see in the hlsl file
            static
            #include "freeform_shader_bezier_ds_compiled.hlsl"

                //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
            throw_if_failed<create_domain_shader>(device->CreateDomainShader(freeform_shader_bezier_ds, sizeof(freeform_shader_bezier_ds), nullptr, &shader));
            return shader;
        }
    }

    class shader_bezier_ds final
    {

    public:
        shader_bezier_ds()
        {

        }

        explicit shader_bezier_ds(d3d11::idomainshader_ptr shader) : m_shader(shader)
        {

        }


        shader_bezier_ds(shader_bezier_ds&&  o) : m_shader(std::move(o.m_shader))
        {

        }

        operator ID3D11DomainShader* () const
        {
            return m_shader.get();
        }

        shader_bezier_ds& operator=(shader_bezier_ds&& o)
        {
            m_shader = std::move(o.m_shader);
            return *this;
        }

        d3d11::idomainshader_ptr     m_shader;
    };

    inline shader_bezier_ds   create_shader_samples_ds(ID3D11Device* device)
    {
        return shader_bezier_ds(details::create_shader_bezier_ds(device));
    }

    inline std::future< shader_bezier_ds> create_shader_samples_ds_async(ID3D11Device* device)
    {
        return std::async(std::launch::async, create_shader_samples_ds, device);
    }
}


