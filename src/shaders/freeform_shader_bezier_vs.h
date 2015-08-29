#pragma once

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>


namespace freeform
{
    typedef std::tuple< d3d11::ivertexshader_ptr, const void*, uint32_t> vertex_shader_create_info;

    namespace details
    {
        inline vertex_shader_create_info   create_shader_bezier_vs(ID3D11Device* device)
        {
            using namespace d3d11;
            d3d11::ivertexshader_ptr   shader;

            using namespace os::windows;

            //strange? see in the hlsl file
            static
            #include "freeform_shader_bezier_vs_compiled.hlsl"

                //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
            throw_if_failed<create_vertex_shader>(device->CreateVertexShader(freeform_shader_bezier_vs, sizeof(freeform_shader_bezier_vs), nullptr, &shader));

            return std::make_tuple(shader, &freeform_shader_bezier_vs[0], static_cast<uint32_t> (sizeof(freeform_shader_bezier_vs)));
        }
    }

    class shader_bezier_vs final
    {

    public:
        shader_bezier_vs()
        {

        }

        explicit shader_bezier_vs(vertex_shader_create_info info) :
            m_shader(std::get<0>(info))
            , m_code(std::get<1>(info))
            , m_code_size(std::get<2>(info))
        {
        }



        shader_bezier_vs(shader_bezier_vs&&  o) : 
            m_shader(std::move(o.m_shader))
            , m_code(std::move(o.m_code))
            , m_code_size(std::move(o.m_code_size))
        {

        }

        operator ID3D11VertexShader* () const
        {
            return m_shader.get();
        }

        shader_bezier_vs& operator=(shader_bezier_vs&& o)
        {
            m_shader = std::move(o.m_shader);
            m_code = std::move(o.m_code);
            m_code_size = std::move(o.m_code_size);
            return *this;
        }

        d3d11::ivertexshader_ptr     m_shader;
        const void*                  m_code;
        uint32_t                     m_code_size;
    };

    inline shader_bezier_vs   create_shader_bezier_vs(ID3D11Device* device)
    {
        return shader_bezier_vs(details::create_shader_bezier_vs(device));
    }

    inline std::future< shader_bezier_vs> create_shader_bezier_vs_async(ID3D11Device* device)
    {
        return std::async(std::launch::async, create_shader_bezier_vs, device);
    }

    class shader_bezier_layout final
    {
        public:

        shader_bezier_layout()
        {

        }

        shader_bezier_layout(ID3D11Device* device, const shader_bezier_vs& shader)
        {
            D3D11_INPUT_ELEMENT_DESC desc[] =
            {
                { "bezier_position", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
            };

            //create description of the vertices that will go into the vertex shader
            os::windows::throw_if_failed<d3d11::create_input_layout>(device->CreateInputLayout(&desc[0], sizeof(desc) / sizeof(desc[0]), shader.m_code, shader.m_code_size, &m_input_layout));
        }

        operator ID3D11InputLayout*()
        {
            return m_input_layout.get();
        }

        operator ID3D11InputLayout*() const
        {
            return m_input_layout.get();
        }

        d3d11::iinputlayout_ptr	m_input_layout;
    };

}


