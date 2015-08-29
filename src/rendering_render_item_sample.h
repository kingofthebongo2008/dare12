#pragma once


#include <d3d11/d3d11_helpers.h>

#include <gx/gx_constant_buffer_helper.h>

#include "rendering_render_item.h"
#include "graphic_types.h"

#include "shaders/freeform_shader_samples_vs.h"
#include "shaders/freeform_shader_samples_ps.h"

namespace freeform
{
    class render_item_sample : public render_item
    {
        private:

        shader_samples_vs                       m_samples_vs;
        d3d11::ipixelshader_ptr                 m_samples_ps;
        d3d11::ibuffer_ptr                      m_control_points_buffer;
        size_t                                  m_control_points_count;
        d3d11::ibuffer_ptr                      m_transform_buffer;

        graphic::transform_info                 m_transform_info;
        shader_samples_layout                   m_samples_ia_layout;

        public:

        render_item_sample(ID3D11Device* device, ID3D11DeviceContext* context, const graphic::patch_draw_info& gdi, const graphic::transform_info& transform_info) : render_item(device, context)
        , m_transform_info(transform_info)
        , m_transform_buffer(d3d11::create_constant_buffer(device, sizeof(graphic::transform_info)))
        {
            auto r0 = create_shader_samples_vs_async(device);
            auto r3 = create_shader_samples_ps_async(device);

            r0.wait();
            r3.wait();

            m_samples_vs = std::move(r0.get());
            m_samples_ps = std::move(r3.get());

            m_samples_ia_layout = std::move(shader_samples_layout(device, m_samples_vs));
            m_control_points_buffer = d3d11::create_default_vertex_buffer(device, gdi.get_patches(), graphic::get_patch_size(gdi));
            m_control_points_count = gdi.get_count();
            gx::constant_buffer_update(context, m_transform_buffer, m_transform_info);

        }

        virtual ~render_item_sample()
        {

        }

        render_item_sample( render_item_sample&& o ) :
            m_samples_vs( std::move( o.m_samples_vs))
            , m_samples_ps(std::move(o.m_samples_ps))
            , m_control_points_buffer(std::move(o.m_control_points_buffer))
            , m_control_points_count(std::move(o.m_control_points_count))
            , m_transform_buffer(std::move(o.m_transform_buffer))
            , m_transform_info(std::move(o.m_transform_info))
            , m_samples_ia_layout(std::move(o.m_samples_ia_layout))
        {

        }

        render_item_sample& operator= (render_item_sample&& o)
        {
            m_samples_vs = (std::move(o.m_samples_vs));
            m_samples_ps = (std::move(o.m_samples_ps));
            m_control_points_buffer = (std::move(o.m_control_points_buffer));
            m_control_points_count = (std::move(o.m_control_points_count));
            m_transform_buffer = (std::move(o.m_transform_buffer));
            m_transform_info = (std::move(o.m_transform_info));
            m_samples_ia_layout = (std::move(o.m_samples_ia_layout));

            return *this;
        }

        
        private:


        void on_draw(const render_context* ctx, ID3D11DeviceContext* device_context) const override
        {
            uint32_t strides[] = { 2 * sizeof(float) }; // 3 dimensions per control point (x,y,z)
            uint32_t offsets[] = { 0 };


            d3d11::ia_set_primitive_topology(device_context, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP);
            d3d11::ia_set_input_layout(device_context, m_samples_ia_layout);

            ID3D11Buffer* buffers[] =
            {
                m_control_points_buffer.get()
            };

            device_context->IASetVertexBuffers(0, 1, buffers, strides, offsets);

            d3d11::vs_set_shader(device_context, m_samples_vs);
            d3d11::ps_set_shader(device_context, m_samples_ps);
            d3d11::gs_set_shader(device_context, nullptr);

            ID3D11Buffer* cbuffers[] =
            {
                m_transform_buffer.get()
            };

            device_context->VSSetConstantBuffers(0, 1, cbuffers);
            device_context->Draw(m_control_points_count * 4, 0);

            d3d11::om_set_blend_state(device_context, ctx->m_opaque_state.get());
        }
    };

    class render_item_creator_sample : public render_item_creator
    {
    public:

        explicit render_item_creator_sample(const graphic::patch_draw_info& gdi, const graphic::transform_info&  transform_info) : m_gdi(gdi)
        , m_transform_info( transform_info )
        {

        }

    private:

        const graphic::patch_draw_info m_gdi;
        const graphic::transform_info  m_transform_info;

        render_item_handle on_create(ID3D11Device* device, ID3D11DeviceContext* context) const override
        {
            return std::make_shared< render_item_sample>(render_item_sample(device, context, m_gdi, m_transform_info));
        }
    };
}


