#pragma once


#include <d3d11/d3d11_helpers.h>

#include <gx/gx_constant_buffer_helper.h>

#include "rendering_render_item.h"
#include "graphic_types.h"

#include "shaders/freeform_shader_bezier_vs.h"
#include "shaders/freeform_shader_bezier_hs.h"
#include "shaders/freeform_shader_bezier_ds.h"
#include "shaders/freeform_shader_bezier_ps.h"

namespace freeform
{
    class render_item_patch : public render_item
    {
        private:

        shader_bezier_vs                        m_bezier_vs;
        d3d11::ihullshader_ptr                  m_bezier_hs;
        d3d11::idomainshader_ptr                m_bezier_ds;
        d3d11::ipixelshader_ptr                 m_bezier_ps;
        d3d11::ibuffer_ptr                      m_control_points_buffer;
        size_t                                  m_control_points_count;
        d3d11::ibuffer_ptr                      m_transform_buffer;

        graphic::transform_info                 m_transform_info;
        shader_bezier_layout                    m_bezier_ia_layout;

        public:

        render_item_patch(ID3D11Device* device, ID3D11DeviceContext* context, const graphic::patch_draw_info& gdi, const graphic::transform_info& transform_info) : render_item(device, context)
        , m_transform_info(transform_info)
        , m_transform_buffer(d3d11::create_constant_buffer(device, sizeof(graphic::transform_info)))

        {
            auto r0 = create_shader_bezier_vs_async(device);
            auto r1 = create_shader_bezier_hs_async(device);
            auto r2 = create_shader_bezier_ds_async(device);
            auto r3 = create_shader_bezier_ps_async(device);

            r0.wait();
            r1.wait();
            r2.wait();
            r3.wait();

            m_bezier_vs = std::move(r0.get());
            m_bezier_hs = std::move(r1.get());
            m_bezier_ds = std::move(r2.get());
            m_bezier_ps = std::move(r3.get());

            m_bezier_ia_layout = std::move(shader_bezier_layout(device, m_bezier_vs));
            m_control_points_buffer = d3d11::create_default_vertex_buffer(device, gdi.get_patches(), graphic::get_patch_size(gdi));
            m_control_points_count = gdi.get_count();
            gx::constant_buffer_update(context, m_transform_buffer, m_transform_info);
        }

        render_item_patch(const render_item_patch&& o) :
            m_bezier_vs(std::move(o.m_bezier_vs))
            , m_bezier_hs(std::move(o.m_bezier_hs))
            , m_bezier_ds(std::move(o.m_bezier_ds))
            , m_bezier_ps(std::move(o.m_bezier_ps))
            , m_control_points_buffer(std::move(o.m_control_points_buffer))
            , m_control_points_count(std::move(o.m_control_points_count))
            , m_transform_buffer(std::move(o.m_transform_buffer))
            , m_transform_info(std::move(o.m_transform_info))
            , m_bezier_ia_layout(std::move(o.m_bezier_ia_layout))
        {
            
        }

        render_item_patch& operator= (const render_item_patch&& o)
        {
            m_bezier_vs = (std::move(o.m_bezier_vs));
            m_bezier_hs = (std::move(o.m_bezier_hs));
            m_bezier_ds = (std::move(o.m_bezier_ds));
            m_bezier_ps = (std::move(o.m_bezier_ps));
            m_control_points_buffer = (std::move(o.m_control_points_buffer));
            m_control_points_count = (std::move(o.m_control_points_count));
            m_transform_buffer = (std::move(o.m_transform_buffer));
            m_transform_info = (std::move(o.m_transform_info));
            m_bezier_ia_layout = (std::move(o.m_bezier_ia_layout));

            return *this;
        }

        virtual ~render_item_patch()
        {

        }
        
        private:

        void on_draw(const render_context* ctx, ID3D11DeviceContext* device_context) const override
        {

            d3d11::om_set_blend_state(device_context, ctx->m_opaque_state.get());

            uint32_t strides[] = { 2 * sizeof(float) }; // 3 dimensions per control point (x,y,z)
            uint32_t offsets[] = { 0 };

            device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST); // 4 control points per primitive
            device_context->IASetInputLayout(m_bezier_ia_layout );

            ID3D11Buffer* buffers[] =
            {
                m_control_points_buffer.get()
            };

            device_context->IASetVertexBuffers(0, 1, buffers, strides, offsets);

            d3d11::vs_set_shader(device_context, m_bezier_vs);
            d3d11::ps_set_shader(device_context, m_bezier_ps);
            d3d11::gs_set_shader(device_context, nullptr);
            d3d11::hs_set_shader(device_context, m_bezier_hs);
            d3d11::ds_set_shader(device_context, m_bezier_ds);

            ID3D11Buffer* cbuffers[] =
            {
                m_transform_buffer.get()
            };

            device_context->VSSetConstantBuffers(0, 1, cbuffers);
            device_context->DSSetConstantBuffers(0, 1, cbuffers);

            device_context->Draw(m_control_points_count * 4, 0);
        }
    };

    class render_item_creator_patch : public render_item_creator
    {
    public:

        explicit render_item_creator_patch(const graphic::patch_draw_info& gdi, const graphic::transform_info&  transform_info) : m_gdi(gdi)
            , m_transform_info(transform_info)
        {

        }

    private:

        const graphic::patch_draw_info m_gdi;
        const graphic::transform_info  m_transform_info;

        render_item_handle on_create(ID3D11Device* device, ID3D11DeviceContext* context) const override
        {
            return std::make_shared< render_item_patch>(render_item_patch(device, context, m_gdi, m_transform_info));
        }
    };
}


