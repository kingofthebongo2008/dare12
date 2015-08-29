#pragma once


#include <d3d11/d3d11_helpers.h>

#include <gx/gx_render_resource.h>
#include <gx/gx_render_functions.h>
#include <gx/shaders/gx_shader_copy_texture.h>

#include "imaging_utils.h"

#include "rendering_render_item.h"
#include "graphic_types.h"

namespace freeform
{
    inline d3d11::itexture2d_ptr create_texture(ID3D11Device* device, const imaging::cuda_texture& t)
    {
        D3D11_TEXTURE2D_DESC d = {};

        d.Format = DXGI_FORMAT_R8_UNORM;
        d.ArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Height = t.get_height();
        d.Width = t.get_width();
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        auto proxy = t.get_pixels();

        D3D11_SUBRESOURCE_DATA sd = {};

        sd.pSysMem = proxy.get_pixels_cpu();
        sd.SysMemPitch = t.get_pitch();
        sd.SysMemSlicePitch = t.get_size();

        return d3d11::create_texture_2d(device, &d, &sd);
    }


    class render_item_texture : public render_item
    {
        public:

        render_item_texture(ID3D11Device* device, ID3D11DeviceContext* context, const imaging::cuda_texture& t ) : render_item(device, context)
        , m_texture( create_texture(device, t))
        , m_texture_view(d3d11::create_shader_resource_view(device, m_texture))
        , m_full_screen_draw(device)
        , m_copy_texture_ps(gx::create_shader_copy_texture_ps(device))
        {

        }

        render_item_texture(render_item_texture&& o) :
            m_texture(std::move(o.m_texture))
            , m_texture_view(std::move(o.m_texture_view))
            , m_full_screen_draw(std::move(o.m_full_screen_draw))
            , m_copy_texture_ps(std::move(o.m_copy_texture_ps))
        {

        }

        render_item_texture& operator = (render_item_texture&& o)
        {
            m_texture           = std::move(o.m_texture);
            m_texture_view      = std::move(o.m_texture_view);
            m_full_screen_draw  = std::move(o.m_full_screen_draw);
            m_copy_texture_ps   = std::move(o.m_copy_texture_ps);

            return *this;
        }

        virtual ~render_item_texture()
        {

        }
        
        private:

        d3d11::itexture2d_ptr                   m_texture;
        d3d11::ishaderresourceview_ptr          m_texture_view;

        gx::full_screen_draw                    m_full_screen_draw;
        gx::shader_copy_texture_ps              m_copy_texture_ps;


        void on_draw(const render_context* ctx, ID3D11DeviceContext* device_context) const override
        {
            //compose direct2d render target over the back buffer by rendering full screen quad that copies one texture onto another with alpha blending
            d3d11::ps_set_shader(device_context, m_copy_texture_ps);
            d3d11::ps_set_shader_resource(device_context, m_texture_view);
            d3d11::ps_set_sampler_state(device_context, ctx->m_point_sampler.get());

            //cull all back facing triangles
            d3d11::rs_set_state(device_context, ctx->m_cull_back_raster_state.get());

            d3d11::om_set_blend_state(device_context, ctx->m_premultiplied_alpha_state.get());

            //disable depth culling
            d3d11::om_set_depth_state(device_context, ctx->m_depth_disable_state.get());
            m_full_screen_draw.draw(device_context);
        }
    };

    class render_item_creator_texture : public render_item_creator
    {
        public: 
        explicit render_item_creator_texture(const imaging::cuda_texture& t) : m_t(t)
        {

        }

        private:
        const imaging::cuda_texture m_t;

        render_item_handle on_create(ID3D11Device* device, ID3D11DeviceContext* context) const override
        {
            return std::make_shared< render_item_texture>(render_item_texture(device, context, m_t));
        }
    };
}


