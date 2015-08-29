#pragma once

#include <string>

#include <d3d11/d3d11_helpers.h>

#include <d2d/d2d_helpers.h>
#include <d2d/dwrite_helpers.h>

#include <gx/gx_default_application.h>
#include <gx/gx_render_resource.h>
#include <gx/gx_render_functions.h>
#include <gx/gx_view_port.h>

#include <gx/shaders/gx_shader_copy_texture.h>
#include <gx/gx_constant_buffer_helper.h>

#include <sys/sys_profile_timer.h>

#include "imaging_utils.h"



#include "graphic_types.h"



namespace freeform
{

    class rendering_application : public gx::default_application
    {
        typedef gx::default_application base;

    public:

        rendering_application(const wchar_t* window_title) : base(window_title)
            , m_d2d_factory(d2d::create_d2d_factory_single_threaded())
            , m_dwrite_factory(dwrite::create_dwrite_factory())
            , m_text_format(dwrite::create_text_format(m_dwrite_factory))
            , m_d2d_resource(gx::create_render_target_resource(m_context.m_device, 8, 8, DXGI_FORMAT_R8G8B8A8_UNORM))
            , m_opaque_state(gx::create_opaque_blend_state(m_context.m_device))
            , m_premultiplied_alpha_state(gx::create_premultiplied_alpha_blend_state(m_context.m_device))
            , m_cull_back_raster_state(gx::create_cull_back_rasterizer_state(m_context.m_device))
            , m_cull_none_raster_state(gx::create_cull_none_rasterizer_state(m_context.m_device))
            , m_depth_disable_state(gx::create_depth_test_disable_state(m_context.m_device))
            , m_point_sampler(gx::create_point_sampler_state(m_context.m_device))
            , m_elapsed_update_time(0.0)
        {
        }

    protected:
        virtual void on_render_scene()
        {

        }

        void render_scene()
        {
            on_render_scene();
        }

        virtual void on_update_scene()
        {

        }

        void update_scene()
        {
            on_update_scene();
        }

        void on_update()
        {
            sys::profile_timer timer;

            update_scene();

            //Measure the update time and pass it to the render function
            m_elapsed_update_time = timer.milliseconds();
        }

        void on_render_frame()
        {
            sys::profile_timer timer;

            //get immediate context to submit commands to the gpu
            auto device_context = m_context.m_immediate_context.get();

            device_context->VSSetShader(nullptr, nullptr, 0);
            device_context->PSSetShader(nullptr, nullptr, 0);
            device_context->HSSetShader(nullptr, nullptr, 0);
            device_context->DSSetShader(nullptr, nullptr, 0);
            device_context->GSSetShader(nullptr, nullptr, 0);

            //set render target as the back buffer, goes to the operating system
            d3d11::om_set_render_target(device_context, m_back_buffer_render_target);

            on_render_scene();

            //set a view port for rendering
            D3D11_VIEWPORT v = m_view_port;
            device_context->RSSetViewports(1, &v);

            //clear the back buffer
            const float fraction = 25.0f / 128.0f;
            d3d11::clear_render_target_view(device_context, m_back_buffer_render_target, math::set(fraction, fraction, fraction, 1.0f));

        }

        void on_resize(uint32_t width, uint32_t height)
        {
            //Reset back buffer render targets
            m_back_buffer_render_target.reset();

            base::on_resize(width, height);

            //Recreate the render target to the back buffer again
            m_back_buffer_render_target = d3d11::create_render_target_view(m_context.m_device, dxgi::get_buffer(m_context.m_swap_chain));

            //Reset view port dimensions
            m_view_port.set_dimensions(width, height);

        }



    protected:

        gx::render_target_resource              m_d2d_resource;

        d2d::ifactory_ptr                       m_d2d_factory;
        dwrite::ifactory_ptr                    m_dwrite_factory;

        d2d::irendertarget_ptr		            m_d2d_render_target;
        d2d::isolid_color_brush_ptr             m_brush;
        d2d::isolid_color_brush_ptr             m_brush2;
        dwrite::itextformat_ptr                 m_text_format;

        d3d11::id3d11rendertargetview_ptr       m_back_buffer_render_target;

        d3d11::iblendstate_ptr                  m_opaque_state;
        d3d11::iblendstate_ptr                  m_premultiplied_alpha_state;

        d3d11::iblendstate_ptr                  m_alpha_blend_state;
        d3d11::irasterizerstate_ptr             m_cull_back_raster_state;
        d3d11::irasterizerstate_ptr             m_cull_none_raster_state;

        d3d11::idepthstencilstate_ptr           m_depth_disable_state;
        d3d11::isamplerstate_ptr                m_point_sampler;

        gx::view_port                           m_view_port;

        double                                  m_elapsed_update_time;
    };
}
