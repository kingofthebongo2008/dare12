#pragma once

#include "rendering_application.h"



namespace freeform
{
    /*
    class patch_application : public rendering_application
    {
        typedef rendering_application base;

    public:

        patch_application(const wchar_t* window_title, const imaging::cuda_texture& t, const graphic::patch_draw_info& gdi, const graphic::transform_info& transform_info) : base(window_title)
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

            //compose direct2d render target over the back buffer by rendering full screen quad that copies one texture onto another with alpha blending
            d3d11::ps_set_shader( device_context, m_copy_texture_ps );
            d3d11::ps_set_shader_resource(device_context, m_texture_view);
            d3d11::ps_set_sampler_state(device_context, m_point_sampler);

            //cull all back facing triangles
            d3d11::rs_set_state(device_context, m_cull_back_raster_state);

            d3d11::om_set_blend_state(device_context, m_premultiplied_alpha_state);

            //disable depth culling
            d3d11::om_set_depth_state(device_context, m_depth_disable_state);
            m_full_screen_draw.draw(device_context);


        }

    protected:


    };
    */
}
