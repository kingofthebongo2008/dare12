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

#include "shaders/freeform_shader_bezier_vs.h"
#include "shaders/freeform_shader_bezier_hs.h"
#include "shaders/freeform_shader_bezier_ds.h"
#include "shaders/freeform_shader_bezier_ps.h"

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

    class sample_application : public gx::default_application
    {
        typedef gx::default_application base;

    public:

        sample_application(const wchar_t* window_title, const imaging::cuda_texture& t, const graphic::patch_draw_info& gdi, const graphic::transform_info& transform_info) : base(window_title)
            , m_d2d_factory(d2d::create_d2d_factory_single_threaded())
            , m_dwrite_factory(dwrite::create_dwrite_factory())
            , m_text_format(dwrite::create_text_format(m_dwrite_factory))
            , m_full_screen_draw(m_context.m_device)
            , m_copy_texture_ps(gx::create_shader_copy_texture_ps(m_context.m_device))
            , m_d2d_resource(gx::create_render_target_resource(m_context.m_device, 8, 8, DXGI_FORMAT_R8G8B8A8_UNORM))
            , m_opaque_state(gx::create_opaque_blend_state(m_context.m_device))
            , m_premultiplied_alpha_state(gx::create_premultiplied_alpha_blend_state(m_context.m_device))
            , m_cull_back_raster_state(gx::create_cull_back_rasterizer_state(m_context.m_device))
            , m_cull_none_raster_state(gx::create_cull_none_rasterizer_state(m_context.m_device))
            , m_depth_disable_state(gx::create_depth_test_disable_state(m_context.m_device))
            , m_point_sampler(gx::create_point_sampler_state(m_context.m_device))
            , m_elapsed_update_time(0.0)
            , m_texture(create_texture( m_context.m_device, t ))
            , m_texture_view( d3d11::create_shader_resource_view( m_context.m_device, m_texture) )
            , m_transform_info( transform_info )
            , m_transform_buffer(d3d11::create_constant_buffer( m_context.m_device, sizeof(graphic::transform_info)))
        {

            auto r0 = create_shader_samples_vs_async( m_context.m_device );
            auto r1 = create_shader_samples_hs_async(m_context.m_device);
            auto r2 = create_shader_samples_ds_async(m_context.m_device);
            auto r3 = create_shader_samples_ps_async(m_context.m_device);


            r0.wait();
            r1.wait();
            r2.wait();
            r3.wait();

            m_bezier_vs = std::move( r0.get() );
            m_bezier_hs = std::move( r1.get() );
            m_bezier_ds = std::move( r2.get());
            m_bezier_ps = std::move( r3.get());

            m_bezier_ia_layout = std::move(shader_bezier_layout(m_context.m_device.get(), m_bezier_vs));

            //draw the control points
            float constrol_points[] =
            {
                -1.0f,  -1.0f,
                4.0f,   -1.0f,
                -4.0f,  1.0f,
                1.0f,   1.0f,

                -2.0f, -0.8f,
                4.0f, -1.0f,
                -4.0f, 1.0f,
                2.0f, 0.8f
            };

            m_control_points_buffer = d3d11::create_default_vertex_buffer(m_context.m_device.get(), gdi.get_patches(), graphic::get_patch_size(gdi));
            m_control_points_count = gdi.get_count();
            gx::constant_buffer_update(m_context.m_immediate_context, m_transform_buffer, m_transform_info);
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

            d3d11::om_set_blend_state(device_context, m_opaque_state);


            uint32_t strides[] = { 2 * sizeof(float) }; // 3 dimensions per control point (x,y,z)
            uint32_t offsets[] = { 0 };

            device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST); // 4 control points per primitive
            device_context->IASetInputLayout(m_bezier_ia_layout);

            ID3D11Buffer* buffers[] =
            {
                m_control_points_buffer.get()
            };

            device_context->IASetVertexBuffers(0, 1, buffers, strides, offsets);


            device_context->VSSetShader(m_bezier_vs, nullptr, 0);
            device_context->PSSetShader(m_bezier_ps, nullptr, 0);
            device_context->HSSetShader(m_bezier_hs, nullptr, 0);
            device_context->DSSetShader(m_bezier_ds, nullptr, 0);
            device_context->GSSetShader(nullptr, nullptr, 0);


            ID3D11Buffer* cbuffers[] =
            {
                m_transform_buffer.get()
            };

            device_context->VSSetConstantBuffers(0, 1, cbuffers);
            device_context->DSSetConstantBuffers(0, 1, cbuffers);

            device_context->Draw( m_control_points_count * 4 , 0);
           
        }

        void on_resize(uint32_t width, uint32_t height)
        {
            //Reset back buffer render targets
            m_back_buffer_render_target.reset();

            base::on_resize(width, height);

            //Recreate the render target to the back buffer again
            m_back_buffer_render_target = d3d11::create_render_target_view(m_context.m_device, dxgi::get_buffer(m_context.m_swap_chain));

            /*
            using namespace os::windows;

            //Direct 2D resources
            m_d2d_resource = gx::create_render_target_resource( m_context.m_device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM );
            m_d2d_render_target = d2d::create_render_target( m_d2d_factory, m_d2d_resource );
            m_brush = d2d::create_solid_color_brush( m_d2d_render_target );
            m_brush2 = d2d::create_solid_color_brush2(m_d2d_render_target);
            */
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

        gx::full_screen_draw                    m_full_screen_draw;
        gx::shader_copy_texture_ps              m_copy_texture_ps;
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

        d3d11::itexture2d_ptr                   m_texture;
        d3d11::ishaderresourceview_ptr          m_texture_view;

        shader_bezier_vs                        m_bezier_vs;
        d3d11::ihullshader_ptr                  m_bezier_hs;
        d3d11::idomainshader_ptr                m_bezier_ds;
        d3d11::ipixelshader_ptr                 m_bezier_ps;
        d3d11::ibuffer_ptr                      m_control_points_buffer;
        size_t                                  m_control_points_count;
        d3d11::ibuffer_ptr                      m_transform_buffer;

        graphic::transform_info                 m_transform_info;
        shader_bezier_layout                    m_bezier_ia_layout;
    };
}
