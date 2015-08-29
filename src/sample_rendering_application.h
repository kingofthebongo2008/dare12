#pragma once

#include "rendering_application.h"

namespace freeform
{
    namespace details
    {
        std::vector< render_item_handle> create_render_items(ID3D11Device* device, ID3D11DeviceContext* context, const std::vector< std::shared_ptr<render_item_creator> >& items )
        {
            std::vector< render_item_handle> render_items;

            for_each(std::cbegin(items), std::cend(items), [device, context, &render_items](const std::shared_ptr<render_item_creator> & c)
            {
                render_items.push_back(c->create(device, context));

            });

            return std::move(render_items);
        }
    }

    class sample_application : public rendering_application
    {
        typedef rendering_application base;

    public:

        sample_application(const wchar_t* window_title, const std::vector< std::shared_ptr<render_item_creator> >& items) : base(window_title)
        , m_render_items (std::move(details::create_render_items(m_context.m_device.get(), m_context.m_immediate_context.get(), items)))
        {
            
        }

    protected:
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

            //set a view port for rendering
            D3D11_VIEWPORT v = m_view_port;
            device_context->RSSetViewports(1, &v);

            //clear the back buffer
            const float fraction = 25.0f / 128.0f;
            d3d11::clear_render_target_view(device_context, m_back_buffer_render_target, math::set(fraction, fraction, fraction, 1.0f));

            on_render_scene();
        }

        virtual void on_render_scene()
        {
            render_context ctx = { m_opaque_state, m_premultiplied_alpha_state,
                m_alpha_blend_state, m_cull_back_raster_state,
                m_cull_none_raster_state, m_depth_disable_state,
                m_point_sampler
            };

            auto device_context = m_context.m_immediate_context.get();
            std::for_each(std::begin(m_render_items), std::end(m_render_items), [&ctx, device_context]( const render_item_handle& handle)
            {
                handle->draw(&ctx, device_context);
            });
        }

    protected:

        std::vector< render_item_handle> m_render_items;

    };
}
