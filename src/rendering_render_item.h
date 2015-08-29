#pragma once

#include <memory>

#include <d3d11/d3d11_pointers.h>
#include <util/util_noncopyable.h>


namespace freeform
{
    struct render_context
    {
        d3d11::iblendstate_ptr                  m_opaque_state;
        d3d11::iblendstate_ptr                  m_premultiplied_alpha_state;

        d3d11::iblendstate_ptr                  m_alpha_blend_state;
        d3d11::irasterizerstate_ptr             m_cull_back_raster_state;
        d3d11::irasterizerstate_ptr             m_cull_none_raster_state;

        d3d11::idepthstencilstate_ptr           m_depth_disable_state;
        d3d11::isamplerstate_ptr                m_point_sampler;
    };


    class render_item : private util::noncopyable
    {

        protected:


        virtual ~render_item()
        {

        }

        public:
        render_item()
        {

        }

        render_item(ID3D11Device* d, ID3D11DeviceContext* context)
        {

        }

        void draw(const render_context* ctx, ID3D11DeviceContext* context) const
        {
            on_draw(ctx, context);
        }
        
        private:
        virtual void on_draw(const render_context* ctx, ID3D11DeviceContext* context) const = 0;
    };

    typedef std::shared_ptr< render_item > render_item_handle;

    class render_item_creator : private util::noncopyable
    {
        public:

        virtual ~render_item_creator()
        {

        }

        render_item_handle create(ID3D11Device* device, ID3D11DeviceContext* context) const
        {
            return on_create(device, context);
        }

        private:

        virtual render_item_handle on_create(ID3D11Device* device, ID3D11DeviceContext* context) const = 0;
    };
}


