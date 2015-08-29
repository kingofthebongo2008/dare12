#include "precompiled.h"
#include "rendering_render_item_patch.h"

#include "rendering_application_sample.h"

#include "imaging_utils.h"

#include "freeform_patch.h"
#include "graphic_types.h"

#include "rendering_render_item_texture.h"
#include "rendering_render_item_sample.h"
#include "rendering_render_item_patch.h"


namespace freeform
{
    template <typename t> graphic::patch_draw_info create_draw_info(const thrust::host_vector< t > & p)
    {
        graphic::patch_draw_info di;
        di.m_patches.resize( p.size() );
        auto i = 0;

        for (auto& it : p)
        {
            graphic::patch pt;

            pt.m_points[0].m_x = it.x0;
            pt.m_points[1].m_x = it.x1;
            pt.m_points[2].m_x = it.x2;
            pt.m_points[3].m_x = it.x3;

            pt.m_points[0].m_y = it.y0;
            pt.m_points[1].m_y = it.y1;
            pt.m_points[2].m_y = it.y2;
            pt.m_points[3].m_y = it.y3;

            di.m_patches[i++] = pt;
        }

        return std::move(di);
    }

    void display(const imaging::cuda_texture& t, const thrust::host_vector<patch>& p, const graphic::transform_info& transform)
    {
        std::vector < std::shared_ptr<render_item_creator> > items;

        items.push_back(std::make_shared< render_item_creator_texture >(t));
        items.push_back(std::make_shared< render_item_creator_patch >(create_draw_info<freeform::patch>(p), transform));

        std::unique_ptr< rendering_application >  app(new sample_application(L"Patches", items));
        app->run();
    }

    void display(const imaging::cuda_texture& t, const thrust::host_vector<sample>& p, const graphic::transform_info& transform)
    {
        std::vector < std::shared_ptr<render_item_creator> > items;

        items.push_back(std::make_shared< render_item_creator_texture >(t));
        items.push_back(std::make_shared< render_item_creator_sample >(create_draw_info<freeform::sample>(p), transform));

        std::unique_ptr< rendering_application >  app(new sample_application(L"Samples", items));
        app->run();

    }

    void display(const imaging::cuda_texture& t)
    {
        std::vector < std::shared_ptr<render_item_creator> > items;

        items.push_back(std::make_shared< render_item_creator_texture >(t) );

        std::unique_ptr< rendering_application >  app(new sample_application(L"Samples", items ));
        app->run();
    }


    void    display(const imaging::cuda_texture& t, const patches& p)
    {
        //copy patches to the host and transfer to directx back on the gpu for displaying
        //this goes through the pci bus, so is slow, but is for display purposes only
        thrust::host_vector< freeform::patch > display_patches;

        display_patches.resize(p.size());
        thrust::copy(p.begin(), p.end(), display_patches.begin());

        //make a transform to rescale to perspective space
        freeform::graphic::transform_info transform;
        transform.m_center_x = static_cast<float>(t.get_width());
        transform.m_center_y = static_cast<float>(t.get_height());
        transform.m_image_height = static_cast<float>(t.get_height());
        transform.m_image_width = static_cast<float>(t.get_width());

        display(t, display_patches, transform);
    }

    void    display(const imaging::cuda_texture& t, const samples& p)
    {
        //copy samples to the host and transfer to directx back on the gpu for displaying
        //this goes through the pci bus, so is slow, but is for display purposes only
        thrust::host_vector< freeform::sample > display_samples;

        display_samples.resize(p.size());
        thrust::copy(p.begin(), p.end(), display_samples.begin());

        //make a transform to rescale to perspective space
        freeform::graphic::transform_info transform;
        transform.m_center_x = static_cast<float>(t.get_width());
        transform.m_center_y = static_cast<float>(t.get_height());
        transform.m_image_height = static_cast<float>(t.get_height());
        transform.m_image_width = static_cast<float>(t.get_width());

        display(t, display_samples, transform);
    }
}

