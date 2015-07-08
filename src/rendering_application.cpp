#include "precompiled.h"
#include "rendering_application.h"

#include "imaging_utils.h"

#include "freeform_patch.h"
#include "graphic_types.h"


namespace freeform
{
    graphic::patch_draw_info create_draw_info(const thrust::host_vector< patch > & p)
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

    void    display(const imaging::cuda_texture& t, const thrust::host_vector< patch>& p, const graphic::transform_info& transform)
    {
        std::unique_ptr< sample_application >  app(new sample_application(L"Sample Application", t, create_draw_info(p), transform));



        auto result = app->run();
    }
}

