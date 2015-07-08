#pragma once

#include <vector>

namespace freeform
{
    namespace graphic
    {
        struct point
        {
            float m_x;
            float m_y;
        };

        struct patch
        {
            point m_points[4];
        };

        struct patch_draw_info
        {
            std::vector< patch > m_patches;

            const patch* get_patches() const
            {
                return &m_patches[0];
            }

            size_t get_count() const
            {
                return m_patches.size();
            }
        };

        inline size_t get_patch_size(const patch_draw_info&  gdi)
        {
            return gdi.get_count() * sizeof(patch);
        }

        struct transform_info
        {
            float m_center_x;
            float m_center_y;

            float m_image_width;
            float m_image_height;
        };
    }
}
