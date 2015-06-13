#pragma once

#include "imaging_utils.h"
#include <cstdint>

namespace cuda
{
    class image_kernel_info
    {
        public:

            __device__ __host__ image_kernel_info(size_t bpp, size_t size, uint32_t pitch, uint32_t width, uint32_t height) :
            m_bpp(bpp)
            , m_size(size)
            , m_pitch(pitch)
            , m_width(width)
            , m_height(height)
        {

        }

        __device__ __host__ size_t bpp() const
        {
            return m_bpp;
        }

        __device__ __host__ uint32_t pitch() const
        {
            return m_pitch;
        }


        __device__ __host__ size_t  size() const
        {
            return m_size;
        }

        __device__ __host__ uint32_t  width() const
        {
            return m_width;
        }

        __device__ __host__ uint32_t  height() const
        {
            return m_height;
        }
        
        private:

        size_t      m_bpp;
        size_t      m_size;

        uint32_t    m_pitch;
        uint32_t    m_width;
        uint32_t    m_height;
    };

    template <typename texture > image_kernel_info create_image_kernel_info(const texture& t)
    {
        return image_kernel_info( t.get_bpp(), t.get_size(), t.get_pitch(), t.get_width(), t.get_height() );
    }

    __device__ inline bool is_in_interior(const image_kernel_info& info, uint32_t x, uint32_t y)
    {
        return (x < info.width() && y < info.height() );
    }

    template < typename t > __device__ inline const t* sample_2d(const uint8_t * buffer, const image_kernel_info& info, uint32_t x, uint32_t y)
    {
        return reinterpret_cast<const t*> (buffer + y * info.pitch() + x * sizeof(t));
    }
}

