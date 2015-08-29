#pragma once

#include <cstdint>
#include <memory>


namespace imaging
{
    enum image_type : int32_t
    {
        rgb = 0,
        grayscale = 1,
        float32 = 2
    };

    template <image_type> inline uint32_t get_bpp();

    template <> inline uint32_t get_bpp< image_type::rgb> ()
    {
        return 24;
    }

    template <> inline uint32_t get_bpp< image_type::grayscale>()
    {
        return 8;
    }

    template <> inline uint32_t get_bpp< image_type::float32>()
    {
        return 32;
    }

    template <image_type t> inline uint32_t get_pitch(uint32_t width)
    {
        return  (get_bpp<t>() * width + 7) / 8;
    }

    template <image_type t> inline uint32_t get_size(uint32_t width, uint32_t height)
    {
        return  get_pitch<t>(width) * height;
    }

    template <typename pixels_storage>
    class texture : public pixels_storage
    {
    public:
        texture(uint32_t width, uint32_t height, size_t bpp, size_t size, uint32_t pitch, image_type type, uint8_t pixels[]) :
            m_width(width)
            , m_height(height)
            , m_bpp(bpp)
            , m_size(size)
            , m_row_pitch(pitch)
            , m_image_type(type)
            , pixels_storage(pixels, size)
        {
        }

        uint32_t get_width() const
        {
            return m_width;
        }

        uint32_t get_height() const
        {
            return m_height;
        }

        size_t   get_bpp() const
        {
            return m_bpp;
        }

        uint32_t get_pitch() const
        {
            return m_row_pitch;
        }

        image_type get_image_type() const
        {
            return m_image_type;
        }


        size_t get_size() const
        {
            return m_size;
        }

    private:

        image_type  m_image_type;
        size_t      m_bpp;
        uint32_t    m_row_pitch;
        size_t      m_size;

        uint32_t    m_width;
        uint32_t    m_height;
    };
}
