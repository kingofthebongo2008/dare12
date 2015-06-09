#pragma once

#include <cstdint>
#include <memory>

#include "imaging.h"

namespace imaging
{
    enum image_type : int32_t
    {
        rgb = 0,
        grayscale = 1
    };

    class texture
    {
        public:
        texture(uint32_t width, uint32_t height, size_t bpp, uint32_t pitch, image_type type, uint8_t pixels[]) :
            m_width(width)
            , m_height(height)
            , m_bpp(bpp)
            , m_row_pitch(pitch)
            , m_image_type(type)
            , m_pixels(pixels, std::default_delete< uint8_t[] >() )
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

        void*  get_pixels() const
        {
            return m_pixels.get();
        }

    private:

        image_type  m_image_type;
        size_t      m_bpp;
        uint32_t    m_row_pitch;

        uint32_t    m_width;
        uint32_t    m_height;

        std::shared_ptr< uint8_t > m_pixels;
    };


    inline texture read_texture(const wchar_t* url_path)
    {
        auto factory = imaging::create_factory();
        auto stream0 = imaging::create_stream(factory, url_path );
        auto decoder0 = imaging::create_decoder_reading(factory, stream0);
        auto frame0 = imaging::create_decode_frame(decoder0);

        imaging::bitmap_source bitmap(frame0);

        auto format = bitmap.get_pixel_format();
        auto size = bitmap.get_size();

        auto bpp = imaging::wic_bits_per_pixel(factory, GUID_WICPixelFormat24bppRGB);
        auto row_pitch = (bpp * std::get<0>(size) +7) / 8;
        auto row_height = std::get<1>(size);
        auto image_size = row_pitch * row_height;

        std::unique_ptr<uint8_t[]> temp(new (std::nothrow) uint8_t[image_size]);

        bitmap.copy_pixels(nullptr, row_pitch, image_size, temp.get());


        return texture(std::get<0>(size), std::get<1>(size), bpp, row_pitch, image_type::rgb, temp.release() );
    }
}
