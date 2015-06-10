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

    class cpu_texture_storage
    {
        public:
        cpu_texture_storage(uint8_t pixels[]) :
        m_pixels(pixels, std::default_delete< uint8_t[] >())
        {

        }

        uint8_t*  get_pixels_cpu() const
        {
            return m_pixels.get();
        }

        private:

        std::shared_ptr< uint8_t > m_pixels;
    };

    template <typename pixels_storage>
    class texture : public pixels_storage
    {
        public:
        texture( uint32_t width, uint32_t height, size_t bpp, size_t size, uint32_t pitch, image_type type, uint8_t pixels[] ) :
            m_width(width)
            , m_height(height)
            , m_bpp(bpp)
            , m_size(size)
            , m_row_pitch(pitch)
            , m_image_type(type)
            , pixels_storage( pixels )
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

    typedef texture < cpu_texture_storage > cpu_texture;


    inline cpu_texture read_texture(const wchar_t* url_path)
    {
        auto factory    = imaging::create_factory();
        auto stream0    = imaging::create_stream_reading(factory, url_path );
        auto decoder0   = imaging::create_decoder_reading(factory, stream0);
        auto frame0     = imaging::create_decode_frame(decoder0);

        imaging::bitmap_source bitmap(frame0);

        auto format     = bitmap.get_pixel_format();
        auto size       = bitmap.get_size();

        auto bpp        = imaging::wic_bits_per_pixel(factory, GUID_WICPixelFormat24bppBGR);
        auto row_pitch  = (bpp * std::get<0>(size) +7) / 8;
        auto row_height = std::get<1>(size);
        auto image_size = row_pitch * row_height;

        std::unique_ptr<uint8_t[]> temp(new (std::nothrow) uint8_t[image_size]);

        bitmap.copy_pixels(nullptr, row_pitch, image_size, temp.get());
        return cpu_texture(std::get<0>(size), std::get<1>(size), bpp, image_size, row_pitch, image_type::rgb, temp.release());
    }

    inline void write_texture(const cpu_texture& t, const wchar_t* url_path)
    {
        using namespace os::windows;

        auto factory    = imaging::create_factory();
        auto stream0    = imaging::create_stream_writing(factory, url_path);
        auto encoder0   = imaging::create_encoder_writing(factory, stream0);
        auto frame0     = imaging::create_encode_frame(encoder0);

        throw_if_failed<com_exception>(frame0->SetSize(t.get_width(), t.get_height()));
        
        WICPixelFormatGUID formatGUID;
        WICPixelFormatGUID formatGUID_required;


        switch(t.get_image_type())
        {
            case rgb:
            {
                formatGUID = formatGUID_required = GUID_WICPixelFormat24bppBGR;
            }
            break;

            case grayscale:
            {
                formatGUID = formatGUID_required = GUID_WICPixelFormat8bppGray;
            }
            break;
        }
            

        throw_if_failed<com_exception>(frame0->SetPixelFormat(&formatGUID));
        throw_if_failed<com_exception>(IsEqualGUID(formatGUID, formatGUID_required));


        throw_if_failed<com_exception>( frame0->WritePixels( t.get_height(), t.get_pitch(), t.get_size(), t.get_pixels_cpu() ) );
        throw_if_failed<com_exception>( frame0->Commit() );
        throw_if_failed<com_exception>( encoder0->Commit() );
    }
}
