#pragma once

#include <cstdint>
#include <memory>

#include "imaging.h"
#include "imaging_utils_base.h"
#include "imaging_utils_cpu.h"
#include "imaging_utils_cuda.h"

namespace imaging
{
    typedef texture < cpu_texture_storage >     cpu_texture;
    typedef texture < cuda_texture_storage >    cuda_texture;

    inline cpu_texture read_texture(const wchar_t* url_path)
    {
        auto factory = imaging::create_factory();
        auto stream0 = imaging::create_stream_reading(factory, url_path);
        auto decoder0 = imaging::create_decoder_reading(factory, stream0);
        auto frame0 = imaging::create_decode_frame(decoder0);

        imaging::bitmap_source bitmap(frame0);

        auto format = bitmap.get_pixel_format();
        auto size = bitmap.get_size();

        auto bpp = imaging::wic_bits_per_pixel(factory, GUID_WICPixelFormat24bppBGR);
        auto row_pitch = (bpp * std::get<0>(size) +7) / 8;
        auto row_height = std::get<1>(size);
        auto image_size = row_pitch * row_height;

        std::unique_ptr<uint8_t[]> temp(new (std::nothrow) uint8_t[image_size]);

        bitmap.copy_pixels(nullptr, row_pitch, image_size, temp.get());
        return cpu_texture(std::get<0>(size), std::get<1>(size), bpp, image_size, row_pitch, image_type::rgb, temp.release());
    }

    template <typename texture > inline void write_texture(const texture& t, const wchar_t* url_path)
    {
        using namespace os::windows;

        auto factory = imaging::create_factory();
        auto stream0 = imaging::create_stream_writing(factory, url_path);
        auto encoder0 = imaging::create_encoder_writing(factory, stream0);
        auto frame0 = imaging::create_encode_frame(encoder0);

        throw_if_failed<com_exception>(frame0->SetSize(t.get_width(), t.get_height()));

        WICPixelFormatGUID formatGUID;
        WICPixelFormatGUID formatGUID_required;


        switch (t.get_image_type())
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
        case float32:
        {
            formatGUID = formatGUID_required = GUID_WICPixelFormat32bppGrayFloat;
        }
        break;
        }


        throw_if_failed<com_exception>(frame0->SetPixelFormat(&formatGUID));
        throw_if_failed<com_exception>(IsEqualGUID(formatGUID, formatGUID_required));

        auto proxy = t.get_pixels();


        throw_if_failed<com_exception>(frame0->WritePixels(t.get_height(), t.get_pitch(), t.get_size(), proxy.get_pixels_cpu()));
        throw_if_failed<com_exception>(frame0->Commit());
        throw_if_failed<com_exception>(encoder0->Commit());
    }

}
