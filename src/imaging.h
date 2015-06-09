#pragma once

#include <tuple>

#include <os/windows/com_error.h>
#include <os/windows/com_pointers.h>

#include <wincodec.h>

namespace imaging
{
    typedef os::windows::com_ptr<IWICImagingFactory>     wic_imaging_factory;
    typedef os::windows::com_ptr<IWICStream>             wic_stream;
    typedef os::windows::com_ptr<IWICBitmapDecoder>      wic_decoder;
    typedef os::windows::com_ptr<IWICBitmapFrameDecode>  wic_frame_decode;
    typedef os::windows::com_ptr<IWICBitmapSource>       wic_bitmap_source;
    typedef os::windows::com_ptr<IWICComponentInfo>      wic_component_info;
    typedef os::windows::com_ptr<IWICPixelFormatInfo>    wic_pixel_format_info;


    wic_imaging_factory create_factory()
    {
        using namespace os::windows;
        wic_imaging_factory factory;
        throw_if_failed<com_exception>(CoCreateInstance(CLSID_WICImagingFactory2, nullptr, CLSCTX_INPROC_SERVER, __uuidof(IWICImagingFactory2), (LPVOID*)&factory));
        return std::move(factory);
    }

    wic_stream create_stream(wic_imaging_factory f, const wchar_t* path)
    {
        using namespace os::windows;

        wic_stream stream;

        throw_if_failed<com_exception>(f->CreateStream(&stream));
        throw_if_failed<com_exception>(stream->InitializeFromFilename(path, GENERIC_READ));

        return std::move(stream);
    }

    wic_decoder create_decoder_reading(wic_imaging_factory f, wic_stream s)
    {
        using namespace os::windows;

        wic_decoder decoder;
        throw_if_failed<com_exception>(f->CreateDecoderFromStream(s.get(), 0, WICDecodeMetadataCacheOnDemand, &decoder));

        return std::move(decoder);
    }

    wic_frame_decode create_decode_frame(wic_decoder decoder)
    {
        using namespace os::windows;
        wic_frame_decode frame;

        throw_if_failed<com_exception>(decoder->GetFrame(0, &frame));

        return frame;
    }


    wic_component_info create_component_info(wic_imaging_factory factory, REFGUID targetGuid)
    {
        using namespace os::windows;
        wic_component_info r;

        throw_if_failed<com_exception>(factory->CreateComponentInfo(targetGuid, &r));

        return r;
    }

    static size_t wic_bits_per_pixel(wic_imaging_factory factory, REFGUID targetGuid)
    {
        auto info = create_component_info(factory, targetGuid);
        using namespace os::windows;

        WICComponentType type;

        throw_if_failed<com_exception>(info->GetComponentType(&type));

        if (type != WICPixelFormat)
            return 0;

        wic_pixel_format_info pixel_info;

        throw_if_failed<com_exception>(info->QueryInterface(__uuidof(IWICPixelFormatInfo), (void**)&pixel_info));


        uint32_t bpp;
        throw_if_failed<com_exception>(pixel_info->GetBitsPerPixel(&bpp));
        return bpp;
    }


    class bitmap_source
    {
    public:
        explicit bitmap_source(wic_bitmap_source  source) : m_source(source)
        {

        }


        std::tuple < uint32_t, uint32_t> get_size() const
        {
            using namespace os::windows;
            uint32_t width;
            uint32_t height;
            throw_if_failed<com_exception>(m_source->GetSize(&width, &height));
            return std::move(std::make_tuple(width, height));
        }

        WICPixelFormatGUID get_pixel_format() const
        {
            using namespace os::windows;

            WICPixelFormatGUID r;

            throw_if_failed<com_exception>(m_source->GetPixelFormat(&r));
            return std::move(r);
        }

        void copy_pixels(const WICRect *prc, uint32_t stride, uint32_t buffer_size, uint8_t* buffer)
        {
            using namespace os::windows;
            throw_if_failed<com_exception>(m_source->CopyPixels(prc, stride, buffer_size, buffer));
        }

    private:

        wic_bitmap_source m_source;
    };
}
