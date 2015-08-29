#pragma once

#include "imaging_utils.h"

#include "cuda_memory_helper.h"

namespace cuda
{
    template <typename imaging::image_type t > inline imaging::cuda_texture create_texture(uint32_t width, uint32_t height)
    {
        auto bpp = imaging::get_bpp<t>();

        auto memory_buffer = cuda::make_memory_buffer( imaging::get_pitch<t>(width) * height );
        return std::move(imaging::cuda_texture(width, height, imaging::get_bpp<t>(), imaging::get_size<t>(width, height), imaging::get_pitch<t>(width), t, reinterpret_cast<uint8_t*> (memory_buffer->reset())));
    }

}


