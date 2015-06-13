#include "precompiled.h"
#include <cstdint>
#include <memory>

#include "imaging_utils.h"

namespace cuda
{
    static inline std::tuple < dim3, dim3 > create_texture_kernel_params( uint32_t width, uint32_t height )
    {
        //1x1 squares
        const dim3 work_items(width, height, 1);
        const dim3 per_block(16, 16, 1);

        const dim3 grid((work_items.x + per_block.x - 1) / per_block.x, (work_items.y + per_block.y - 1) / per_block.y);
        return std::make_tuple(grid, per_block);
    }

    imaging::cuda_texture canny_edge_detector ( const imaging::cuda_texture& texture_grayscale , float threshold )
    {
        auto bpp        = 8;
        auto row_pitch  = (bpp * texture_grayscale.get_width() + 7) / 8;
        auto width      = texture_grayscale.get_width();
        auto height     = texture_grayscale.get_height();
        auto size       = row_pitch * height;

        auto memory_buffer = cuda::make_memory_buffer( size ); 
        imaging::cuda_texture t( width, height, bpp, size, row_pitch, imaging::image_type::grayscale, reinterpret_cast<uint8_t*> (memory_buffer->reset()) );

        auto params = create_texture_kernel_params(width, height);

        //kernel_gray_scale << < std::get<0>(params), std::get<1>(params) >> >  (texture_grayscale.get_gpu_pixels(), t.get_gpu_pixels(), width, height, texture_grayscale.get_pitch(), t.get_pitch());

        cuda::throw_if_failed(cudaDeviceSynchronize());

        return std::move(t);
    }
}


