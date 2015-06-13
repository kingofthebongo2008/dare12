#include "precompiled.h"
#include <cstdint>
#include <memory>

#include "imaging_utils.h"
#include "cuda_imaging.h"

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

    static __global__ void dx(const uint8_t* img_in, uint8_t* img_out, image_kernel_info src, image_kernel_info  dst)
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (is_in_interior(src, x, y))
        {

            const uint8_t* v = sample_2d< uint8_t >( img_in, src, x, y );
            auto  r = *v / 255.0f;


            write_2d<float>(img_out, dst, x, y, r);
        }
    }

    imaging::cuda_texture create_canny_texture(const imaging::cuda_texture& texture_grayscale, float threshold)
    {
        auto width = texture_grayscale.get_width();
        auto height = texture_grayscale.get_height();
        auto t = create_cuda_texture<imaging::image_type::float32>(width, height);

        auto params     = create_texture_kernel_params(width, height);

        //kernel_gray_scale << < std::get<0>(params), std::get<1>(params) >> >  (texture_grayscale.get_gpu_pixels(), t.get_gpu_pixels(), width, height, texture_grayscale.get_pitch(), t.get_pitch());

        cuda::throw_if_failed(cudaDeviceSynchronize());

        return std::move(t);
    }
}


