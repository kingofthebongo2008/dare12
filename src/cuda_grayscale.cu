#include "precompiled.h"
#include <cstdint>
#include <memory>

#include "imaging_utils.h"
#include "cuda_imaging.h"

namespace cuda
{
    struct rgb
    {
        uint8_t b;
        uint8_t g;
        uint8_t r;
    };

    __global__ void kernel_gray_scale(const uint8_t* rgb_t, uint8_t* grayscale, image_kernel_info src, image_kernel_info  dst )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if ( is_in_interior ( src, x, y ) )
        {
            const rgb* rgb_ = sample_2d<rgb>(rgb_t, src, x, y);
            auto  r = rgb_->r / 255.0f;
            auto  g = rgb_->g / 255.0f;
            auto  b = rgb_->b / 255.0f;

            auto  gray = 0.2989f * ( r  * r )  + 0.5870f * ( g *  g ) + 0.1140f * ( b  *  b ) ;
            auto  gray_quantized = static_cast<uint8_t> ( sqrtf( gray ) * 255.0f );
            

            grayscale[y * dst.pitch()  + x] = gray_quantized;
        }
    }

    static inline std::tuple < dim3, dim3 > create_texture_kernel_params( uint32_t width, uint32_t height )
    {
        //1x1 squares
        const dim3 work_items(width, height, 1);
        const dim3 per_block(16, 16, 1);

        const dim3 grid((work_items.x + per_block.x - 1) / per_block.x, (work_items.y + per_block.y - 1) / per_block.y);
        return std::make_tuple(grid, per_block);
    }

    imaging::cuda_texture create_grayscale_texture ( const imaging::cuda_texture& texture_color )
    {
        auto bpp        = 8;
        auto row_pitch  = (bpp * texture_color.get_width() + 7) / 8;
        auto width      = texture_color.get_width();
        auto height     = texture_color.get_height();
        auto size       = row_pitch * height;


        //allocate memory buffer
        auto memory_buffer = cuda::make_memory_buffer( size ); 
        imaging::cuda_texture t( width, height, bpp, size, row_pitch, imaging::image_type::grayscale, reinterpret_cast<uint8_t*> (memory_buffer->reset()) );

        //launch cuda kernel
        auto params = create_texture_kernel_params(width, height);
        kernel_gray_scale << < std::get<0>(params), std::get<1>(params) >> >  (texture_color.get_gpu_pixels(), t.get_gpu_pixels(), create_image_kernel_info(texture_color), create_image_kernel_info( t ) );

        cuda::throw_if_failed(cudaDeviceSynchronize());

        return std::move(t);
    }
}


