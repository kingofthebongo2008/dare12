#include "precompiled.h"
#include <cstdint>
#include <memory>
#include <algorithm>

#include "imaging_utils.h"
#include "cuda_imaging.h"


namespace freeform
{

    static inline std::tuple < dim3, dim3 > create_texture_kernel_params( uint32_t width, uint32_t height )
    {
        //1x1 squares
        const dim3 work_items(width, height, 1);
        const dim3 per_block(16, 16, 1);

        const dim3 grid((work_items.x + per_block.x - 1) / per_block.x, (work_items.y + per_block.y - 1) / per_block.y);
        return std::make_tuple(grid, per_block);
    }

    __device__ static inline uint8_t    compute_sobel(
        uint8_t ul, // upper left
        uint8_t um, // upper middle
        uint8_t ur, // upper right
        uint8_t ml, // middle left
        uint8_t mm, // middle (unused)
        uint8_t mr, // middle right
        uint8_t ll, // lower left
        uint8_t lm, // lower middle
        uint8_t lr, // lower right
        float scale)
    {
        int32_t horizontal  = ur + 2 * mr + lr - ul - 2 * ml - ll;
        int32_t vertical    = ul + 2 * um + ur - ll - 2 * lm - lr;

        int32_t sum = static_cast<int16_t> (scale * ( abs(horizontal) + abs(vertical) ) ) ;

        sum = max( sum, 0);
        sum = min( sum, 0xff);

        return static_cast<uint8_t> (sum);
    }


    static __global__ void sobel(const uint8_t* img_in, uint8_t* img_out, cuda::image_kernel_info src, cuda::image_kernel_info  dst)
    {
        using namespace cuda;
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (is_in_interior(src, x, y))
        {

            const uint8_t* pix00 = sample_2d< uint8_t, border_type::clamp >( img_in, src, x - 1, y - 1 );
            const uint8_t* pix01 = sample_2d< uint8_t, border_type::clamp >(img_in, src, x - 0, y - 1);
            const uint8_t* pix02 = sample_2d< uint8_t, border_type::clamp >(img_in, src, x + 1, y - 1);


            const uint8_t* pix10 = sample_2d< uint8_t, border_type::clamp> (img_in, src, x - 1, y);
            const uint8_t* pix11 = sample_2d< uint8_t, border_type::clamp> (img_in, src, x - 0, y);
            const uint8_t* pix12 = sample_2d< uint8_t, border_type::clamp> (img_in, src, x + 1, y);

            const uint8_t* pix20 = sample_2d< uint8_t, border_type::clamp >(img_in, src, x - 1, y + 1);
            const uint8_t* pix21 = sample_2d< uint8_t, border_type::clamp >(img_in, src, x - 0, y + 1);
            const uint8_t* pix22 = sample_2d< uint8_t, border_type::clamp >(img_in, src, x + 1, y + 1);

            auto  u00 = *pix00;
            auto  u01 = *pix01;
            auto  u02 = *pix02;

            auto  u10 = *pix10;
            auto  u11 = *pix11;
            auto  u12 = *pix12;

            auto  u20 = *pix20;
            auto  u21 = *pix21;
            auto  u22 = *pix22;


            auto  r = compute_sobel(
                u00, u01, u02,
                u10, u11, u12,
                u20, u21, u22, 1.0f
                );
            
            write_2d<uint8_t>(img_out, dst, x, y, r );

        }
    }

    imaging::cuda_texture create_canny_texture(const imaging::cuda_texture& texture_grayscale, float threshold)
    {
        using namespace cuda;
        auto width = texture_grayscale.get_width();
        auto height = texture_grayscale.get_height();
        auto t = cuda::create_cuda_texture<imaging::image_type::grayscale>(width, height);

        auto params     = create_texture_kernel_params(width, height);

        sobel << < std::get<0>(params), std::get<1>(params) >> >  (texture_grayscale.get_gpu_pixels(), t.get_gpu_pixels(), create_image_kernel_info(texture_grayscale), create_image_kernel_info(t));

        cuda::throw_if_failed(cudaDeviceSynchronize());

        return std::move(t);
    }
}


