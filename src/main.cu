#include "precompiled.h"

#include <chrono>
#include <cstdint>
#include <memory>


#include <thrust/tuple.h>
#include <thrust/device_vector.h>

#include <sys/sys_profile_timer.h>
#include <os/windows/com_initializer.h>

#include "imaging_utils_cuda.h"

#include "cuda_helper.h"
#include "cuda_memory_helper.h"
#include "imaging_utils.h"
#include "freeform_patch.h"


#include <fs/fs_media.h>

static void initialize_cuda()
{
    sys::profile_timer timer;

    //initialize cuda
    cuda::check_device_capabilites(3, 0, [&timer]()->void
    {
        std::cout << "Cuda memory system initialization...." << std::endl;
        std::cout << "Cuda memory system initialized for " << timer.milliseconds() << "ms" << std::endl;
    }
        , []()->void
    {

    });
}

class cuda_initializer
{
    public:
    cuda_initializer()
    {
        initialize_cuda();
    }
};

namespace freeform
{
    imaging::cuda_texture create_grayscale_texture(const imaging::cuda_texture& texture_color);
    imaging::cuda_texture create_canny_texture(const imaging::cuda_texture& texture_color, float threshold);


    thrust::tuple< patches, patches, thrust::device_vector<math::float4> > inititialize_free_form(uint32_t center_image_x, uint32_t center_image_y, float radius, uint32_t patch_count);
    patches test_distances(const patches& p, const patches& p_n);
    patches normal_curve(const patches& n);

    patches displace_points(const patches& m, const patches& nor, const imaging::cuda_texture& grad);
}

static inline float l2_norm(float x, float y)
{
    return sqrtf(x * x + y * y);
}


int32_t main( int argc, char const* argv[] )
{
    using namespace     os::windows;
    com_initializer     com;
    cuda_initializer    cuda;

    auto bpp = imaging::get_bpp<imaging::image_type::rgb>();


    fs::media_source source(L"../../../media/");

    auto url0 = fs::build_media_url(source, L"essaisynth2.png");
    auto url1 = fs::build_media_url(source, L"essaisynth1.png");
    auto url2 = fs::build_media_url(source, L"essaisynth2_grayscale.png");
    auto url3 = fs::build_media_url(source, L"essaisynth2_canny.png");

    //read the png texture
    auto texture = imaging::read_texture(url0.get_path());
    auto pixels  = texture.get_pixels();


    //copy the png texture to the gpu
    auto memory_buffer = cuda::make_memory_buffer( texture.get_size(), pixels.get_pixels_cpu() );
    imaging::cuda_texture t( texture.get_width(), texture.get_height(), texture.get_bpp(), texture.get_size(), texture.get_pitch(), texture.get_image_type(), reinterpret_cast<uint8_t*> (memory_buffer->reset() ) );


    auto gray   = freeform::create_grayscale_texture(t);
    auto canny  = freeform::create_canny_texture(gray, 0.05f);


    imaging::write_texture( texture,    url1.get_path() );
    imaging::write_texture( gray,       url2.get_path() );
    imaging::write_texture( canny,      url3.get_path() );



    //filter out the records that match the composite criteria
    std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();

    auto center_image_x = 240;
    auto center_image_y = 341;
    auto x = 341;
    auto y = 240;
    auto radius = l2_norm(x - static_cast<float> ( center_image_x ) , y - static_cast<float> ( center_image_y ));
    auto patch_count = 10;

    auto init = freeform::inititialize_free_form( center_image_x, center_image_y, radius, patch_count);

    auto m    = freeform::test_distances(thrust::get<0>(init), thrust::get<1>(init) );
    auto nor  = freeform::normal_curve(m);
    auto n    = displace_points( m, nor, canny );


    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "Filtering on device took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms" << std::endl;
   

    return 0;

}
