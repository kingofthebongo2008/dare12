#include "precompiled.h"
#include <cstdint>
#include <memory>

#include <sys/sys_profile_timer.h>
#include <os/windows/com_initializer.h>

#include "imaging_utils_cuda.h"

#include "cuda_helper.h"
#include "cuda_memory_helper.h"
#include "imaging_utils.h"


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

namespace cuda
{
    imaging::cuda_texture create_grayscale_texture(const imaging::cuda_texture& texture_color);
    imaging::cuda_texture create_canny_texture(const imaging::cuda_texture& texture_color, float threshold);
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
    auto url2 = fs::build_media_url(source, L"essaisynth0.png");

    //read the png texture
    auto texture = imaging::read_texture(url0.get_path());
    auto pixels  = texture.get_pixels();


    //copy the png texture to the gpu
    auto memory_buffer = cuda::make_memory_buffer( texture.get_size(), pixels.get_pixels_cpu() );
    imaging::cuda_texture t( texture.get_width(), texture.get_height(), texture.get_bpp(), texture.get_size(), texture.get_pitch(), texture.get_image_type(), reinterpret_cast<uint8_t*> (memory_buffer->reset() ) );


    auto gray   = cuda::create_grayscale_texture(t);

    auto canny  = cuda::create_canny_texture(t, 0.05f);


    imaging::write_texture( texture, url1.get_path() );
    imaging::write_texture( gray, url2.get_path());


   

    return 0;

}
