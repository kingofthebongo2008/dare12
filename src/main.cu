#include "precompiled.h"


#include <chrono>
#include <cstdint>
#include <memory>
#include <tuple>
#include <algorithm>


#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <sys/sys_profile_timer.h>
#include <os/windows/com_initializer.h>

#include "imaging_utils_cuda.h"

#include "cuda_helper.h"
#include "cuda_memory_helper.h"
#include "imaging_utils.h"
#include "freeform_patch.h"
#include "graphic_types.h"

#include <fs/fs_media.h>

#include "gaussian_kernel.h"

#include "cuda_texture_utils.h"



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
        void* a;
        cudaMalloc(&a, 20);

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


    std::tuple< samples, patches  > inititialize_free_form(float center_image_x, float center_image_y, float radius, uint32_t patch_count);
    patches split(const patches& p, float pixel_size);
    patches flip( patches& p);
    void    deform(const patches& p, const imaging::cuda_texture& grad, patches& deformed, thrust::device_vector<uint32_t>& stop);
    bool    converged(thrust::device_vector<uint32_t>& stop);

    samples sample_patches(const patches& p);

    void display( const imaging::cuda_texture& t, const patches& p );
    void display(const imaging::cuda_texture& t,  const samples& p );
    void display( const imaging::cuda_texture& t);
}

inline std::ostream& operator<<(std::ostream& s, const float4& p)
{
    s << "x: " << p.x << " " << p.y << " " << p.z << " " << p.w << std::endl;
    return s;
}

int32_t main( int argc, char const* argv[] )
{
    using namespace     os::windows;
    com_initializer     com;
    cuda_initializer    cuda;

    auto bpp = imaging::get_bpp<imaging::image_type::rgb>();


    fs::media_source source(L"../../../media/");
    
    /*
    auto url0 = fs::build_media_url(source, L"essaisynth2.png");
    auto url1 = fs::build_media_url(source, L"essaisynth1.png");
    auto url2 = fs::build_media_url(source, L"essaisynth2_grayscale.png");
    auto url3 = fs::build_media_url(source, L"essaisynth2_canny.png");
    */
    
    /*
    auto url0 = fs::build_media_url(source, L"basic2_obstacles.png");
    auto url1 = fs::build_media_url(source, L"basic1_obstacles.png");
    auto url2 = fs::build_media_url(source, L"basic2_obstacles_grayscale.png");
    auto url3 = fs::build_media_url(source, L"basic2_obstacles_canny.png");
    */

    auto url0 = fs::build_media_url(source, L"circle.png");
    auto url1 = fs::build_media_url(source, L"circle1_obstacles.png");
    auto url2 = fs::build_media_url(source, L"circle_grayscale.png");
    auto url3 = fs::build_media_url(source, L"circle_canny.png");


    //read the png texture
    auto texture = imaging::read_texture(url0.get_path());
    auto pixels  = texture.get_pixels();

    //copy the png texture to the gpu
    auto memory_buffer = cuda::make_memory_buffer( texture.get_size(), pixels.get_pixels_cpu() );
    imaging::cuda_texture t( texture.get_width(), texture.get_height(), texture.get_bpp(), texture.get_size(), texture.get_pitch(), texture.get_image_type(), reinterpret_cast<uint8_t*> (memory_buffer->reset() ) );


    //do gray scale conversion and edge detection
    auto gray   = freeform::create_grayscale_texture(t);
    auto canny  = freeform::create_canny_texture(gray, 0.05f);

    imaging::write_texture( texture,    url1.get_path() );
    imaging::write_texture( gray,       url2.get_path() );
    imaging::write_texture( canny,      url3.get_path() );

    //filter out the records that match the composite criteria
    std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();

    auto center_image_x = 0.5f;// 341.0f / gray.get_width();
    auto center_image_y = 0.5f;// 240.0f / gray.get_height();

    auto pixel_size = std::max(1.0f / gray.get_width(), 1.0f / gray.get_height());
    auto radius = 20.0f * pixel_size;
    auto patch_count = 20;

    auto init = freeform::inititialize_free_form( center_image_x, center_image_y, radius, patch_count);

    auto tex = cuda::create_texture<imaging::image_type::rgb>(320, 240);

    freeform::display(t);

    //deform the patches along the normal

    auto deformed                  = std::get<1>(init);
    freeform::patches              old;
    thrust::device_vector<uint32_t> stop;
    bool stop_iterations = false;

    while (!stop_iterations)
    {
        old = split(deformed, pixel_size);
        old = flip(old);
        
        freeform::deform(old, canny, deformed, stop);
        //freeform::display(gray, deformed);
        stop_iterations = freeform::converged(stop);
    }

    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "Filtering on device took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms" << std::endl;

    //display the results
    //freeform::display(gray, std::get<1>(init));
    //freeform::display(gray, freeform::sample_patches(deformed));
    //freeform::display(gray, deformed);

    return 0;

}


