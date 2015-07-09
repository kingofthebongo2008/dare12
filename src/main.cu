#include "precompiled.h"

#include <chrono>
#include <cstdint>
#include <memory>


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


    thrust::tuple< patches, patches, tabs > inititialize_free_form(uint32_t center_image_x, uint32_t center_image_y, float radius, uint32_t patch_count);

    patches test_distances(const patches& p, const patches& p_n);
    patches normal_curve(const patches& n);

    thrust::tuple<points, thrust::device_vector<uint8_t> >      displace_points(const patches& m, const patches& nor, const imaging::cuda_texture& grad);
    thrust::tuple<patches, tabs >                               polygon_computation(points& n);
    
    patches flip( patches& p, tabs& t);

    void    display(const imaging::cuda_texture& t, const thrust::host_vector< patch>& p, const graphic::transform_info& transform);
}

static inline float l2_norm(float x, float y)
{
    return sqrtf(x * x + y * y);
}

struct lexicographical_sorter
{
    __device__ bool operator()(freeform::tab a0, freeform::tab b0) const
    {
        math::float4 a = a0.m_aabb;
        math::float4 b = b0.m_aabb;

        return a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y = b.y && (a.z < b.z || (a.z == b.z && a.w < b.w)))));
    }
};


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


    //do gray scale conversion and edge detection
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
    auto n_s  = displace_points( m, nor, canny );

    auto n    = polygon_computation(thrust::get<0>(n_s));
    thrust::sort(thrust::get<1>(n).begin(), thrust::get<1>(n).end(), lexicographical_sorter());

    
    auto flipped = flip(thrust::get<0>(n), thrust::get<1>(n));



    //output results
    thrust::host_vector<freeform::tab > test;
    test.resize(thrust::get<1>(n).size());

    thrust::copy(thrust::get<1>(n).begin(), thrust::get<1>(n).end(), test.begin() );

    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "Filtering on device took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms" << std::endl;

    thrust::copy(test.begin(), test.end(), std::ostream_iterator< freeform::tab >(std::cout, " "));



    //copy for rendering
    thrust::host_vector< freeform::patch > display_patches;

    //display_patches.resize(flipped.size());
    //thrust::copy(flipped.begin(), flipped.end(), display_patches.begin() );

    display_patches.resize(thrust::get<1>(init).size() );
    thrust::copy(thrust::get<1>(init).begin(), thrust::get<1>(init).end(), display_patches.begin() );
    
    /*
    
    display_patches.resize(2);
    freeform::patch p0;

    //draw the control points
    float constrol_points[] =
    {
        -1.0f, -1.0f,
        4.0f, -1.0f,
        -4.0f, 1.0f,
        1.0f, 1.0f,

        -2.0f, -0.8f,
        4.0f, -1.0f,
        -4.0f, 1.0f,
        2.0f, 0.8f
    };

    auto w = gray.get_width();
    auto h = gray.get_height();
    p0.x0 = 0.0;
    p0.x1 = 0.0f;// w / 2;
    p0.x2 = w;// w / 2;
    p0.x3 = w;

    p0.y0 = 0;
    p0.y1 = h;// / 2;
    p0.y2 = h;// / 2;
    p0.y3 = 0;

    display_patches[0] = p0;
    */
    /*
    p0.x0 = 0;
    p0.x1 = 4.0f * w;
    p0.x2 = -4.0f * w;
    p0.x3 = 2.0f * w;

    p0.y0 = -1.0f * h;
    p0.y1 = -1.0f * h;
    p0.y2 = 1.0f * h;
    p0.y3 = 1.0f * h;
    
    display_patches[1] = p0;
    */
    

    freeform::graphic::transform_info transform;

    transform.m_center_x     = gray.get_width();
    transform.m_center_y     = gray.get_height();
    transform.m_image_height = gray.get_height();
    transform.m_image_width  = gray.get_width();

    

    freeform::display(gray, display_patches, transform );

    return 0;

}
