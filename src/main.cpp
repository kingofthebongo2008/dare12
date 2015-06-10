#include "precompiled.h"
#include <cstdint>
#include <memory>

#include <sys/sys_profile_timer.h>
#include <os/windows/com_initializer.h>

#include "imaging_utils_cuda.h"

#include "cuda_helper.h"
#include "imaging.h"
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


int32_t main( int argc, char const* argv[] )
{
    using namespace     os::windows;
    com_initializer     com;

    fs::media_source source(L"../../../media/");

    auto url0 = fs::build_media_url(source, L"essaisynth2.png");
    auto url1 = fs::build_media_url(source, L"essaisynth1.png");

    //read the png texture
    auto texture = imaging::read_texture(url0.get_path());

    imaging::write_texture( texture, url1.get_path() );



    return 0;

}
