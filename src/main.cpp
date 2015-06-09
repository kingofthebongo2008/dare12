#include "precompiled.h"
#include <cstdint>

#include "cuda_helper.h"


static void initialize_cuda()
{
    //initialize cuda
    cuda::check_device_capabilites(3, 0, []()->void
    {
        std::cout << "Cuda memory system initialization...." << std::endl;
    }
        , []()->void
    {
      });
}

extern void test_cuda4();

int32_t main( int argc, char const* argv[] )
{
    initialize_cuda();
    test_cuda4();
    return 0;
}
