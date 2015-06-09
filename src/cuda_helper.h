#pragma once

#include <cstdint>
#include <iostream>
#include <exception>

#include <cuda_runtime.h>

namespace cuda
{
    class exception : public std::exception
    {
    public:

        exception(cudaError_t error) : m_error(error)
        {

        }

        const char * what() const override
        {
            return cudaGetErrorString(m_error);
        }

    private:

        cudaError_t m_error;
    };

    template < typename exception > inline void throw_if_failed(cudaError_t error)
    {
        if (error != cudaSuccess)
        {
            throw exception(error);
        }
    }

    inline void throw_if_failed(cudaError_t error)
    {
        throw_if_failed<exception>(error);
    }

    // General check for CUDA GPU SM Capabilities
    template <typename f, typename g> inline void check_device_capabilites(int major_version, int minor_version, f success, g fail)
    {
        cudaDeviceProp deviceProp = {};
        std::int32_t dev;

        throw_if_failed(cudaGetDevice(&dev));
        throw_if_failed(cudaGetDeviceProperties(&deviceProp, dev));

        if (    (deviceProp.major > major_version) ||
                (deviceProp.major == major_version && deviceProp.minor >= minor_version)    )
        {
            std::cout << "Device " << dev << ":" << deviceProp.name << " Compute SM " << deviceProp.major << "." << deviceProp.minor << std::endl;
            success();
        }
        else
        {
            fail();
        }
    }
}

