#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include "cuda_helper.h"

namespace imaging
{
    struct cuda_default_delete
    {
        cuda_default_delete() throw()
        {

        }

        void operator()( uint8_t* pointer ) const throw()
        {	
            ::cudaFree( pointer );
        }
    };

    struct cuda_free_array
    {
        cuda_free_array() throw()
        {

        }

        void operator()(cudaArray_t pointer) const throw()
        {
            ::cudaFreeArray(pointer);
        }
    };

    class cuda_texture_storage
    {
        public:

        class storage_proxy
        {
            public:
            storage_proxy( std::shared_ptr< uint8_t > pixels ) : m_pixels(pixels)
            {

            }

            uint8_t* get_pixels_cpu() const
            {
                return m_pixels.get();
            }

            private:

            std::shared_ptr< uint8_t > m_pixels;
        };

        cuda_texture_storage( uint8_t pixels[], size_t size ) :
        m_pixels(pixels, cuda_default_delete() )
        , m_size(size)
        {

        }

        storage_proxy  get_pixels( ) const
        {
            std::unique_ptr<uint8_t[]> pixels(new uint8_t[m_size]);

            cuda::throw_if_failed(cudaMemcpy(pixels.get(), m_pixels.get(), m_size, cudaMemcpyDeviceToHost));

            return storage_proxy(std::shared_ptr<uint8_t>( pixels.release() , std::default_delete< uint8_t[] >()));
        }

        private:

        std::shared_ptr< uint8_t > m_pixels;    //points to device memory
        size_t                     m_size;
    };

}
