#pragma once

#include <cstdint>
#include <memory>

namespace imaging
{
    class cpu_texture_storage
    {

    public:

        class storage_proxy
        {

        public:

            storage_proxy(uint8_t* pixels) : m_pixels(pixels)
            {

            }

            uint8_t* get_pixels_cpu() const
            {
                return m_pixels;
            }

        private:

            uint8_t* m_pixels;
        };


        cpu_texture_storage(uint8_t pixels[], size_t size) :
            m_pixels(pixels, std::default_delete< uint8_t[] >())
        {

        }

        storage_proxy  get_pixels() const
        {
            return storage_proxy(m_pixels.get());
        }

    private:

        std::shared_ptr< uint8_t > m_pixels;
    };
}
