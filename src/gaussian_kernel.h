#pragma once

#include <vector>
#include <cmath>

namespace freeform
{
    //gaussian kernels and their derivatives for image processing
    // derivative ( f conv gaussian ) = f conv (derivative gaussian)
    // so we build gaussian kernels here to convolve with the image
    enum gaussian_kernel_type
    {
        x,
        y,
        xx,
        yy,
        xy,
        yx
    };

    struct gaussian_kernel_samples
    {
        std::vector< float > m_samples;

        uint32_t m_width;
        uint32_t m_height;

        uint32_t width() const
        {
            return m_width;
        }

        uint32_t height() const
        {
            return m_height;
        }

        float value( uint32_t x, uint32_t y ) const
        {
            return m_samples[ y * m_height + x];
        }

        inline void set_sample(uint32_t x, uint32_t y,float value )
        {
            auto index = y * m_height + x;
            m_samples[index] = value;
        }
    };


    namespace details
    {
        template < typename gaussian > inline gaussian_kernel_samples create_samples(float sigma, gaussian g)
        {
            gaussian_kernel_samples s;

            auto min = static_cast<int32_t>  (floorf(-3.0f * sigma));
            auto max = static_cast<int32_t>  (ceilf(3.0f * sigma));

            auto width = max - min + 1;
            auto height = width;

            s.m_samples.resize(width * height);
            s.m_width = width;
            s.m_height = height;


            auto px = 0;
            auto py = 0;
            for (auto y = min; y <= max; ++y, py++)
            {
                px = 0;
                for (auto x = min; x <= max; ++x, px++)
                {
                    s.set_sample(px, py, g(x,y, sigma));
                }
            }

            return std::move(s);
        }
    }

    template <gaussian_kernel_type type > inline gaussian_kernel_samples gaussian_kernel(float sigma);

    template <> inline gaussian_kernel_samples gaussian_kernel<gaussian_kernel_type::x>(float sigma)
    {
        return details::create_samples(sigma, [](int32_t x, int32_t y, float sigma) -> float
        {
            const float pi = 3.141592653589793238462643383279502884e+00f;
            return -(x / (2.0 * pi * sigma * sigma * sigma * sigma)) * exp(-(x * x + y * y) / (2.0 * sigma * sigma));
        });
    }

    template <> inline gaussian_kernel_samples gaussian_kernel<gaussian_kernel_type::y>(float sigma)
    {
        return details::create_samples(sigma, [](int32_t x, int32_t y, float sigma) -> float
        {
            const float pi = 3.141592653589793238462643383279502884e+00f;
            return -(y / (2.0 * pi * sigma * sigma * sigma * sigma)) * exp(-(x * x + y * y) / (2.0 * sigma * sigma));
        });
    }

    template <> inline gaussian_kernel_samples gaussian_kernel<gaussian_kernel_type::xx>(float sigma)
    {
        return details::create_samples(sigma, [](int32_t x, int32_t y, float sigma) -> float
        {
            const float pi = 3.141592653589793238462643383279502884e+00f;
            
            double a = 1.0 / (2.0 * pi * sigma * sigma * sigma * sigma);
            double b = ( (x * x) / (sigma * sigma) - 1.0);
            double c = exp(-(x * x + y * y) / (2.0 * sigma * sigma));

            return a * b * c;
        });
    }

    template <> inline gaussian_kernel_samples gaussian_kernel<gaussian_kernel_type::yy>(float sigma)
    {
        return details::create_samples(sigma, [](int32_t x, int32_t y, float sigma) -> float
        {
            const float pi = 3.141592653589793238462643383279502884e+00f;

            double a = 1.0 / (2.0 * pi * sigma * sigma * sigma * sigma);
            double b = ((y * y) / (sigma * sigma) - 1.0);
            double c = exp(-(x * x + y * y) / (2.0 * sigma * sigma));

            return a * b * c;
        });
    }

    template <> inline gaussian_kernel_samples gaussian_kernel<gaussian_kernel_type::xy>(float sigma)
    {
        return details::create_samples(sigma, [](int32_t x, int32_t y, float sigma) -> float
        {
            const float pi = 3.141592653589793238462643383279502884e+00f;

            double a = 1.0 / (2.0 * pi * sigma * sigma * sigma * sigma * sigma * sigma);
            double b = (x * y);
            double c = exp(-(x * x + y * y) / (2.0 * sigma * sigma));

            return a * b * c;
        });
    }

    template <> inline gaussian_kernel_samples gaussian_kernel<gaussian_kernel_type::yx>(float sigma)
    {
        return gaussian_kernel<xy>(sigma);
    }
}

