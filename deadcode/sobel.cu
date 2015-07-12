    __device__ inline void voisinage(float x, float y, int32_t v1[], int32_t v2[])
    {
        int32_t x1 = static_cast<int32_t> (floorf(x));  //todo cast to int
        int32_t y1 = static_cast<int32_t> (floorf(y));  //todo cast to int

        //% Returns the "pixelique" coordinates of the point neighborhood( its size is 9 * 9 )? 8x8?

        const int32_t indices_v1[8] = { -1, -1, -1, 0, 0, +1, +1, +1 };
        const int32_t indices_v2[8] = { -1, 0, 1, -1, 1, -1, 0, +1 };

        for (uint32_t i = 0; i < 8; ++i)
        {
            v1[i] = x1 + indices_v1[i];
        }

        for (uint32_t i = 0; i < 8; ++i)
        {
            v2[i] = y1 + indices_v2[i];
        }
    }

struct deform_points_kernel
    {
        const   cuda::image_kernel_info m_sampler;
        const   uint8_t*                m_grad;

        deform_points_kernel(   const cuda::image_kernel_info& sampler, const uint8_t* grad ) : m_sampler(sampler), m_grad(grad)
        {

        }

        __device__ static inline float    compute_sobel(
            float ul, // upper left
            float um, // upper middle
            float ur, // upper right
            float ml, // middle left
            float mm, // middle (unused)
            float mr, // middle right
            float ll, // lower left
            float lm, // lower middle
            float lr, // lower right
            float& dx,
            float& dy
            )
        {
            dx = ur + 2 * mr + lr - ul - 2 * ml - ll;
            dy   = ul + 2 * um + ur - ll - 2 * lm - lr;

            float  sum = static_cast<float> (abs(dx) + abs(dy));
            return sum;
        }

        __device__ thrust::tuple<point, uint8_t> operator() (const thrust::tuple < point, point> p)
        {
            using namespace cuda;

            auto pt = thrust::get<0>(p);
            auto normal = thrust::get<1>(p);

            auto x = pt.x;
            auto y = pt.y;

            const uint8_t* pix00 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 1, y - 1);
            const uint8_t* pix01 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 0, y - 1);
            const uint8_t* pix02 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x + 1, y - 1);


            const uint8_t* pix10 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x - 1, y);
            const uint8_t* pix11 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x - 0, y);
            const uint8_t* pix12 = sample_2d< uint8_t, border_type::clamp>(m_grad, m_sampler, x + 1, y);

            const uint8_t* pix20 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 1, y + 1);
            const uint8_t* pix21 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x - 0, y + 1);
            const uint8_t* pix22 = sample_2d< uint8_t, border_type::clamp >(m_grad, m_sampler, x + 1, y + 1);

            float c   = 1.0f / 255.0f;

            auto  u00 = *pix00 * c;
            auto  u01 = *pix01 * c;
            auto  u02 = *pix02 * c;

            auto  u10 = *pix10 * c;
            auto  u11 = *pix11 * c;
            auto  u12 = *pix12 * c;

            auto  u20 = *pix20 * c;
            auto  u21 = *pix21 * c;
            auto  u22 = *pix22 * c;

            float dx = 0.0f;
            float dy = 0.0f;

            auto  r = compute_sobel(
                u00, u01, u02,
                u10, u11, u12,
                u20, u21, u22, dx, dy
                );

            //normalize the gradient
            float g = 1.0f / (r + 0.0001f);
            dx = dx * g;
            dy = dy * g;

            //dx = 22.0f; //test to see if the gradient works

            
            float pixel_size = max( 1.0f / m_sampler.width(), 1.0f / m_sampler.height() );
            float scale      = 1.0f;// pixel_size * 5.0;

            point k1         = make_point(scale, scale);
            point k          = make_point(-scale * 1.1f, -scale * 1.1f);

            point grad       = make_point(dx, dy);
            point d0         = mad(k1, normal, pt);
            point d1         = mad(k, grad, d0);

            return thrust::make_tuple( d1, 1 );
        }
    };
    


