#pragma once

#include <cstdint>

#include <device_functions.h>
#include <vector_functions.h>

namespace math
{
    typedef ::float4 float4;

    enum component : int32_t
    {
        x = 0,
        y = 1,
        z = 2,
        w = 3
    };

    //memory control and initialization

    __device__ inline float4 zero()
    {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __device__ inline float4 one()
    {
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    __device__ inline float4 minus_one()
    {
        return make_float4(-1.0f, -1.0f, -1.0f, -1.0f);
    }

    __device__ inline float4 identity_r0()
    {
        return make_float4( 1.0f, 0.0f, 0.0f, 0.0f);
    }

    __device__ inline float4 identity_r1()
    {
        return make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    }

    __device__ inline float4 identity_r2()
    {
        return make_float4(0.0f, 0.0f, 1.0f, 0.0f);
    }

    __device__ inline float4 identity_r3()
    {
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    namespace details
    {
        template <uint32_t c > __device__ inline float get_component(float4 v)
        {
            float a;
            switch (c)
            {
                case component::x:
                    a = v.x;
                    break;
                case component::y:
                    a = v.y;
                case component::z:
                    a = v.z;
                case component::w:
                    a = v.w;
            }
            return a;
        }
    }

    template <uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4> __device__ inline float4 shuffle(float4 value1, float4 value2)
    {
        float a;
        float b;

        float c;
        float d;

        a = details::get_component< v1>(value1);
        b = details::get_component< v2>(value1);
        c = details::get_component< v3>(value2);
        d = details::get_component< v4>(value2);
        return make_float4(a, b, c, d);
    }

    template <uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4> __device__ inline float4 swizzle(float4 value)
    {
        return shuffle<v1, v2, v3, v4>(value, value);
    }

    __device__ inline float4 merge_xy(float4 v1, float4 v2)
    {
        return make_float4(v1.x, v2.x, v1.y, v2.y);
    }

    __device__ inline float4 merge_zw(float4 v1, float4 v2)
    {
        return make_float4(v1.z, v2.z, v1.w, v2.w);
    }
    
    __device__ inline float4 splat(float value)
    {
        return make_float4(value, value, value, value);
    }
            
    __device__ inline float4 splat_x(float4 value)
    {
        return swizzle<x,x,x,x>(value);
    }

    __device__ inline float4 splat_y(float4 value)
    {
        return swizzle<y,y,y,y>(value);
    }

    __device__ inline float4 splat_z(float4 value)
    {
        return swizzle<z,z,z,z>(value);
    }

    __device__ inline float4 splat_w(float4 value)
    {
        return swizzle<w,w,w,w>(value);
    }

    __device__ inline float4 set(float v1, float v2, float v3, float v4)
    {
        return make_float4(v1, v2, v3, v4);
    }

    __device__ inline uint4 set_uint32(uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4)
    {
        return make_uint4(v1, v2, v3, v4);
    }

    __device__ inline float4 load1(const void* const __restrict address)
    {
        float v = *reinterpret_cast<const float* __restrict> (address);
        return make_float4(v, 0.0f, 0.0f, 0.0f);
    }

    __device__ inline float4 load2(const void* const address)
    {
        const float* __restrict v = reinterpret_cast<const float* __restrict> (address);
        return make_float4(*v, *(v + 1), 0.0f, 0.0f);
    }

    __device__ inline float4 load3(const void* __restrict const address)
    {
        const float* __restrict v = reinterpret_cast<const float* __restrict> (address);
        return make_float4(*v, *(v + 1), *(v + 2), 0.0f);
    }

    __device__ inline uint4 load3u(const void* __restrict const address)
    {
        const uint32_t* __restrict v = reinterpret_cast<const uint32_t* __restrict> (address);
        return make_uint4(*v, *(v + 1), *(v + 2), 0);
    }

    __device__ inline float4 load4(const void* __restrict const address)
    {
        const float* __restrict v = reinterpret_cast<const float* __restrict> (address);
        return make_float4(*v, *(v + 1), *(v + 2), *(v+3) );
    }

    __device__ inline float4 load4(const float* __restrict const address)
    {
        const float* __restrict v = reinterpret_cast<const float* __restrict> (address);
        return make_float4(*v, *(v + 1), *(v + 2), *(v + 3));
    }

    __device__ inline int4 load4i(const void* __restrict const address)
    {
        const uint32_t* __restrict v = reinterpret_cast<const uint32_t* __restrict> (address);
        return make_int4(*v, *(v + 1), *(v + 2), *(v+3) );
    }

    __device__ inline uint4 load4u(const void* __restrict const address)
    {
        const uint32_t* __restrict v = reinterpret_cast<const uint32_t* __restrict> (address);
        return make_uint4(*v, *(v + 1), *(v + 2), *(v + 3));
    }

    __device__ inline uint4 load4u(const float* __restrict const address)
    {
        const uint32_t* __restrict v = reinterpret_cast<const uint32_t * __restrict> (address);
        return make_uint4(*v, *(v + 1), *(v + 2), *(v + 3));
    }

    __device__ inline void store1(void* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
    }

    __device__ inline void store1(float* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
    }

    __device__ inline void store2(void* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
        *(v+1) = value.y;
    }

    __device__ inline void store2(float* __restrict address, uint4 value)
    {
        uint32_t* __restrict v = reinterpret_cast<uint32_t * __restrict  > (address);
        *v = value.x;
        *(v + 1) = value.y;
        *(v + 2) = value.z;
        *(v + 3) = value.w;
    }

    __device__ inline void store3(void* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
        *(v + 1) = value.y;
        *(v + 2) = value.z;
    }

    __device__ inline void store3(float* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
        *(v + 1) = value.y;
        *(v + 2) = value.z;
    }

    __device__ inline void store4(void* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
        *(v + 1) = value.y;
        *(v + 2) = value.z;
        *(v + 3) = value.w;
    }

    __device__ inline void store4(float* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
        *(v + 1) = value.y;
        *(v + 2) = value.z;
        *(v + 3) = value.w;
    }

    __device__ inline void stream(void* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
        *(v + 1) = value.y;
        *(v + 2) = value.z;
        *(v + 3) = value.w;
    }

    __device__ inline void stream(float* __restrict address, float4 value)
    {
        float* __restrict v = reinterpret_cast<float * __restrict  > (address);
        *v = value.x;
        *(v + 1) = value.y;
        *(v + 2) = value.z;
        *(v + 3) = value.w;
    }

    //compare operations

    namespace details
    {
        __device__ inline uint32_t cmpeq(float a, float b)
        {
            return a == b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmplt(float a, float b)
        {
            return a < b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmple(float a, float b)
        {
            return a <= b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmpgt(float a, float b)
        {
            return a > b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmpgt(uint32_t a, uint32_t b)
        {
            return a > b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmpge(float a, float b)
        {
            return a >= b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmpneq(float a, float b)
        {
            return a != b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmpnlt(float a, float b)
        {
            return a > b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmpnle(float a, float b)
        {
            return a >= b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmpngt(float a, float b)
        {
            return a <= b ? 0xFFFFFFFF : 0;
        }

        __device__ inline uint32_t cmpnge(float a, float b)
        {
            return a < b ? 0xFFFFFFFF : 0;
        }

        template <typename T> __device__ inline int32_t signum(T x, std::false_type is_signed)
        {
            return T(0) < x;
        }

        template <typename T> __device__ inline int32_t signum(T x, std::true_type is_signed)
        {
            return (T(0) < x) - (x < T(0));
        }

        template <typename T> __device__ inline int32_t signum(T x)
        {
            return signum(x, std::is_signed<T>());
        }
    }

    __device__ inline uint4 compare_eq(float4 v1, float4 v2)
    {
        return make_uint4(details::cmpeq(v1.x, v2.x), details::cmpeq(v1.y, v2.y), details::cmpeq(v1.z, v2.z), details::cmpeq(v1.w, v2.w) );
    }

    __device__ inline uint4 compare_lt(float4 v1, float4 v2)
    {
        return make_uint4(details::cmplt(v1.x, v2.x), details::cmplt(v1.y, v2.y), details::cmplt(v1.z, v2.z), details::cmplt(v1.w, v2.w));
    }

    __device__ inline uint4 compare_le(float4 v1, float4 v2)
    {
        return make_uint4(details::cmple(v1.x, v2.x), details::cmple(v1.y, v2.y), details::cmple(v1.z, v2.z), details::cmple(v1.w, v2.w));
    }

    __device__ inline uint4 compare_gt(float4 v1, float4 v2)
    {
        return make_uint4(details::cmpgt(v1.x, v2.x), details::cmpgt(v1.y, v2.y), details::cmpgt(v1.z, v2.z), details::cmpgt(v1.w, v2.w));
    }

    __device__ inline uint4 compare_gt(uint4 v1, uint4 v2)
    {
        return make_uint4(details::cmpgt(v1.x, v2.x), details::cmpgt(v1.y, v2.y), details::cmpgt(v1.z, v2.z), details::cmpgt(v1.w, v2.w));
    }

    __device__ inline uint4 compare_ge(float4 v1, float4 v2)
    {
        return make_uint4(details::cmpge(v1.x, v2.x), details::cmpge(v1.y, v2.y), details::cmpge(v1.z, v2.z), details::cmpge(v1.w, v2.w));
    }

    __device__ inline uint4 compare_not_eq(float4 v1, float4 v2)
    {
        return make_uint4(details::cmpneq(v1.x, v2.x), details::cmpneq(v1.y, v2.y), details::cmpneq(v1.z, v2.z), details::cmpneq(v1.w, v2.w));
    }

    __device__ inline uint4 compare_not_lt(float4 v1, float4 v2)
    {
        return make_uint4(details::cmpnlt(v1.x, v2.x), details::cmpnlt(v1.y, v2.y), details::cmpnlt(v1.z, v2.z), details::cmpnlt(v1.w, v2.w));
    }

    __device__ inline uint4 compare_not_le(float4 v1, float4 v2)
    {
        return make_uint4(details::cmpnle(v1.x, v2.x), details::cmpnle(v1.y, v2.y), details::cmpnle(v1.z, v2.z), details::cmpnle(v1.w, v2.w));
    }

    __device__ inline uint4 compare_not_gt(float4 v1, float4 v2)
    {
        return make_uint4(details::cmpngt(v1.x, v2.x), details::cmpngt(v1.y, v2.y), details::cmpngt(v1.z, v2.z), details::cmpngt(v1.w, v2.w));
    }

    __device__ inline uint4 compare_not_ge(float4 v1, float4 v2)
    {
        return make_uint4(details::cmpnge(v1.x, v2.x), details::cmpnge(v1.y, v2.y), details::cmpnge(v1.z, v2.z), details::cmpnge(v1.w, v2.w));
    }

    //simple logical operations
    __device__ inline uint4 and(uint4 v1, uint4 v2)
    {
        return make_uint4(v1.x & v2.x, v1.y & v2.y, v1.z & v2.z, v1.w & v2.w);
    }

    __device__ inline uint4 and_not(uint4 v1, uint4 v2)
    {
        return make_uint4(~(v1.x & v2.x), ~(v1.y & v2.y), ~(v1.z & v2.z), ~(v1.w & v2.w));
    }

    __device__ inline uint4 or(uint4 v1, uint4 v2)
    {
        return make_uint4(v1.x | v2.x, v1.y | v2.y, v1.z | v2.z, v1.w | v2.w);
    }

    __device__ inline uint4 xor(uint4 v1, uint4 v2)
    {
        return make_uint4(v1.x ^ v2.x, v1.y ^ v2.y, v1.z ^ v2.z, v1.w ^ v2.w);
    }

    //misc functions
    __device__ inline uint32_t movemask(float4 v)
    {
        uint32_t r = details::signum(v.w) << 3 | details::signum(v.z) << 2 | details::signum(v.y) << 1 | details::signum(v.x);
        return r;
    }

    __device__ inline float4 select(float4 value1, float4 value2, uint4 control)
    {
        uint4 v0 = make_uint4(__float_as_int(value1.x), __float_as_int(value1.y), __float_as_int(value1.z), __float_as_int(value1.w));
        uint4 v1 = make_uint4(__float_as_int(value2.x), __float_as_int(value2.y), __float_as_int(value2.z), __float_as_int(value2.w));

        auto  v2 = and_not(control, v0);
        auto  v3 = and(v1, control);
        auto  v4 = or(v2, v3);

        return make_float4(__int_as_float(v4.x), __int_as_float(v4.y), __int_as_float(v4.z), __int_as_float(v4.w) );
    }

    __device__ inline uint4 select_control(uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4)
    {
        uint4 v = set_uint32(v1, v2, v3, v4);
        uint4 z = make_uint4(0, 0, 0, 0 );
        return compare_gt( v, z );
    }

    //simple math operations
    __device__ inline float4 add(float4 v1, float4 v2)
    {
        return make_float4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v1.w);
    }

    //simple math operations
    __device__ inline float4 horizontal_add(float4 v1, float4 v2)
    {
        return make_float4(v1.x + v1.y, v1.z + v1.w, v2.x + v2.y, v2.z + v2.w);
    }

    __device__ inline float4 sub(float4 v1, float4 v2)
    {
        return make_float4(v1.x - v1.y, v1.z - v1.w, v2.x - v2.y, v2.z - v2.w);
    }

    //simple math operations
    __device__ inline float4 horizontal_sub(float4 v1, float4 v2)
    {
        return make_float4(v1.x - v1.y, v1.z - v1.w, v2.x - v2.y, v2.z - v2.w);
    }

    __device__ inline float4 mul(float4 v1, float4 v2)
    {
        return make_float4(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v1.w);
    }

    __device__ inline float4 mul(float4 v, float f)
    {
        float4 v1 = splat(f);
        return mul(v, v1);
    }

    __device__ inline float4 mad(float4 v1, float4 v2, float4 v3)
    {
        return add( mul(v1, v2) , v3 ) ;
    }

    __device__ inline float4 div(float4 v1, float4 v2)
    {
        return make_float4(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v1.w);
    }

    __device__ inline float4 sqrt(float4 v)
    {
        return make_float4( sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
    }

    __device__ inline float4 rcp(float4 v)
    {
        return make_float4(__frcp_rz(v.x), __frcp_rz(v.y), __frcp_rz(v.z), __frcp_rz(v.w) );
    }

    __device__ inline float4 rsqrt(float4 v)
    {
        return make_float4(::rsqrt(v.x), ::rsqrt(v.x), ::rsqrt(v.x), ::rsqrt(v.x));
    }

    __device__ inline float4 min(float4 v1, float4 v2)
    {
        return make_float4(::min(v1.x, v2.x), ::min(v1.y, v2.y), ::min(v1.z, v2.z), ::min(v1.w, v2.w));
    }

    __device__ inline float4 max(float4 v1, float4 v2)
    {
        return make_float4(::max(v1.x, v2.x), ::max(v1.y, v2.y), ::max(v1.z, v2.z), ::max(v1.w, v2.w));
    }

    __device__ inline float4 clamp(float4 v, float4 a, float4 b)
    {
        float4 v1 = min(v,b);
        float4 v2 = max(v1,a);
        return v2;
    }

    __device__ inline float4 saturate(float4 v)
    {
        return clamp(v, zero(), one() );
    }

    __device__ inline float4 negate(float4 v)
    {
        return mul( v, minus_one() );
    }

    __device__ inline float4 abs(float4 v)
    {
        float4 v3 = sub( zero(), v );
        float4 v4 = max( v, v3);
        return v4;
    }

    __device__ inline float4 lerp(float4 v1, float4 v2, float4 l)
    {
        float4 a = sub (v2, v1);
        return mad( l, a , v2 );
    }

    __device__ inline float4 lerp(float4 v1, float4 v2, float l)
    {
        float4 s = splat(l);
        return lerp(v1, v2, s);
    }

    //math functions
    __device__ inline float4 dot2(float4 v1, float4 v2)
    {
        float4 v3 = mul(v1, v2);
        float4 v4 = swizzle<x,x,x,x>(v3);
        float4 v5 = swizzle<y,y,y,y>(v3);
        return add(v4, v5);
    }

    __device__ inline float4 dot3(float4 v1, float4 v2)
    {
        float4 v3 = mul(v1, v2);
        float4 v4 = swizzle<x,x,x,x>(v3);
        float4 v5 = swizzle<y,y,y,y>(v3);
        float4 v6 = swizzle<z,z,z,z>(v3);
        float4 v7 = add(v4, v5);
        return add(v6, v7);
    }

    __device__ inline float4 dot4(float4 v1, float4 v2)
    {
        float4 v3 = mul(v1, v2);
        float4 v4 = horizontal_add(v3, v3);
        float4 v5 = horizontal_add(v4, v4);
        return v5;
    }


    __device__ inline float4 length2(float4 v)
    {
        float4 d = dot2(v, v);
        float4 l = sqrt(d);
        return l;
    }

    __device__ inline float4 length3(float4 v)
    {
        float4 d = dot3(v, v);
        float4 l = sqrt(d);
        return l;
    }

    __device__ inline float4 length4(float4 v)
    {
        float4 d = dot4(v, v);
        float4 l = sqrt(d);
        return l;
    }

    __device__ inline float4 normalize2(float4 v)
    {
        float4 l = length2(v);
        float4 n = div(v, l);
        return n;
    }

    __device__ inline float4 normalize3(float4 v)
    {
        float4 l = length3(v);
        float4 n = div(v, l);
        return n;
    }

    __device__ inline float4 normalize4(float4 v)
    {
        float4 l = length4(v);
        float4 n = div(v, l);
        return n;
    }

    __device__ inline float4 normalize_plane(float4 v)
    {
        float4 l = length3(v);
        float4 n = div(v, l);
        return n;
    }

    __device__ inline float4 cross2(float4 v1, float4 v2)
    {
        float4 v3 = swizzle<x,y,x,y>(v2);
        float4 v4 = mul(v1, v3);
        float4 v5 = swizzle<y,y,y,y>(v4);
        float4 v6 = sub(v4, v5);
        float4 v7 = swizzle<x,x,x,x>(v6);
        return v7;
    }

    __device__ inline float4 cross3(float4 v1, float4 v2)
    {
        float4 v3 = swizzle<y,z,x,w>(v1);
        float4 v4 = swizzle<z,x,y,w>(v2);
        float4 v5 = mul(v3, v4);

        float4 v6 = swizzle<z,x,y,w>(v1);
        float4 v7 = swizzle<y,z,x,w>(v2);

        float4 v8 = mul(v6,v7);
        float4 v9 = sub( v5, v8);

        return v9;
    }

    __device__ inline float4 ortho2(float4 v)
    {
        float4 v3 = swizzle<y,x,z,w>(v);
        float4 v4 = negate(v3);
        return v4;
    }

    __device__ inline float4 ortho4(float4 v)
    {
        float4 v3 = swizzle<y,x,w,z>(v);
        float4 v4 = negate(v3);
        return v4;
    }

    __device__ inline float get_x(float4 v)
    {
        return details::get_component<x>(v);
    }

    __device__ inline float get_y(float4 v)
    {
        return details::get_component<y>(v);
    }

    __device__ inline float get_z(float4 v)
    {
        return details::get_component<z>(v);
    }

    __device__ inline float get_w(float4 v)
    {
        return details::get_component<w>(v);
    }

    __device__ inline uint4 mask_x()
    {
        return make_uint4(0xFFFFFFFF, 0, 0, 0);
    }

    __device__ inline uint4 mask_y()
    {
        return make_uint4(0, 0xFFFFFFFF, 0, 0);
    }

    __device__ inline uint4 mask_z()
    {
        return make_uint4(0, 0, 0xFFFFFFFF, 0 );
    }

    __device__ inline uint4 mask_w()
    {
        return make_uint4(0, 0, 0, 0xFFFFFFFF );
    }
}
