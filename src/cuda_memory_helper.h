#pragma once

#include <cstdint>
#include <memory>
#include <cuda_runtime.h>

#include "cuda_helper.h"

namespace cuda
{
    inline void* malloc(size_t size)
    {
        void* r = nullptr;
        throw_if_failed(cudaMalloc(&r, size));
        return r;
    }

    inline void free(void* pointer)
    {
        throw_if_failed(cudaFree(pointer));
    }

    class default_cuda_allocator
    {
    public:

        void* allocate(std::size_t size)
        {
            return cuda::malloc(size);
        }

        void free(void* pointer)
        {
            cuda::free(pointer);
        }
    };

    template < typename allocator_t >
    class memory_buffer_
    {
    public:

        typedef allocator_t                       allocator_type;

        typedef memory_buffer_<allocator_t>       this_type;

    private:

        void*                               m_value;
        size_t                              m_size;
        allocator_type                      m_allocator;

        void swap(this_type & rhs)
        {
            std::swap(m_allocator, rhs.m_allocator);
            std::swap(m_value, rhs.m_value);
            std::swap(m_size, rhs.m_size);
        }

    public:

        memory_buffer_(size_t size, allocator_type alloc = allocator_type()) :
            m_size(size)
            , m_allocator(alloc)
        {
            m_value = m_allocator.allocate(size);
        }

        memory_buffer_(void* value, size_t size) :
            m_value(reinterpret_cast<int*> (value))
            , m_size(size)
        {

        }

        memory_buffer_(this_type&& rhs) : m_value(rhs.m_value), m_size(rhs.m_size), m_allocator(std::move(rhs.m_allocator))
        {
            rhs.m_value = nullptr;
        }

        memory_buffer_ & operator=(this_type && rhs)
        {
            this_type(static_cast<this_type &&>(rhs)).swap(*this);
            return *this;
        }


        ~memory_buffer_()
        {
            m_allocator.free(m_value);
        }

        const void*    get() const
        {
            return m_value;
        }

        void*    get()
        {
            return m_value;
        }

        void*    reset()
        {
            auto r = m_value;
            m_value = nullptr;
            return r;
        }

        size_t size() const
        {
            return m_size;
        }

    private:

        memory_buffer_(const memory_buffer_ &);
        memory_buffer_& operator=(const memory_buffer_&);
    };

    class memory_buffer : public memory_buffer_ < default_cuda_allocator >
    {
        typedef memory_buffer_<default_cuda_allocator>  base;
        typedef memory_buffer                           this_type;

    public:

        memory_buffer(size_t size, default_cuda_allocator alloc = default_cuda_allocator()) :
            base(size, alloc)
        {

        }

        memory_buffer(void* pointer, size_t size) :
            base(pointer, size)
        {

        }

        memory_buffer(memory_buffer&& rhs) : base(std::move(rhs))
        {

        }

        memory_buffer & operator=(memory_buffer && rhs)
        {
            base::operator=(std::move(rhs));
            return *this;
        }

    private:

        memory_buffer(const memory_buffer&);
        memory_buffer& operator=(const memory_buffer&);
    };

    inline memory_buffer* make_memory_buffer( size_t size )
    {
        return new memory_buffer( size );
    }

    inline memory_buffer* make_memory_buffer( size_t size, const void* initial_host_data )
    {
        std::unique_ptr<memory_buffer> buffer( new memory_buffer(size) ) ;
        cuda::throw_if_failed< cuda::exception >( cudaMemcpy (buffer->get() , initial_host_data, size, cudaMemcpyHostToDevice) );
        return buffer.release();
    }

    template <typename t>
    struct default_delete
    {
        default_delete() throw()
        {

        }

        void operator()( t* pointer ) const throw()
        {
            ::cudaFree(pointer);
        }
    };

}


