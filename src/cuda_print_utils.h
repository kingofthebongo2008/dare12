#pragma once

#include <iostream>
#include <thrust/copy.h>

namespace freeform
{
    template <typename container, typename t > inline void print(const container& c)
    {
        thrust::copy(c.begin(), c.end(), std::ostream_iterator< t >(std::cout, " "));
    }
}