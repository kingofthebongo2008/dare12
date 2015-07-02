#include "precompiled.h"
#include "rendering_application.h"

#include "imaging_utils.h"


namespace freeform
{
    void display(const imaging::cuda_texture& t)
    {
        std::unique_ptr< sample_application >  app( new sample_application(L"Sample Application", t )  );


        

        auto result = app->run();
    }
}

