#include "precompiled.h"
#include "rendering_application.h"

#include "imaging_utils.h"

#include "freeform_patch.h"


namespace freeform
{
    void display(const imaging::cuda_texture& t )
    {
        std::unique_ptr< sample_application >  app( new sample_application(L"Sample Application", t )  );

        patch p;

        p.x0 = 0.0f;
        p.x1 = 0.25f;
        p.x2 = 0.75f;
        p.x3 = 1.0f;

        p.y0 = 0.0f;
        p.y1 = 0.25f;
        p.y2 = 0.25f;
        p.y3 = 0.0f;
        

        auto result = app->run();
    }
}

