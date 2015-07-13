    std::vector<patch> create_vector(const thrust::device_vector<patch>& p)
    {
        using namespace thrust;
        host_vector<patch> h;
        std::vector<patch> h1;

        h.resize(p.size());
        h1.resize(p.size());

        copy(p.begin(), p.end(), h.begin());

        std::copy(h.begin(), h.end(), h1.begin());

        return h1;
    }

    std::vector<patch> create_vector(const std::vector<patch>& p0, const std::vector<patch>& p1)
    {
        std::vector<patch> rv;

        rv.resize(p0.size());

        for (auto i = 0; i < p0.size(); ++i)
        {
            patch a = p0[i];
            patch b = p1[i];


            patch r;

            r.x0 = a.x0 - b.x0;
            r.x1 = a.x1 - b.x1;
            r.x2 = a.x2 - b.x2;
            r.x3 = a.x3 - b.x3;

            r.y0 = a.y0 - b.y0;
            r.y1 = a.y1 - b.y1;
            r.y2 = a.y2 - b.y2;
            r.y3 = a.y3 - b.y3;

            rv[i] = r;
        }

        return rv;
    }

    float abs_diff(const std::vector<patch>& p0)
    {
        float maximum = 0.0f;

        for (auto i = 0; i < p0.size(); ++i)
        {
            patch r = p0[i];

            auto a0 = abs(r.x0);
            auto a1 = abs(r.x1);
            auto a2 = abs(r.x2);
            auto a3 = abs(r.x3);

            auto b0 = abs(r.y0);
            auto b1 = abs(r.y1);
            auto b2 = abs(r.y2);
            auto b3 = abs(r.y3);

            auto m = std::max(a0, a1);

            m = std::max(m, a2);
            m = std::max(m, a3);

            auto n = std::max(b0, b1);

            n = std::max(n, b2);
            n = std::max(n, b3);

            maximum = std::max(maximum, std::max(m, n));

        }

        return maximum;
    }

