// Decode-attention microbenchmark for the nextsycl inkling2 silo (attn.cpp + dec.cpp merge), synthetic data.
// Built once per source tree (scripts/inkling-attn-bench.sh); each binary times the silo's own entry points and writes the
// outputs of a fixed-seed case to a file, so two trees can be compared bit for bit (cmp) and timed alternately.
// usage: inkling-attn-bench OUTDIR [iters]   (BENCH_WIDE=1: the wide case list)
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int ns_fail(const std::string& what) { fprintf(stderr, "ns_fail: %s\n", what.c_str()); return -1; }

using half = sycl::half;
static constexpr int D = 128, HQ = 24, HKV = 8, R = 16, E = 1024, WIN = 1024;

struct Case { const char* name; int64_t T, p0, win, cap; bool block = false; };

int main(int argc, char** argv) {
    const std::string outdir = argc > 1 ? argv[1] : ".";
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    ns_gpu g{q.get_device(), q.get_context(), q, q, q};
    std::vector<Case> cases;
    const int64_t wcap = WIN + 8192 - 1;  // a windowed ring: window + prompt chunk - 1 (engine.rs)
    const bool wide = getenv("BENCH_WIDE") != nullptr;  // BENCH_WIDE=1: every row count and the boundary positions
    if (!wide) {
        for (int64_t T : {1, 2, 4, 7}) {
            cases.push_back({"win", T, 300, WIN, wcap});
            cases.push_back({"win", T, 8000, WIN, wcap});
            for (int64_t p : {2000, 7300, 39300, 99900}) cases.push_back({"full", T, p, 0, p + 64});
        }
    } else {
        for (int64_t T = 1; T <= 8; ++T) {
            for (int64_t p : {0, 63, 64, 1000, 1020, 5000}) cases.push_back({"win", T, p, WIN, wcap});
            for (int64_t p : {0, 63, 1000, 4090, 4096, 4100, 9000}) cases.push_back({"full", T, p, 0, p + 64});
        }
        // the windowed ring's wrap (cap 9,215: a window straddling a multiple of it), 1-8 rows and 9-16 (the one-row kernel)
        for (int64_t T : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16})
            for (int64_t p : {9150, 9214, 9215, 9245, 9700, 10238, 18440}) cases.push_back({"wwrap", T, p, WIN, wcap});
        for (int64_t T : {9, 12, 16}) {
            cases.push_back({"win", T, 5000, WIN, wcap});
            cases.push_back({"full", T, 3000, 0, 3064});
        }
        for (int64_t T : {2, 4, 7, 8})
            for (int64_t p : {9200, 9214, 10238}) cases.push_back({"bwrap", T, p, WIN, wcap, true});
        // the drafter's block attention (every row sees the window from the block's start and the whole block)
        for (int64_t T : {2, 4, 7, 8})
            for (int64_t p : {300, 5000}) {
                cases.push_back({"bwin", T, p, WIN, wcap, true});
                cases.push_back({"bfull", T, p, 0, p + 64, true});
            }
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t maxcap = 100000;
    const size_t kvn = (size_t) HKV * maxcap * D;
    std::vector<half> hk(kvn), hv(kvn);
    for (size_t i = 0; i < kvn; ++i) { hk[i] = half(nd(rng) * 0.5f); hv[i] = half(nd(rng)); }
    half* kc = sycl::malloc_device<half>(kvn, q);
    half* vc = sycl::malloc_device<half>(kvn, q);
    q.memcpy(kc, hk.data(), kvn * 2).wait();
    q.memcpy(vc, hv.data(), kvn * 2).wait();
    const int64_t TM = 16;
    std::vector<float> hq(TM * HQ * D), hr(TM * HQ * R), hb(E * R), hs(TM * HQ), hg(TM * HQ * D);
    for (auto& x : hq) x = nd(rng);
    for (auto& x : hr) x = nd(rng) * 0.3f;
    for (auto& x : hb) x = nd(rng) * 0.3f;
    for (auto& x : hs) x = nd(rng);
    for (auto& x : hg) x = nd(rng);
    auto up = [&](std::vector<float>& h) { float* d = sycl::malloc_device<float>(h.size(), q); q.memcpy(d, h.data(), h.size() * 4).wait(); return d; };
    float *qd = up(hq), *rd = up(hr), *bd = up(hb), *sd = up(hs), *gd = up(hg);
    const int64_t maxns = ns_ink2s_attn_splits(TM, 100000, 0) + 2;
    float* part = sycl::malloc_device<float>(TM * HQ * maxns * (D + 2), q);
    float* scratch = sycl::malloc_device<float>(TM * HQ * (maxns + 1) + TM * HQ, q);
    float* out = sycl::malloc_device<float>(TM * HQ * D, q);
    const int64_t xqb = ns_ink2s_xq_bytes(TM, HQ * D);
    void* xq = sycl::malloc_device(xqb, q);
    printf("%-5s %2s %7s %5s %10s %10s %10s\n", "case", "T", "p0", "ns", "attn_us", "merge_us", "total_us");
    for (const Case& c : cases) {
        const int64_t ns = ns_ink2s_attn_splits(c.T, c.p0, c.win);
        const int64_t ext = c.win ? WIN : E;
        auto attn = [&] {
            if (c.block ? ns_ink2s_attn_part_block(&g, qd, kc, vc, part, c.T, HQ, HKV, c.cap, c.p0, c.win, 1.0f / sqrtf((float) D))
                        : ns_ink2s_attn_part(&g, qd, kc, vc, rd, bd, ext, R, part, c.T, HQ, HKV, c.cap, c.p0, c.win, 1.0f / sqrtf((float) D)))
                { fprintf(stderr, "attn failed\n"); exit(1); }
        };
        auto merge = [&] {
            int rc = c.block ? ns_ink2s_merge_gate_xq(&g, part, nullptr, nullptr, out, xq, c.T, HQ, ns)
                     : (ns >= 128 && ns <= 4096) ? ns_ink2s_merge_exact(&g, part, sd, gd, out, xq, c.T, HQ, ns, scratch)
                                                 : ns_ink2s_merge_gate_xq(&g, part, sd, gd, out, xq, c.T, HQ, ns);
            if (rc) { fprintf(stderr, "merge failed\n"); exit(1); }
        };
        for (int i = 0; i < 5; ++i) { attn(); merge(); }
        q.wait();
        double ta = 0, tm = 0;
        for (int rep = 0; rep < 2; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) attn();
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) merge();
            q.wait();
            auto t2 = std::chrono::steady_clock::now();
            ta += std::chrono::duration<double, std::micro>(t1 - t0).count();
            tm += std::chrono::duration<double, std::micro>(t2 - t1).count();
        }
        ta /= 2.0 * iters; tm /= 2.0 * iters;
        attn(); merge(); q.wait();
        printf("%-5s %2ld %7ld %5ld %10.2f %10.2f %10.2f\n", c.name, (long) c.T, (long) c.p0, (long) ns, ta, tm, ta + tm);
        std::vector<float> ho(c.T * HQ * D);
        q.memcpy(ho.data(), out, ho.size() * 4).wait();
        std::vector<uint8_t> hx(ns_ink2s_xq_bytes(c.T, HQ * D));
        q.memcpy(hx.data(), xq, hx.size()).wait();
        char fn[512];
        snprintf(fn, sizeof fn, "%s/%s-T%ld-p%ld.bin", outdir.c_str(), c.name, (long) c.T, (long) c.p0);
        FILE* f = fopen(fn, "wb");
        fwrite(ho.data(), 4, ho.size(), f);
        fwrite(hx.data(), 1, hx.size(), f);
        fclose(f);
    }
    return 0;
}
