/* test_graph.c —— capture 两个 memcpy 节点 → instantiate → 连续 replay N 次 → 逐次校验。
 * 走的是 cuStreamBeginCapture / cuGraphInstantiate / cuGraphLaunch 全链路，
 * replay 时由 0x38a900 引擎逐节点重调公开 API，命令被写进 UMD 的 cmd 块。
 * 旧容量 8192 时，累积后必然越界；本用例用来验证抬高容量后的行为。
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef CUresult (*PFN_begin)(CUstream, CUstreamCaptureMode);

static PFN_begin resolve_begin(void)
{
    PFN_begin f = (PFN_begin)dlsym(RTLD_DEFAULT, "cuStreamBeginCapture_v2");
    if (!f) f = (PFN_begin)dlsym(RTLD_DEFAULT, "cuStreamBeginCapture_v1");
    if (!f) f = (PFN_begin)dlsym(RTLD_DEFAULT, "cuStreamBeginCapture");
    return f;
}

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
    const char *s_ = "?"; cuGetErrorName(r_, &s_); \
    fprintf(stderr, "FAIL %s rc=%d (%s) line=%d\n", #x, (int)r_, s_, __LINE__); \
    return 1; } } while (0)

#define N       (4096)
#define REPLAY  200

int main(void)
{
    CUdevice dev; CUcontext ctx; CUstream st; CUgraph g; CUgraphExec ge;
    CUdeviceptr dst; unsigned char *src = NULL;
    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));
    CK(cuCtxCreate(&ctx, 0, dev));
    CK(cuMemAlloc(&dst, N));
    CK(cuMemAllocHost((void **)&src, N));
    for (int i = 0; i < N; i++) src[i] = (unsigned char)(i & 0xff);
    CK(cuStreamCreate(&st, CU_STREAM_NON_BLOCKING));

    PFN_begin begin = resolve_begin();
    if (!begin) { fprintf(stderr, "no cuStreamBeginCapture symbol\n"); return 1; }
    CK(begin(st, CU_STREAM_CAPTURE_MODE_GLOBAL));
    int nodes = 2;
    const char *ne = getenv("GRAPH_NODES");
    if (ne) nodes = atoi(ne);
    if (nodes < 1) nodes = 1;
    if (nodes > 512) nodes = 512;
    for (int i = 0; i < nodes; i++) {
        int off = (i * 16) % (N - 16);
        CK(cuMemcpyHtoDAsync(dst + off, src + off, 16, st));
    }
    CK(cuStreamEndCapture(st, &g));
    printf("CAPTURE_OK nodes=%d\n", nodes); fflush(stdout);

    CK(cuGraphInstantiate(&ge, g, NULL, NULL, 0));
    printf("INSTANTIATE_OK\n"); fflush(stdout);

    int bad = 0;
    unsigned char *host = NULL;
    CK(cuMemAllocHost((void **)&host, N));
    for (int i = 0; i < REPLAY; i++) {
        CK(cuMemsetD8(dst, 0, N));
        CK(cuGraphLaunch(ge, st));
        CK(cuCtxSynchronize());
        CK(cuMemcpyDtoH(host, dst, N));
        for (int k = 0; k < nodes; k++) {
            int off = (k * 16) % (N - 16);
            if (memcmp(host + off, src + off, 16) != 0) {
                bad++; printf("MISMATCH node=%d replay=%d\n", k, i); fflush(stdout);
                break;
            }
        }
        if ((i % 5) == 4) { printf("REPLAY %d ok bad=%d\n", i + 1, bad); fflush(stdout); }
    }
    cuMemFreeHost(host);
    printf("GRAPH_TEST_DONE replay=%d bad=%d\n", REPLAY, bad); fflush(stdout);
    return bad ? 2 : 0;
}
