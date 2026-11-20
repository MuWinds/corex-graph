/* test_guard.c —— 验证 graph_guard：超 128 节点的捕获必须被拦住、且不伤卡。
 * 只 capture + EndCapture（不 replay 超限图），所以即使护栏失效也最多是建图失败。 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUgraph;
typedef void *CUgraphNode;
typedef void *CUgraphExec;
typedef unsigned long long CUdeviceptr;

static void *h;
static void *sym(const char *a, const char *b)
{
    void *p = dlsym(h, a);
    if (!p && b) p = dlsym(h, b);
    return p;
}
static void st(const char *s) { printf("STAGE %s\n", s); fflush(stdout); }
static int ck(CUresult r, const char *n)
{
    printf("%s rc=%d\n", n, r); fflush(stdout);
    return r == 0;
}

int main(int argc, char **argv)
{
    int nodes = argc > 1 ? atoi(argv[1]) : 2;
    st("start");
    h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen FAIL\n"); return 2; }

    CUresult (*pInit)(int) = sym("cuInit", 0);
    CUresult (*pDev)(CUdevice *, int) = sym("cuDeviceGet", 0);
    CUresult (*pCtx)(CUcontext *, unsigned, CUdevice) = sym("cuCtxCreate_v2", "cuCtxCreate");
    CUresult (*pStream)(CUstream *, unsigned) = sym("cuStreamCreate", 0);
    CUresult (*pAlloc)(CUdeviceptr *, unsigned long long) = sym("cuMemAlloc_v2", "cuMemAlloc");
    CUresult (*pFree)(CUdeviceptr) = sym("cuMemFree_v2", "cuMemFree");
    CUresult (*pCpy)(CUdeviceptr, const void *, unsigned long long, CUstream) = sym("cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    CUresult (*pBegin)(CUstream, int) = sym("cuStreamBeginCapture_v2", "cuStreamBeginCapture");
    CUresult (*pEnd)(CUstream, CUgraph *) = sym("cuStreamEndCapture", 0);
    CUresult (*pInst)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned) =
        sym("cuGraphInstantiate_v2", "cuGraphInstantiate");
    CUresult (*pLaunch)(CUgraphExec, CUstream) = sym("cuGraphLaunch", 0);
    CUresult (*pInstD)(CUgraph) = sym("cuGraphDestroy", 0);
    CUresult (*pSync)(CUstream) = sym("cuStreamSynchronize", 0);
    if (!pInit || !pDev || !pCtx || !pAlloc || !pBegin || !pEnd) { printf("dlsym FAIL\n"); return 2; }
    if (!pCpy) {
        printf("memcpy symbol FAIL\n"); return 2;
    }
    printf("pCpy=%p pBegin=%p pEnd=%p\n", (void *)pCpy, (void *)pBegin, (void *)pEnd);
    fflush(stdout);

    ck(pInit(0), "cuInit");
    CUdevice dev = 0;
    ck(pDev(&dev, 0), "cuDeviceGet");
    CUcontext ctx = 0;
    st("before cuCtxCreate");
    ck(pCtx(&ctx, 0, dev), "cuCtxCreate");
    st("after cuCtxCreate");

    CUstream s = 0;
    if (pStream) ck(pStream(&s, 0), "cuStreamCreate");
    CUdeviceptr buf = 0;
    ck(pAlloc(&buf, 1ULL << 20), "cuMemAlloc");
    static char host[4096];
    for (int i = 0; i < 4096; i++) host[i] = (char)i;

    printf("CAPTURE nodes=%d\n", nodes); fflush(stdout);
    st("before BeginCapture");
    if (!ck(pBegin(s, 0), "cuStreamBeginCapture")) return 1;
    for (int i = 0; i < nodes; i++) {
        CUresult r = pCpy(buf + (unsigned long long)(i % 256) * 4096ULL, host, 4096, s);
        if (r != 0) { printf("memcpy[%d] rc=%d\n", i, r); break; }
    }
    st("before EndCapture");
    CUgraph g = 0;
    CUresult r = pEnd(s, &g);
    printf("cuStreamEndCapture rc=%d (0x%x)\n", r, r); fflush(stdout);

    if (r == 0) {
        CUgraphExec e = 0;
        if (pInst) {
            CUresult ri = pInst(&e, g, NULL, NULL, 0);
            printf("cuGraphInstantiate rc=%d (0x%x)\n", ri, ri);
            if (ri == 0x1) {
                printf("GUARD_REJECTED nodes=%d\n", nodes);
            } else if (ri == 0 && pLaunch && pSync) {
                ck(pLaunch(e, s), "cuGraphLaunch");
                ck(pSync(s), "cuStreamSynchronize");
            }
            fflush(stdout);
        }
        if (pInstD) pInstD(g);
    }

    if (pFree) ck(pFree(buf), "cuMemFree");
    printf("TEST_GUARD_DONE nodes=%d rc=%d\n", nodes, r); fflush(stdout);
    return 0;
}
