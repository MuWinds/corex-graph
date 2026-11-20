/* test_split.c —— 验证 cap_split：一次 capture N 个节点，自动切成若干 ≤128 段的
 * 真图，replay 后**逐槽校验每个节点都真的执行了**（不是只看最终缓冲）。
 *
 * 每个节点 = cuMemsetD32Async(dev + i*4, i, 1, stream)，
 * 所以 replay 完 slot[i] 必须等于 i；预填 0xffffffff，没执行的槽一眼能看出来。
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUgraph;
typedef void *CUgraphNode;
typedef void *CUgraphExec;
typedef unsigned long long CUdeviceptr;

static void st(const char *s) { printf("STAGE %s\n", s); fflush(stdout); }
static int ck(CUresult r, const char *n)
{
    printf("%s rc=%d\n", n, r); fflush(stdout);
    return r == 0;
}

int main(int argc, char **argv)
{
    int nodes = argc > 1 ? atoi(argv[1]) : 400;
    int reps  = argc > 2 ? atoi(argv[2]) : 20;
    if (nodes < 1) nodes = 1;
    if (nodes > 4000) nodes = 4000;

    setvbuf(stdout, NULL, _IONBF, 0);
    st("start");
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen FAIL\n"); return 2; }

    CUresult (*pInit)(int) = dlsym(RTLD_DEFAULT, "cuInit");
    CUresult (*pDev)(CUdevice *, int) = dlsym(RTLD_DEFAULT, "cuDeviceGet");
    CUresult (*pCtx)(CUcontext *, unsigned, CUdevice) = dlsym(RTLD_DEFAULT, "cuCtxCreate_v2");
    if (!pCtx) pCtx = (CUresult (*)(CUcontext *, unsigned, CUdevice))dlsym(RTLD_DEFAULT, "cuCtxCreate");
    CUresult (*pStream)(CUstream *, unsigned) = dlsym(RTLD_DEFAULT, "cuStreamCreate");
    CUresult (*pAlloc)(CUdeviceptr *, unsigned long long) = dlsym(RTLD_DEFAULT, "cuMemAlloc_v2");
    if (!pAlloc) pAlloc = (CUresult (*)(CUdeviceptr *, unsigned long long))dlsym(RTLD_DEFAULT, "cuMemAlloc");
    CUresult (*pFree)(CUdeviceptr) = dlsym(RTLD_DEFAULT, "cuMemFree_v2");
    if (!pFree) pFree = (CUresult (*)(CUdeviceptr))dlsym(RTLD_DEFAULT, "cuMemFree");
    CUresult (*pSet)(CUdeviceptr, unsigned, unsigned long long, CUstream) =
        dlsym(RTLD_DEFAULT, "cuMemsetD32Async");
    CUresult (*pCpyBack)(void *, CUdeviceptr, unsigned long long, CUstream) =
        dlsym(RTLD_DEFAULT, "cuMemcpyDtoHAsync_v2");
    if (!pCpyBack) pCpyBack = (CUresult (*)(void *, CUdeviceptr, unsigned long long, CUstream))
        dlsym(RTLD_DEFAULT, "cuMemcpyDtoHAsync");
    CUresult (*pBegin)(CUstream, int) = dlsym(RTLD_DEFAULT, "cuStreamBeginCapture_v2");
    if (!pBegin) pBegin = (CUresult (*)(CUstream, int))dlsym(RTLD_DEFAULT, "cuStreamBeginCapture");
    CUresult (*pEnd)(CUstream, CUgraph *) = dlsym(RTLD_DEFAULT, "cuStreamEndCapture");
    CUresult (*pInst)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned) =
        dlsym(RTLD_DEFAULT, "cuGraphInstantiate_v2");
    if (!pInst) pInst = (CUresult (*)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned))
        dlsym(RTLD_DEFAULT, "cuGraphInstantiate");
    CUresult (*pLaunch)(CUgraphExec, CUstream) = dlsym(RTLD_DEFAULT, "cuGraphLaunch");
    CUresult (*pSync)(CUstream) = dlsym(RTLD_DEFAULT, "cuStreamSynchronize");
    CUresult (*pGDestroy)(CUgraph) = dlsym(RTLD_DEFAULT, "cuGraphDestroy");
    CUresult (*pEDestroy)(CUgraphExec) = dlsym(RTLD_DEFAULT, "cuGraphExecDestroy");

    if (!pInit || !pDev || !pCtx || !pStream || !pAlloc || !pSet || !pCpyBack ||
        !pBegin || !pEnd || !pInst || !pLaunch || !pSync) {
        printf("dlsym FAIL set=%d back=%d inst=%d\n", !!pSet, !!pCpyBack, !!pInst);
        return 2;
    }

    ck(pInit(0), "cuInit");
    CUdevice dev = 0;
    ck(pDev(&dev, 0), "cuDeviceGet");
    CUcontext ctx = 0;
    st("before cuCtxCreate");
    ck(pCtx(&ctx, 0, dev), "cuCtxCreate");
    st("after cuCtxCreate");

    CUstream s = 0;
    ck(pStream(&s, 0), "cuStreamCreate");

    size_t alloc = ((size_t)(nodes + 16) * 4ULL + 4095ULL) & ~4095ULL;
    CUdeviceptr buf = 0;
    ck(pAlloc(&buf, alloc), "cuMemAlloc");

    int *back = (int *)calloc((size_t)nodes + 16, sizeof(int));

    /* 预填 0xffffffff（这发生在 capture 之外，不计数） */
    ck(pSet(buf, 0xffffffffu, (unsigned long long)(nodes + 16), s), "prefill");
    if (pSync) pSync(s);

    printf("CAPTURE nodes=%d\n", nodes); fflush(stdout);
    if (!ck(pBegin(s, 0), "cuStreamBeginCapture")) return 1;
    int bad_cap = 0;
    for (int i = 0; i < nodes; i++) {
        CUresult r = pSet(buf + (CUdeviceptr)i * 4ULL, (unsigned)i, 1, s);
        if (r != 0) { printf("memset[%d] rc=%d\n", i, r); bad_cap = 1; break; }
    }
    CUgraph g = 0;
    CUresult r = pEnd(s, &g);
    printf("cuStreamEndCapture rc=%d graph=%p\n", r, (void *)g);
    if (r != 0) { printf("TEST_SPLIT_FAIL stage=end\n"); return 1; }

    CUgraphExec e = 0;
    CUresult ri = pInst(&e, g, NULL, NULL, 0);
    printf("cuGraphInstantiate rc=%d exec=%p\n", ri, (void *)e);
    if (ri != 0) { printf("TEST_SPLIT_FAIL stage=inst rc=%d\n", ri); return 1; }

    int bad = 0, nb = 0, fb = -1;
    for (int t = 0; t < reps; t++) {
        if (!ck(pLaunch(e, s), "cuGraphLaunch")) { bad = 1; break; }
        ck(pSync(s), "cuStreamSynchronize");
        memset(back, 0, ((size_t)nodes + 16) * sizeof(int));
        CUresult rb = pCpyBack(back, buf, (unsigned long long)(nodes * 4), s);
        if (rb != 0) { printf("copyback rc=%d\n", rb); bad = 1; break; }
        pSync(s);
        nb = 0; fb = -1;
        for (int i = 0; i < nodes; i++)
            if (back[i] != i) { nb++; if (fb < 0) fb = i; }
        if (nb) {
            printf("REPLAY %d bad=%d first_bad=%d (want %d got %d)\n",
                   t, nb, fb, fb, back[fb]);
            bad = 1;
            break;
        }
    }
    if (!bad) printf("REPLAY %d ok bad=0\n", reps - 1);

    if (!bad) printf("VERIFY nodes=%d reps=%d ALL_OK\n", nodes, reps);
    else      printf("VERIFY nodes=%d reps=%d FAIL bad_slots=%d first_bad=%d\n",
                    nodes, reps, nb, fb);

    if (pSync) pSync(s);
    if (pEDestroy) pEDestroy(e);
    if (pGDestroy) pGDestroy(g);
    if (pFree) pFree(buf);
    free(back);

    printf("TEST_SPLIT_DONE nodes=%d bad=%d bad_cap=%d\n", nodes, bad, bad_cap);
    return 0;
}
