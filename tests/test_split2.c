/* test_split2.c —— 定位两个现象：
 *   A) 同一 stream 上"第一次 capture 已 End 并 instantiate"之后，
 *      第二次 capture 出来的图还能不能 instantiate？
 *   B) 换一条新 stream 做第二次 capture，能不能 instantiate？
 *   C) 第一次 capture 后不 instantiate，直接再 capture，能不能 instantiate？
 * 每步都打 rc，不做任何 replay，不会伤卡。
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

int main(void)
{
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
    CUresult (*pCpy)(CUdeviceptr, const void *, unsigned long long, CUstream) =
        dlsym(RTLD_DEFAULT, "cuMemcpyHtoDAsync_v2");
    if (!pCpy) pCpy = (CUresult (*)(CUdeviceptr, const void *, unsigned long long, CUstream))
        dlsym(RTLD_DEFAULT, "cuMemcpyHtoDAsync");
    CUresult (*pBegin)(CUstream, int) = dlsym(RTLD_DEFAULT, "cuStreamBeginCapture_v2");
    if (!pBegin) pBegin = (CUresult (*)(CUstream, int))dlsym(RTLD_DEFAULT, "cuStreamBeginCapture");
    CUresult (*pEnd)(CUstream, CUgraph *) = dlsym(RTLD_DEFAULT, "cuStreamEndCapture");
    CUresult (*pInst)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned) =
        dlsym(RTLD_DEFAULT, "cuGraphInstantiate_v2");
    if (!pInst) pInst = (CUresult (*)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned))
        dlsym(RTLD_DEFAULT, "cuGraphInstantiate");
    CUresult (*pNodes)(CUgraph, CUgraphNode *, unsigned *) = dlsym(RTLD_DEFAULT, "cuGraphGetNodes");
    CUresult (*pSync)(CUstream) = dlsym(RTLD_DEFAULT, "cuStreamSynchronize");

    if (!pInit || !pDev || !pCtx || !pStream || !pAlloc || !pCpy ||
        !pBegin || !pEnd || !pInst) {
        printf("dlsym FAIL init=%d dev=%d ctx=%d str=%d alloc=%d cpy=%d begin=%d end=%d inst=%d\n",
               !!pInit, !!pDev, !!pCtx, !!pStream, !!pAlloc, !!pCpy,
               !!pBegin, !!pEnd, !!pInst);
        return 2;
    }
    printf("SYM begin=%p end=%p inst=%p\n", (void *)pBegin, (void *)pEnd, (void *)pInst);
    fflush(stdout);

    printf("cuInit rc=%d\n", pInit(0));
    CUdevice dev = 0;
    printf("cuDeviceGet rc=%d\n", pDev(&dev, 0));
    CUcontext ctx = 0;
    st("before cuCtxCreate");
    printf("cuCtxCreate rc=%d\n", pCtx(&ctx, 0, dev));
    st("after cuCtxCreate");

    CUstream s1 = 0, s2 = 0;
    printf("stream1 rc=%d\n", pStream(&s1, 0));
    printf("stream2 rc=%d\n", pStream(&s2, 0));
    CUdeviceptr buf = 0;
    printf("alloc rc=%d\n", pAlloc(&buf, 1ULL << 20));
    int tag[64];
    for (int i = 0; i < 64; i++) tag[i] = i;

    CUgraph g1 = 0, g2 = 0, g3 = 0;
    CUgraphExec e1 = 0, e2 = 0, e3 = 0;
    CUresult rc = 0;
    unsigned n = 0;

    /* 1) 第一次 capture：12 节点 -> End -> instantiate */
    st("A1");
    printf("A1 begin rc=%d\n", pBegin(s1, 0));
    for (int i = 0; i < 12; i++) pCpy(buf + (CUdeviceptr)i * 4ULL, &tag[i], 4, s1);
    rc = pEnd(s1, &g1); printf("A1 end rc=%d g=%p\n", rc, (void *)g1);
    n = 0; if (pNodes && g1) pNodes(g1, NULL, &n);
    printf("A1 nodes=%u\n", n);
    rc = pInst(&e1, g1, NULL, NULL, 0); printf("A1 inst rc=%d e=%p\n", rc, (void *)e1);
    fflush(stdout);

    /* 2) 同一 stream 再 capture 一次：12 节点 -> End -> instantiate */
    st("A2");
    printf("A2 begin rc=%d\n", pBegin(s1, 0));
    for (int i = 0; i < 12; i++) pCpy(buf + (CUdeviceptr)(12 + i) * 4ULL, &tag[i], 4, s1);
    rc = pEnd(s1, &g2); printf("A2 end rc=%d g=%p\n", rc, (void *)g2);
    n = 0; if (pNodes && g2) pNodes(g2, NULL, &n);
    printf("A2 nodes=%u\n", n);
    rc = pInst(&e2, g2, NULL, NULL, 0); printf("A2 inst rc=%d e=%p\n", rc, (void *)e2);
    fflush(stdout);

    /* 3) 换新 stream 做第三次 capture */
    st("A3");
    printf("A3 begin rc=%d\n", pBegin(s2, 0));
    for (int i = 0; i < 12; i++) pCpy(buf + (CUdeviceptr)(24 + i) * 4ULL, &tag[i], 4, s2);
    rc = pEnd(s2, &g3); printf("A3 end rc=%d g=%p\n", rc, (void *)g3);
    n = 0; if (pNodes && g3) pNodes(g3, NULL, &n);
    printf("A3 nodes=%u\n", n);
    rc = pInst(&e3, g3, NULL, NULL, 0); printf("A3 inst rc=%d e=%p\n", rc, (void *)e3);

    if (pSync) pSync(s1);
    printf("TEST_SPLIT2_DONE\n");
    fflush(stdout);
    return 0;
}
