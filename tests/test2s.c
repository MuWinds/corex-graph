/* test2s.c —— 分步打点版零负载探针：每一步立即 flush，能定位死在哪一步。 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
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
    printf("%s rc=%d %s\n", n, r, r == 0 ? "OK" : "FAIL"); fflush(stdout);
    return r == 0;
}

int main(int argc, char **argv)
{
    int ord = (argc > 1) ? atoi(argv[1]) : 0;   /* 逻辑设备号（CVD 列表里的位置） */
    st("start");
    h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen FAIL: %s\n", dlerror()); return 2; }
    st("dlopen_ok");

    CUresult (*pInit)(int)                               = sym("cuInit", 0);
    CUresult (*pDev)(CUdevice *, int)                    = sym("cuDeviceGet", 0);
    CUresult (*pCtx)(CUcontext *, unsigned, CUdevice)     = sym("cuCtxCreate_v2", "cuCtxCreate");
    CUresult (*pAlloc)(CUdeviceptr *, unsigned long long) = sym("cuMemAlloc_v2", "cuMemAlloc");
    CUresult (*pFree)(CUdeviceptr)                       = sym("cuMemFree_v2", "cuMemFree");
    CUresult (*pCtxD)(CUcontext)                         = sym("cuCtxDestroy_v2", "cuCtxDestroy");
    if (!pInit || !pDev || !pCtx || !pAlloc || !pFree) { printf("dlsym FAIL\n"); return 2; }
    st("dlsym_ok");

    st("before cuInit");
    if (!ck(pInit(0), "cuInit")) return 1;
    st("after cuInit");

    CUdevice dev = 0;
    if (!ck(pDev(&dev, ord), "cuDeviceGet")) return 1;
    st("after cuDeviceGet");

    CUcontext ctx = 0;
    st("before cuCtxCreate");
    if (!ck(pCtx(&ctx, 0, dev), "cuCtxCreate")) return 1;
    st("after cuCtxCreate");

    CUdeviceptr p = 0;
    st("before cuMemAlloc");
    if (!ck(pAlloc(&p, 1ULL << 20), "cuMemAlloc")) return 1;
    printf("ptr=0x%llx\n", p); fflush(stdout);

    st("before cuMemFree");
    ck(pFree(p), "cuMemFree");
    st("after cuMemFree");

    st("before cuCtxDestroy");
    if (pCtxD) ck(pCtxD(ctx), "cuCtxDestroy");
    st("after cuCtxDestroy");
    printf("TEST2S_DONE\n"); fflush(stdout);
    return 0;
}
