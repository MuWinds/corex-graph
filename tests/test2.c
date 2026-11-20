/* test2.c —— 零负载探针：只建上下文 + 分配/释放 1 MiB，不 launch 任何 kernel。
 * 目的：验证 cmdbuf_grow 的缓冲池能否拿到 unify(host VA == device VA) 的设备内存。
 * 不依赖 cuda.h（全走 dlopen），避免头文件/ABI 不一致。 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
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
static int ck(CUresult r, const char *n)
{
    printf("%s rc=%d %s\n", n, r, r == 0 ? "OK" : "FAIL");
    return r == 0;
}

int main(void)
{
    h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen libcuda FAIL: %s\n", dlerror()); return 2; }

    CUresult (*pInit)(int)                                  = sym("cuInit", 0);
    CUresult (*pDev)(CUdevice *, int)                       = sym("cuDeviceGet", 0);
    CUresult (*pCtx)(CUcontext *, unsigned, CUdevice)        = sym("cuCtxCreate_v2", "cuCtxCreate");
    CUresult (*pAlloc)(CUdeviceptr *, unsigned long long)    = sym("cuMemAlloc_v2", "cuMemAlloc");
    CUresult (*pFree)(CUdeviceptr)                          = sym("cuMemFree_v2", "cuMemFree");
    CUresult (*pCtxD)(CUcontext)                            = sym("cuCtxDestroy_v2", "cuCtxDestroy");
    if (!pInit || !pDev || !pCtx || !pAlloc || !pFree) { printf("dlsym FAIL\n"); return 2; }

    if (!ck(pInit(0), "cuInit")) return 1;
    CUdevice dev = 0;
    if (!ck(pDev(&dev, 0), "cuDeviceGet")) return 1;
    CUcontext ctx = 0;
    if (!ck(pCtx(&ctx, 0, dev), "cuCtxCreate")) return 1;
    CUdeviceptr p = 0;
    if (!ck(pAlloc(&p, 1ULL << 20), "cuMemAlloc")) return 1;
    printf("ptr=0x%llx\n", p);
    ck(pFree(p), "cuMemFree");
    if (pCtxD) ck(pCtxD(ctx), "cuCtxDestroy");
    printf("TEST2_DONE\n");
    return 0;
}
