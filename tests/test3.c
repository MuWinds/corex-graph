/* test3.c —— 第 3 级探针：建上下文 + 1 MiB 分配 + cuMemsetD8 写入。
 * 目的：真正走一次驱动提交（memcpy 族），验证 patch 后 cmd 块换成我们的缓冲仍然工作。
 * 全走 dlopen，不依赖 cuda.h。 */
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
    if (!h) { printf("dlopen FAIL: %s\n", dlerror()); return 2; }

    CUresult (*pInit)(int)                                = sym("cuInit", 0);
    CUresult (*pDev)(CUdevice *, int)                     = sym("cuDeviceGet", 0);
    CUresult (*pCtx)(CUcontext *, unsigned, CUdevice)      = sym("cuCtxCreate_v2", "cuCtxCreate");
    CUresult (*pAlloc)(CUdeviceptr *, unsigned long long)  = sym("cuMemAlloc_v2", "cuMemAlloc");
    CUresult (*pFree)(CUdeviceptr)                        = sym("cuMemFree_v2", "cuMemFree");
    CUresult (*pSet8)(CUdeviceptr, unsigned char, unsigned long long) = sym("cuMemsetD8_v2", "cuMemsetD8");
    CUresult (*pSync)(void)                               = sym("cuCtxSynchronize", 0);
    CUresult (*pCtxD)(CUcontext)                          = sym("cuCtxDestroy_v2", "cuCtxDestroy");
    if (!pInit || !pDev || !pCtx || !pAlloc || !pSet8) { printf("dlsym FAIL\n"); return 2; }

    if (!ck(pInit(0), "cuInit")) return 1;
    CUdevice dev = 0;
    if (!ck(pDev(&dev, 0), "cuDeviceGet")) return 1;
    CUcontext ctx = 0;
    if (!ck(pCtx(&ctx, 0, dev), "cuCtxCreate")) return 1;

    CUdeviceptr p = 0;
    if (!ck(pAlloc(&p, 1ULL << 20), "cuMemAlloc")) return 1;
    printf("ptr=0x%llx\n", p);

    for (int i = 0; i < 16; i++) {
        char tag[32]; snprintf(tag, sizeof(tag), "cuMemsetD8 #%d", i);
        if (!ck(pSet8(p + (i << 16), (unsigned char)(i & 0xff), 1ULL << 16), tag)) break;
    }
    if (pSync && !ck(pSync(), "cuCtxSynchronize")) printf("sync FAIL\n");
    printf("MEMSET_ALL_OK\n");

    ck(pFree(p), "cuMemFree");
    if (pCtxD) ck(pCtxD(ctx), "cuCtxDestroy");
    printf("TEST3_DONE\n");
    return 0;
}
