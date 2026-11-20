/* test_mem.c —— 最小连通性检查：memset / HtoD / DtoH 是否真的落数据。
 * 不做 capture，只验证最底层三个操作 + 回读。
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
typedef unsigned long long CUdeviceptr;

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen FAIL\n"); return 2; }

    CUresult (*pInit)(int) = dlsym(RTLD_DEFAULT, "cuInit");
    CUresult (*pDev)(CUdevice *, int) = dlsym(RTLD_DEFAULT, "cuDeviceGet");
    CUresult (*pCtx)(CUcontext *, unsigned, CUdevice) = dlsym(RTLD_DEFAULT, "cuCtxCreate_v2");
    if (!pCtx) pCtx = (CUresult (*)(CUcontext *, unsigned, CUdevice))dlsym(RTLD_DEFAULT, "cuCtxCreate");
    CUresult (*pStream)(CUstream *, unsigned) = dlsym(RTLD_DEFAULT, "cuStreamCreate");
    CUresult (*pAlloc)(CUdeviceptr *, unsigned long long) = dlsym(RTLD_DEFAULT, "cuMemAlloc_v2");
    CUresult (*pSet)(CUdeviceptr, unsigned, unsigned long long, CUstream) =
        dlsym(RTLD_DEFAULT, "cuMemsetD32Async");
    CUresult (*pSetSync)(CUdeviceptr, unsigned, unsigned long long) =
        dlsym(RTLD_DEFAULT, "cuMemsetD32_v2");
    if (!pSetSync) pSetSync = (CUresult (*)(CUdeviceptr, unsigned, unsigned long long))
        dlsym(RTLD_DEFAULT, "cuMemsetD32");
    CUresult (*pHtoD)(CUdeviceptr, const void *, unsigned long long, CUstream) =
        dlsym(RTLD_DEFAULT, "cuMemcpyHtoDAsync_v2");
    CUresult (*pDtoH)(void *, CUdeviceptr, unsigned long long, CUstream) =
        dlsym(RTLD_DEFAULT, "cuMemcpyDtoHAsync_v2");
    CUresult (*pDtoHSync)(void *, CUdeviceptr, unsigned long long) =
        dlsym(RTLD_DEFAULT, "cuMemcpyDtoH_v2");
    CUresult (*pSync)(CUstream) = dlsym(RTLD_DEFAULT, "cuStreamSynchronize");

    printf("sym set=%d setSync=%d htod=%d dtoh=%d dtohsync=%d\n",
           !!pSet, !!pSetSync, !!pHtoD, !!pDtoH, !!pDtoHSync);
    if (!pInit || !pDev || !pCtx || !pStream || !pAlloc || !pSync) { printf("dlsym FAIL\n"); return 2; }

    printf("init=%d\n", pInit(0));
    CUdevice dev = 0;
    printf("dev=%d\n", pDev(&dev, 0));
    CUcontext ctx = 0;
    printf("ctx=%d\n", pCtx(&ctx, 0, dev));

    CUstream s = 0;
    printf("stream=%d\n", pStream(&s, 0));
    CUdeviceptr buf = 0;
    printf("alloc=%d\n", pAlloc(&buf, 1ULL << 20));

    int back[64];

    if (pSet) {
        printf("async memset rc=%d\n", pSet(buf, 0xa5a5a5a5u, 64, s));
        printf("sync rc=%d\n", pSync(s));
    }
    memset(back, 0, sizeof(back));
    if (pDtoH) {
        printf("async dtoh rc=%d\n", pDtoH(back, buf, 256, s));
        printf("sync2 rc=%d\n", pSync(s));
    }
    printf("AFTER_MEMSET back0=0x%08x back1=0x%08x back7=0x%08x\n",
           back[0], back[1], back[7]);

    if (pSetSync) {
        printf("sync memset rc=%d\n", pSetSync(buf, 0x5a5a5a5au, 64));
    }
    memset(back, 0, sizeof(back));
    if (pDtoHSync) {
        printf("sync dtoh rc=%d\n", pDtoHSync(back, buf, 256));
    } else if (pDtoH) {
        pDtoH(back, buf, 256, s); pSync(s);
    }
    printf("AFTER_SYNCMEMSET back0=0x%08x back7=0x%08x\n", back[0], back[7]);

    int host[64];
    for (int i = 0; i < 64; i++) host[i] = 0x11223300 + i;
    memset(back, 0, sizeof(back));
    if (pHtoD) {
        printf("htod rc=%d\n", pHtoD(buf, host, 256, s));
        pSync(s);
    }
    if (pDtoH) pDtoH(back, buf, 256, s);
    pSync(s);
    printf("AFTER_HTOD back0=0x%08x back1=0x%08x back7=0x%08x\n", back[0], back[1], back[7]);

    printf("TEST_MEM_DONE\n");
    return 0;
}
