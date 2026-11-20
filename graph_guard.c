/* graph_guard.c 
 * graph 的护栏层。
 *
 * libcuda 反汇编可得：
 *   1) 命令层序号/描述符索引写死 7 位，描述符表在 0x3bd377 按 desc+8 = 0x2000 分配，
 *      0x3bafd0 用 `idx<<6 + 表基` 寻址且无边界检查可得表深 = 128。
 *   2) 块容量由 0x435410 从 *(席位+0x54) 取，原厂 8192 = 128 × 64，余量为零；
 *      0x4355f0 判据严格小于（4355fe setb / 435601 jb）。
 *   3) graph 路径从不调用 0x4355f0（只在 0x3bdcxx 家族 4 处），超限时直接写穿块。
 *
 * 挂点选择：
 *   用 cuGraphInstantiate 0x109280：前 9 字节 = push r15 / push r14 /
 *   mov %r8,%r14 / push r13（2+2+3+2），全部位置无关，可整段复制。
 *   进入时 rsi = CUgraph（0x109290 处 `mov %rsi,%rbp` 即把 graph 存起来），
 *   与 cuda.h 里 cuGraphInstantiate(phGraphExec, hGraph, ...) 的 rsi 一致。
 *
 * 超限时返回 CUDA_ERROR_INVALID_VALUE(0x1)，不调用原函数；
 * 图不会进入 exec，也就永远不会进 replay，因此不会写穿块。
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef int CUresult;
typedef void *CUgraph;
typedef void *CUgraphNode;

#define CUDA_ERROR_INVALID_VALUE 0x1

#define VMA_INSTANTIATE  0x109280UL
#define LEN_INSTANTIATE  9
#define TAIL_INSTANTIATE 0x109289UL

static uint64_t mod_base;
static FILE *L;
static unsigned max_nodes = 128;
static unsigned long nreject, npass;

static void logline(const char *fmt, ...)
{
    if (!L) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(L, fmt, ap);
    va_end(ap);
    fputc('\n', L);
    fflush(L);
}

static int ph_cb(struct dl_phdr_info *info, size_t sz, void *data)
{
    (void)sz; (void)data;
    const char *nm = info->dlpi_name ? info->dlpi_name : "";
    if (nm && strstr(nm, "libcuda.so")) { mod_base = info->dlpi_addr; return 1; }
    return 0;
}

static uint8_t *emit_jmp(uint8_t *p, uint64_t to)
{
    int32_t rel = (int32_t)(to - (uint64_t)(uintptr_t)(p + 5));
    *p++ = 0xe9;
    memcpy(p, &rel, 4);
    return p + 4;
}

static void *make_tramp(const uint8_t *src, size_t len, uint64_t tail)
{
    uint8_t *t = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t == MAP_FAILED) return NULL;
    memcpy(t, src, len);
    emit_jmp(t + len, tail);
    __builtin___clear_cache((char *)t, (char *)t + len + 5);
    return t;
}

static int patch_site(uint64_t vma, size_t len, void *target)
{
    uint8_t *p = (uint8_t *)((uintptr_t)mod_base + vma);
    if (mprotect((void *)((uintptr_t)p & ~0xfffUL), 8192,
                 PROT_READ | PROT_WRITE | PROT_EXEC) != 0) return 0;
    emit_jmp(p, (uint64_t)(uintptr_t)target);
    for (size_t i = 5; i < len; i++) p[i] = 0x90;
    __builtin___clear_cache((char *)p, (char *)p + len);
    return 1;
}

static unsigned graph_nodes(CUgraph g)
{
    typedef CUresult (*fn_t)(CUgraph, CUgraphNode *, unsigned *);
    fn_t get = (fn_t)dlsym(RTLD_DEFAULT, "cuGraphGetNodes");
    unsigned n = 0;
    if (!get || !g) return 0;
    if (get(g, NULL, &n) != 0) return 0;
    return n;
}

/* 返回 1 = 拦截（超限），0 = 放行 */
static int ck_inst(CUgraph g)
{
    unsigned n = graph_nodes(g);
    if (n <= max_nodes) { npass++; return 0; }
    nreject++;
    logline("GRAPH_REJECT where=Instantiate nodes=%u max=%u", n, max_nodes);
    return 1;
}

/* 手写机器码桩，避免 -fPIC 下的 PC32 重定位问题：
 *   push rax/rcx/rdx/rsi/rdi/r8-r11   (9 条，入口 rsp≡8 → ≡0，可直接 call)
 *   mov %rsi,%rdi                    ; rsi = CUgraph
 *   call ck_inst
 *   pop  反向还原
 *   test %eax,%eax
 *   jne  reject
 *   movabs $tramp,%rax ; jmp *%rax   ; 尾跳进原函数 +9
 * reject:
 *   mov $1,%eax ; ret                ; 返回 CUDA_ERROR_INVALID_VALUE
 */
static void *mk_inst_stub(uint64_t tramp)
{
    uint8_t *t = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t == MAP_FAILED) return NULL;
    uint8_t *p = t;
    *p++ = 0x50; *p++ = 0x51; *p++ = 0x52; *p++ = 0x56; *p++ = 0x57;
    *p++ = 0x41; *p++ = 0x50;
    *p++ = 0x41; *p++ = 0x51;
    *p++ = 0x41; *p++ = 0x52;
    *p++ = 0x41; *p++ = 0x53;
    *p++ = 0x48; *p++ = 0x89; *p++ = 0xf7;                     /* mov %rsi,%rdi */
    {
        int32_t rel = (int32_t)((uint64_t)(uintptr_t)ck_inst - (uint64_t)(uintptr_t)(p + 5));
        *p++ = 0xe8; memcpy(p, &rel, 4); p += 4;
    }
    *p++ = 0x41; *p++ = 0x5b;
    *p++ = 0x41; *p++ = 0x5a;
    *p++ = 0x41; *p++ = 0x59;
    *p++ = 0x41; *p++ = 0x58;
    *p++ = 0x5f; *p++ = 0x5e; *p++ = 0x5a; *p++ = 0x59; *p++ = 0x58;

    *p++ = 0x85; *p++ = 0xc0;                                  /* test %eax,%eax */
    uint8_t *jne_at = p;
    *p++ = 0x75; *p++ = 0x00;                                  /* jne rel8，占位 */

    *p++ = 0x48; *p++ = 0xb8;                                  /* movabs $imm64,%rax */
    memcpy(p, &tramp, 8); p += 8;
    *p++ = 0xff; *p++ = 0xe0;                                  /* jmp *%rax */

    uint8_t *reject = p;
    jne_at[1] = (uint8_t)(reject - (jne_at + 2));
    *p++ = 0xb8; *p++ = 0x01; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;  /* mov $1,%eax */
    *p++ = 0xc3;                                                     /* ret */

    __builtin___clear_cache((char *)t, (char *)p);
    return t;
}

static int installed;

static int try_install(void)
{
    if (installed) return 1;
    mod_base = 0;
    dl_iterate_phdr(ph_cb, NULL);
    if (!mod_base) return 0;         /* libcuda 还没加载 */

    uint8_t *p = (uint8_t *)((uintptr_t)mod_base + VMA_INSTANTIATE);
    void *tramp = make_tramp(p, LEN_INSTANTIATE,
                            (uint64_t)(uintptr_t)(mod_base + TAIL_INSTANTIATE));
    if (!tramp) { logline("GUARD_TRAMP_FAIL"); return 0; }
    void *stub = mk_inst_stub((uint64_t)(uintptr_t)tramp);
    if (!stub) { logline("GUARD_STUB_FAIL"); return 0; }
    if (!patch_site(VMA_INSTANTIATE, LEN_INSTANTIATE, stub)) {
        logline("GUARD_PATCH_FAIL"); return 0;
    }
    installed = 1;
    logline("GUARD_INSTALL base=0x%llx max_nodes=%u tramp=0x%llx",
            (unsigned long long)mod_base, max_nodes,
            (unsigned long long)(uintptr_t)tramp);
    return 1;
}

__attribute__((constructor))
static void ctor(void)
{
    const char *p = getenv("CMDBUF_GLOG");
    L = fopen(p ? p : "/tmp/graph_guard.log", "a");
    const char *m = getenv("CMDBUF_MAXNODES");
    if (m) { unsigned v = (unsigned)strtoul(m, NULL, 0); if (v) max_nodes = v; }
    try_install();
}

/* libcuda 一定在某个 ioctl 之后才真正可用；借 ioctl 做懒装配。 */
int ioctl(int fd, unsigned long request, ...)
{
    va_list ap;
    void *arg;
    va_start(ap, request);
    arg = va_arg(ap, void *);
    va_end(ap);

    if (!installed) try_install();
    return (int)syscall(SYS_ioctl, fd, request, arg);
}
