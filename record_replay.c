/* record_replay.c —— 自己捕获 + 自己回放
 *
 * 背景：
 *   corex 的图捕获**不记录 kernel 节点**（capture 后 cuGraphGetNodes 数出 0 个），
 *   所以它的图回放对计算是空转；`torch.cuda.graph` 同样如此。
 *   于是"图"这条路要靠我们自己补：在**发射入口**把每一步的 kernel 记下来，
 *   回放时自己重发——不经过 torch/aten/python 的任何一层，也不依赖驱动图。
 *
 * 记什么：func / grid / block / shm / stream + 每个参数的值副本。
 *   corex 的 torch 走 libcudart 的 `cudaLaunchKernel(void **args)`，args 里每个元素
 *   是指向实参的指针，而实参的字节宽度没有任何 API 能给（cuFuncGetParamInfo 不存在）。
 *   这里用"按页边界截断的 64 字节保守复制"：多读的字节一定落在同一页内，不会踩空；
 *   多写的部分落在自己的 arena 里，驱动只会按真实宽度读前缀。
 *
 * 回放：按记录顺序逐个重发。省掉的是 torch 分发/Python/簿记，留下的是驱动的
 *   一次 kickoff（这一步在 corex 上没法再省——除非投入 §10.2 的 cmdbuf 批量那条高风险路线）。
 */

#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAXN 8192
#define ARGCAP 32
#define ARGBYTES 4096   /* 每个实参最多快照这么多字节（按 2 的幂向下探，最多到页尾） */


enum { RK_KERNEL = 0, RK_MEMCPY = 1, RK_MEMSET = 2 };

typedef struct {
    int kind;
    void *func;
    dim3 grid, block;
    size_t shm;
    void *stream;
    void **args;          /* 指向 arena 里的指针数组 */
    unsigned nargs;
    unsigned span;        /* 该节点实参快照总字节数 */
    void *aval[ARGCAP];   /* 每个实参快照块的地址 */
    unsigned alen[ARGCAP];/* 每个实参快照块的长度 */
    void *aorig[ARGCAP];  /* 实参的原始地址（重定位时要回算偏移） */
    unsigned aok;         /* bit i = 第 i 个实参的原始地址在回放时仍可写，可直接还原 */
    /* memcpy / memset */
    void *dst;
    const void *src;
    size_t nbytes;
    int mkind;
} rnode;

static rnode R[MAXN];
static unsigned nR;
static int rec_on, degraded;
static char *arena;
static unsigned arena_cap, arena_used;


static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *L;
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

/* 真实符号 */
typedef cudaError_t (*fn_launch)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
typedef cudaError_t (*fn_cpy)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*fn_set)(void *, int, size_t, cudaStream_t);
typedef cudaError_t (*fn_sync)(void);
typedef cudaError_t (*fn_ssync)(cudaStream_t);

static void *hrt;
static fn_launch r_launch;
static fn_cpy r_cpy;
static fn_set r_set;
static fn_sync r_sync;
static fn_ssync r_stream_sync;    /* cudaStreamSynchronize —— 真正的流栅栏 */
static void *hcu;                /* libcuda，用来拿 cuCtxSynchronize 这个更强的栅栏 */
static int (*r_ctx_sync)(void);

/* 真栅栏：cudart 的 cudaDeviceSynchronize 在 corex 上可能压根不排水
 *（实测每段 sync_s 恒为 0.00），所以我们另备 cuCtxSynchronize / cudaStreamSynchronize。 */
static int cr_fence(void)
{
    if (r_ctx_sync) return r_ctx_sync();
    if (r_stream_sync) return (int)r_stream_sync(0);
    if (r_sync) return (int)r_sync();
    return -1;
}

static int bind_rt(void)
{
    if (r_launch) return 1;
    if (!hrt) hrt = dlopen("libcudart.so.10.2", RTLD_NOW | RTLD_LOCAL);
    if (!hrt) hrt = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!hrt) { logline("CR_RT_MISS %s", dlerror()); return 0; }
    r_launch = (fn_launch)dlsym(hrt, "cudaLaunchKernel");
    r_cpy    = (fn_cpy)dlsym(hrt, "cudaMemcpyAsync");
    r_set    = (fn_set)dlsym(hrt, "cudaMemsetAsync");
    r_sync   = (fn_sync)dlsym(hrt, "cudaDeviceSynchronize");
    r_stream_sync = (fn_sync)dlsym(hrt, "cudaStreamSynchronize");
    if (!hcu) hcu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (hcu) r_ctx_sync = (int (*)(void))dlsym(hcu, "cuCtxSynchronize");
    logline("CR_INSTALL launch=%p cpy=%p set=%p sync=%p ssync=%p ctxs=%p", (void *)r_launch,
            (void *)r_cpy, (void *)r_set, (void *)r_sync, (void *)r_stream_sync, (void *)r_ctx_sync);
    return r_launch != NULL;
}

/* 记录 */

static int peekable(const void *p);   /* 定义见下，SIGSEGV 保护读 */

/* 一个实参到底有多少字节？cudart 不告诉我们（cuFuncGetParamInfo 在 corex 上不存在）。
 * 这里按 2 的幂向上探：只要 [p, p+s) 的**最后一个字节**可读、且不跨页，就认为 s 可用；
 * 一旦某档不可读就停在上一档。这样：
 *   - 普通标量实参（栈上 4/8 字节）会顺带把同一帧里的邻居也抄下来 -> 无害；
 *   - 按值传递的大结构（torch 的 index_copy / index_fill kernel 里那类 functor，
 *     内含 Array<char*,N> 一组数据指针，宽度远超 8 字节）能被**完整**抄下来。
 *     —— 之前固定 64 字节截断会把这类 functor 抄成半截，回放时 kernel 读到的
 *        数据指针是残值，直接野指针 -> corex 静默卡死（这就是 631 节点卡在
 *        index_copy_/index_fill_ 的根因）。 */
static unsigned cr_span(const void *p)
{
    uintptr_t a = (uintptr_t)p;
    if (!peekable(p)) return 0;
    unsigned best = 0;
    for (unsigned s = 64; s <= ARGBYTES; s <<= 1) {
        if (((a + s - 1) >> 12) != (a >> 12)) break;    /* 不跨页 */
        if (!peekable((const char *)p + s - 1)) break;
        best = s;
    }
    return best ? best : 64;
}

static rnode *push(int kind)
{
    if (nR >= MAXN) { degraded = 1; return NULL; }
    rnode *q = &R[nR];
    memset(q, 0, sizeof(*q));
    q->kind = kind;
    return q;
}

/* args[] 没有长度信息（cudart 不提供 param info），torch 的 args 数组也
 * 不保证有 NULL 哨兵。所以逐个探：能安全读一个字节才算有效参数。
 * 用 SIGSEGV 保护读来做探测，读到未映射页就停下，绝不复制野指针。 */
static __thread sigjmp_buf __jb;
static __thread volatile int __jb_on;

static void cr_segv(int sig)
{
    if (__jb_on) { __jb_on = 0; siglongjmp(__jb, 1); }
    signal(SIGSEGV, SIG_DFL);
}

static int peekable(const void *p)
{
    if (!p) return 0;
    if (sigsetjmp(__jb, 1) == 0) {
        __jb_on = 1;
        volatile char c = *(volatile const char *)p; (void)c;
        __jb_on = 0;
        return 1;
    }
    return 0;
}

/* 1 级指针重定位 
 * torch 的 index_copy / index_fill 这类 kernel，`args` 里有一个按值传的 functor 结构，
 * 结构内部又存着**指回主机栈上另一个小数组**的指针（数据指针组）。快照字节后直接回放，
 * 这些内部指针仍然指着"录制那一刻的栈"，回放时读到的是残值 -> index 值成了垃圾 ->
 * 越界写 -> corex 静默僵住。录制时逐字扫描快照，凡是"指向另一个已快照 span 的指针"，
 * 就地改写成指向 arena 里的那份副本，回放就自洽了。
 * 只重定位指向**栈区**（0x7f.. 段）的目标，避免把碰巧落在堆段里的设备指针改坏。 */
/* 主机侧还原 */

typedef struct { uintptr_t orig; unsigned len; char *neu; int stacklike; } spanrec;
static spanrec *SP;
static unsigned nSP, capSP, reloc_fixed;

/* 更狠也更准的一招：回放前把录制那一刻主机上的实参内存原样写回去，
 * 然后用原始地址发射。这样 functor 里的内部指针（指回主机堆/栈上的数据指针组）
 * 依然自洽——它们在录制时就是有效的，还原之后依然有效。
 * 之前只快照字节、用 arena 里的副本发射，那些内部指针会指到已被 torch 复用/改写的堆内存，
 * index 值就成了垃圾 -> 越界写 -> corex 静默僵住。
 * 只有当原地址已不可写（被 munmap 掉）时，才退回"arena 副本 + 指针重定位"这条路。 */
static struct { uintptr_t lo, hi; int w; } MP[4096];
static int nMP;

static void load_maps(void)
{
    FILE *f = fopen("/proc/self/maps", "r");
    nMP = 0;
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof(line), f) && nMP < 4096) {
        unsigned long lo = 0, hi = 0;
        char prot[8] = "";
        if (sscanf(line, "%lx-%lx %7s", &lo, &hi, prot) != 3) continue;
        MP[nMP].lo = lo; MP[nMP].hi = hi; MP[nMP].w = (prot[1] == 'w'); nMP++;
    }
    fclose(f);
}

static int writable_at(const void *p, unsigned len)
{
    uintptr_t a = (uintptr_t)p;
    for (int i = 0; i < nMP; i++)
        if (a >= MP[i].lo && a + len <= MP[i].hi) return MP[i].w;
    return 0;
}

/* 把快照过的实参字节写回原地址（录制时的主机内存状态） */
static void cr_restore(void)
{
    unsigned n = 0, bad = 0;
    for (unsigned e = 0; e < nSP; e++) {
        if (!writable_at((void *)SP[e].orig, SP[e].len)) { bad++; continue; }
        memcpy((char *)SP[e].orig, SP[e].neu, SP[e].len);
        n++;
    }
    logline("CR_RESTORE n=%u bad=%u", n, bad);
}

static int stacklike(uintptr_t o)
{
    return o >= 0x7f0000000000ull && o < 0x800000000000ull;
}

static int cmp_span(const void *a, const void *b)
{
    uintptr_t x = ((const spanrec *)a)->orig, y = ((const spanrec *)b)->orig;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static void add_span(void *orig, unsigned len, char *neu)
{
    if (!len) return;
    if (nSP == capSP) {
        unsigned ncap = capSP ? capSP * 2 : 256;
        spanrec *n2 = realloc(SP, ncap * sizeof(spanrec));
        if (!n2) return;
        SP = n2; capSP = ncap;
    }
    SP[nSP].orig = (uintptr_t)orig;
    SP[nSP].len = len;
    SP[nSP].neu = neu;
    SP[nSP].stacklike = stacklike((uintptr_t)orig);
    nSP++;
}

static void cr_relocate(void)
{
    if (!nSP) return;
    qsort(SP, nSP, sizeof(spanrec), cmp_span);
    unsigned fixed = 0;
    for (int pass = 0; pass < 2; pass++) {       /* 两遍：链式引用也能落到最终位置 */
        fixed = 0;
        for (unsigned e = 0; e < nSP; e++) {
            char *p = SP[e].neu;
            for (unsigned o = 0; o + 8 <= SP[e].len; o += 8) {
                uintptr_t w;
                memcpy(&w, p + o, 8);
                unsigned lo = 0, hi = nSP;
                while (lo < hi) { unsigned mid = (lo + hi) / 2; if (SP[mid].orig <= w) lo = mid + 1; else hi = mid; }
                if (!lo) continue;
                spanrec *s = &SP[lo - 1];
                if (w < s->orig + s->len) {
                    uintptr_t nw = (uintptr_t)s->neu + (w - s->orig);
                    memcpy(p + o, &nw, 8);
                    fixed++;
                }
            }
        }
    }
    logline("CR_RELOC spans=%u fixed=%u", nSP, fixed);
    reloc_fixed = fixed;
}

/* 给一个节点分配"指针数组 + 参数值副本"（每个参数按自己的可读宽度快照）；
 * 返回有效参数个数 */
static unsigned alloc_args(rnode *q, void **args)
{
    unsigned na = 0;
    while (na < ARGCAP && peekable(args[na])) na++;
    if (!na) { q->args = NULL; q->nargs = 0; q->span = 0; return 0; }

    unsigned spans[ARGCAP];
    unsigned tot = na * sizeof(void *);
    for (unsigned i = 0; i < na; i++) {
        spans[i] = cr_span(args[i]);
        tot += spans[i];
    }
    if (arena_used + tot > arena_cap) {
        unsigned ncap = arena_cap ? arena_cap * 2 : (1u << 20);
        while (ncap < arena_used + tot) ncap *= 2;
        char *na2 = realloc(arena, ncap);
        if (!na2) { degraded = 1; q->args = NULL; q->nargs = 0; q->span = 0; return 0; }
        arena = na2; arena_cap = ncap;
    }
    char *base = arena + arena_used;
    arena_used += tot;
    void **ptrs = (void **)base;
    char *vals = base + na * sizeof(void *);
    for (unsigned i = 0; i < na; i++) {
        memcpy(vals, args[i], spans[i]);
        ptrs[i] = vals;
        q->aval[i] = vals;
        q->alen[i] = spans[i];
        q->aorig[i] = (void *)args[i];
        add_span(args[i], spans[i], vals);
        vals += spans[i];
    }
    q->args = ptrs;
    q->nargs = na;
    q->span = tot;

    /* 诊断：实参里有没有"指向实参自己的栈帧"的指针？（这类 functor 内部若藏了
     * 指回栈上小数组的指针，快照字节回放时就会指向已失效的栈 -> 野指针） */
    if (getenv("CAPREC_ARGS")) {
        uintptr_t amin = ~(uintptr_t)0, amax = 0;
        for (unsigned i = 0; i < na; i++) {
            uintptr_t p = (uintptr_t)args[i];
            if (p < amin) amin = p;
            if (p + spans[i] > amax) amax = p + spans[i];
        }
        unsigned nin = 0, nstack = 0;
        for (unsigned i = 0; i < na; i++)
            for (unsigned o = 0; o + 8 <= spans[i]; o += 8) {
                uintptr_t w;
                memcpy(&w, (const char *)args[i] + o, 8);
                if (w >= amin && w < amax) nin++;
                if (w > 0x7f0000000000ull && w < 0x7fffffffffffull) nstack++;
            }
        logline("CR_ARGS n=%u amin=%p amax=%p spans=%u,%u,%u,%u inargs=%u stackwords=%u",
                na, (void *)amin, (void *)amax,
                na > 0 ? spans[0] : 0, na > 1 ? spans[1] : 0,
                na > 2 ? spans[2] : 0, na > 3 ? spans[3] : 0, nin, nstack);
    }
    return na;
}

static int in_rec(void) { return __atomic_load_n(&rec_on, __ATOMIC_RELAXED); }

cudaError_t cudaLaunchKernel(const void *f, dim3 grid, dim3 block,
                          void **args, size_t shm, cudaStream_t s)
{
    if (!bind_rt()) return cudaErrorInvalidValue;
    cudaError_t r = r_launch(f, grid, block, args, shm, s);
    if (r != 0 || !in_rec()) return r;

    /* 参数个数不知道：先做可读探测，遇到第一个不能安全读的位置就停。
     * torch 的 args 数组按精确个数分配，没有 NULL 哨兵。 */
    pthread_mutex_lock(&mu);
    rnode *q = push(RK_KERNEL);
    if (q) {
        q->func = (void *)f;
        q->grid = grid;
        q->block = block;
        q->shm = shm;
        q->stream = s;
        alloc_args(q, args);
        nR++;
        if (nR <= 4) logline("CR_REC n=%u func=%p g=%u,%u,%u b=%u,%u,%u shm=%zu nargs=%u span=%u",
                             nR, q->func, grid.x, grid.y, grid.z,
                             block.x, block.y, block.z, shm, q->nargs, q->span);
    }
    pthread_mutex_unlock(&mu);
    return r;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, enum cudaMemcpyKind kind, cudaStream_t s)
{
    if (!bind_rt()) return cudaErrorInvalidValue;
    cudaError_t r = r_cpy(dst, src, n, kind, s);
    if (r != 0 || !in_rec()) return r;
    pthread_mutex_lock(&mu);
    rnode *q = push(RK_MEMCPY);
    if (q) { q->dst = dst; q->src = src; q->nbytes = n; q->mkind = kind; q->stream = s; nR++; }
    pthread_mutex_unlock(&mu);
    return r;
}

cudaError_t cudaMemsetAsync(void *d, int v, size_t n, cudaStream_t s)
{
    if (!bind_rt()) return cudaErrorInvalidValue;
    cudaError_t r = r_set(d, v, n, s);
    if (r != 0 || !in_rec()) return r;
    pthread_mutex_lock(&mu);
    rnode *q = push(RK_MEMSET);
    if (q) { q->dst = d; q->nbytes = n; q->mkind = v; q->stream = s; nR++; }
    pthread_mutex_unlock(&mu);
    return r;
}

/* 公开 API */

/* 上一份捕获（用来做"同一段代码录两遍、逐节点比对"的忠实性判据） */
static rnode P[MAXN];
static unsigned nP;
static char *prev_arena;

int caprec_begin(void)
{
    if (!bind_rt()) return -2;
    pthread_mutex_lock(&mu);
    if (nR) {                       /* 保留上一份，连同它自己的 arena */
        memcpy(P, R, sizeof(rnode) * nR);
        nP = nR;
        prev_arena = arena;         /* 不释放，慢慢攒（每次几十 KB） */
        arena = NULL; arena_cap = 0; arena_used = 0;
    }
    nR = 0; degraded = 0; nSP = 0;
    __atomic_store_n(&rec_on, 1, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&mu);
    logline("CR_BEGIN");
    return 0;
}

/* 同一段代码录两遍：func/维度/每个实参字节必须完全一致。
 * 只要某个实参的快照字节在两次记录之间变了，说明那个实参里带着会变的
 * 主机地址（栈/host 临时对象）—— 回放时就会指向已失效内存，
 * corex 的表现就是"不报错、直接僵住"。 */
static void cr_diff(void)
{
    if (!nP) { logline("CR_DIFF none"); return; }
    if (nP != nR) logline("CR_DIFF_COUNT prev=%u now=%u", nP, nR);
    unsigned n = nP < nR ? nP : nR, bad = 0;
    for (unsigned i = 0; i < n && bad < 24; i++) {
        rnode *a = &P[i], *b = &R[i];
        if (a->func != b->func || a->nargs != b->nargs || a->shm != b->shm ||
            a->grid.x != b->grid.x || a->grid.y != b->grid.y || a->grid.z != b->grid.z ||
            a->block.x != b->block.x || a->block.y != b->block.y || a->block.z != b->block.z) {
            logline("CR_DIFF i=%u META f=%p->%p g=%u,%u,%u nargs=%u->%u shm=%zu->%zu",
                    i, a->func, b->func, a->grid.x, a->grid.y, a->grid.z,
                    a->nargs, b->nargs, a->shm, b->shm);
            bad++;
            continue;
        }
        unsigned nbad = 0, kbad = 0;
        for (unsigned k = 0; k < a->nargs; k++) {
            unsigned L = a->alen[k] < b->alen[k] ? a->alen[k] : b->alen[k];
            if (!a->aval[k] || !b->aval[k]) { nbad++; continue; }
            if (memcmp(a->aval[k], b->aval[k], L)) { nbad++; if (!kbad) kbad = k + 1; }
        }
        if (nbad) {
            logline("CR_DIFF i=%u ARG f=%p nargs=%u bad=%u first_arg=%u len=%u,%u",
                    i, b->func, b->nargs, nbad, kbad - 1,
                    kbad ? a->alen[kbad - 1] : 0, kbad ? b->alen[kbad - 1] : 0);
            bad++;
        }
    }
    logline("CR_DIFF_TOTAL bad=%u of=%u", bad, n);
}

int caprec_end(unsigned *nodes, unsigned *bytes)
{
    __atomic_store_n(&rec_on, 0, __ATOMIC_RELAXED);
    if (nodes) *nodes = nR;
    if (bytes) *bytes = arena_used;
    logline("CR_END nodes=%u arena=%u degraded=%d", nR, arena_used, degraded);
    /* 记下哪些实参的原始地址在回放时还能写（能写就直接还原，不能写才退回 arena 副本） */
    load_maps();
    { unsigned ok = 0, tot = 0;
    for (unsigned i = 0; i < nR; i++) {
        rnode *q = &R[i];
        q->aok = 0;
        for (unsigned k = 0; k < q->nargs; k++) {
            tot++;
            if (q->alen[k] && writable_at(q->aorig[k], q->alen[k])) { q->aok |= 1u << k; ok++; }
        }
    }
    logline("CR_AOK ok=%u tot=%u", ok, tot);
    }
    if (getenv("CAPREC_DIFF")) cr_diff();
    {
        const char *rl = getenv("CAPREC_RELOC");
        if (!(rl && rl[0] == '0')) cr_relocate();   /* 默认开：把指回主机快照内部的指针改指 arena */
        else reloc_fixed = 0;
    }
    if (getenv("CAPREC_DUMP")) {
        FILE *f = fopen("/proc/self/maps", "r");
        for (unsigned i = 0; i < nR; i++) {
            rnode *q = &R[i];
            double a0 = 0.0, a1 = 0.0;
            if (q->args && q->nargs > 0) memcpy(&a0, q->args[0], 8);
            if (q->args && q->nargs > 1) memcpy(&a1, q->args[1], 8);
            logline("CR_DUMP i=%u k=%d f=%p g=%u,%u,%u b=%u,%u,%u nargs=%u shm=%zu st=%p a0=%g a1=%g",
                    i, q->kind, q->func,
                    q->grid.x, q->grid.y, q->grid.z, q->block.x, q->block.y, q->block.z,
                    q->nargs, q->shm, q->stream, a0, a1);
            /* 图优化用：整节点指纹（func/网格/块/shm/每个实参的全部字节）。
             * 两条指纹相同 = 同一 kernel、同样的参数**字节**，也就是"重复的同一件事"，
             * 离线就能拿它做公共子表达式 / 冗余写合并的第一道筛子。 */
            unsigned long long h = 1469598103934665603ull;
            #define CR_FNV(p, n) do { const unsigned char *b_ = (const unsigned char *)(p); \
                for (size_t i_ = 0; i_ < (size_t)(n); i_++) { h ^= b_[i_]; h *= 1099511628211ull; } } while (0)
            CR_FNV(&q->func, sizeof(q->func));
            unsigned dims[6] = { q->grid.x, q->grid.y, q->grid.z, q->block.x, q->block.y, q->block.z };
            CR_FNV(dims, sizeof(dims));
            CR_FNV(&q->shm, sizeof(q->shm));
            CR_FNV(&q->nargs, sizeof(q->nargs));
            for (unsigned k = 0; k < q->nargs; k++) {
                CR_FNV(&q->alen[k], sizeof(q->alen[k]));
                if (q->aval[k]) CR_FNV(q->aval[k], q->alen[k]);
            }
            #undef CR_FNV
            logline("CR_HASH i=%u k=%d h=%016llx nargs=%u", i, q->kind, h, q->nargs);
            /* 精确指针：图优化要知道每个节点碰了哪些缓冲区，%g 会把指针压成 6 位有效数字 */
            {
                char pb[1600];
                int off = 0;
                off += snprintf(pb + off, sizeof(pb) - (size_t)off, "CR_PTR i=%u k=%d nargs=%u", i, q->kind, q->nargs);
                for (unsigned k = 0; k < q->nargs && k < 24 && off < (int)sizeof(pb) - 40; k++) {
                    uintptr_t v = 0;
                    if (q->aval[k] && q->alen[k] >= 8) memcpy(&v, q->aval[k], 8);
                    off += snprintf(pb + off, sizeof(pb) - (size_t)off,
                                    " p%u=%p/%u", k, (void *)v, q->alen[k]);
                }
                logline("%s", pb);
            }
        }
        /* 把每个 func 落到哪个 .so + 偏移，方便离线 nm/addr2line 认 kernel */
        if (f) {
            unsigned prev = 0;
            for (unsigned i = 0; i < nR; i++) {
                rnode *q = &R[i];
                if (q->kind != RK_KERNEL) continue;
                int seen = 0;
                for (unsigned j = 0; j < i; j++)
                    if (R[j].kind == RK_KERNEL && R[j].func == q->func) { seen = 1; break; }
                if (seen) { prev++; continue; }
                rewind(f);
                unsigned long lo = 0, hi = 0, off = 0; char path[256] = "";
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    if (sscanf(line, "%lx-%lx %*s %lx %*s %*s %255s", &lo, &hi, &off, path) >= 3) {
                        unsigned long a = (unsigned long)q->func;
                        if (a >= lo && a < hi) break;
                    }
                    path[0] = 0;
                }
                logline("CR_MAP f=%p %s +0x%lx", q->func, path[0] ? path : "?", 
                        (unsigned long)q->func - lo + off);
            }
        }
    }
    return degraded ? -1 : 0;
}

int caprec_replay_range(unsigned from, unsigned to);
int caprec_replay(void)
{
    const char *nr = getenv("CAPREC_NOREPLAY");
    if (nr && nr[0] == '1') { logline("CR_RP_SKIPPED n=%u", nR); return 0; }
    return caprec_replay_range(0, nR);
}

int caprec_replay_range(unsigned from, unsigned to)
{
    if (!bind_rt()) return -2;
    const char *tr = getenv("CAPREC_TRACE");
    const char *se = getenv("CAPREC_SYNC_EACH");
    const char *ck = getenv("CAPREC_CHUNK");
    unsigned chunk = (ck && *ck) ? (unsigned)atoi(ck) : 0;
    const char *rst = getenv("CAPREC_RESTORE");
    int do_restore = (rst && rst[0] == '1');        /* 默认关：还原会踩坏当场活的主机内存 */
    if (to > nR) to = nR;
    if (do_restore) { load_maps(); cr_restore(); }   /* 先把主机侧实参内存还原成录制那一刻 */
    for (unsigned i = from; i < to; i++) {
        rnode *q = &R[i];
        cudaError_t r;
        if (tr)
            logline("CR_RP i=%u k=%d f=%p g=%u,%u,%u b=%u,%u,%u nargs=%u shm=%zu",
                    i, q->kind, q->func,
                    q->grid.x, q->grid.y, q->grid.z, q->block.x, q->block.y, q->block.z,
                    q->nargs, q->shm);
        switch (q->kind) {
        case RK_KERNEL: {
            void *targs[ARGCAP];
            void **ap = q->args;
            if (q->nargs) {
                for (unsigned k = 0; k < q->nargs; k++)
                    targs[k] = (do_restore && ((q->aok >> k) & 1)) ? q->aorig[k] : q->aval[k];
                ap = targs;
            }
            r = r_launch(q->func, q->grid, q->block, ap, q->shm, q->stream);
            break;
        }
        case RK_MEMCPY:
            r = r_cpy(q->dst, q->src, q->nbytes, q->mkind, q->stream);
            break;
        default:
            r = r_set(q->dst, q->mkind, q->nbytes, q->stream);
            break;
        }
        if (se && se[0] == '1') {
            double t0 = (double)clock() / CLOCKS_PER_SEC;
            int rs = cr_fence();
            logline("CR_RP_SYNC i=%u rc=%d src=%d ms=%.3f", i, rs, r,
                    ((double)clock() / CLOCKS_PER_SEC - t0) * 1e3);
        }
        /* 分段回放：每 chunk 个节点**真栅栏**一次，把"设备停在哪一段"直接写成时间戳 */
        if (chunk && ((i + 1) % chunk == 0 || i + 1 == to)) {
            double t0 = (double)clock() / CLOCKS_PER_SEC;
            int rs = cr_fence();
            double dt = (double)clock() / CLOCKS_PER_SEC - t0;
            logline("CR_CHUNK upto=%u sync_s=%.4f fence_rc=%d", i + 1, dt, rs);
        }
        if (r != 0) {
            logline("CR_REPLAY_FAIL i=%u kind=%d rc=%d", i, q->kind, r);
            return -1;
        }
    }
    return 0;
}

void caprec_info(char *buf, int n)
{
    unsigned k = 0, c = 0, m = 0, noarg = 0;
    for (unsigned i = 0; i < nR; i++) {
        if (R[i].kind == RK_KERNEL) { k++; if (!R[i].nargs) noarg++; }
        else if (R[i].kind == RK_MEMCPY) c++;
        else m++;
    }
    snprintf(buf, n, "nodes=%u kernel=%u memcpy=%u memset=%u noarg=%u arena=%u degraded=%d reloc=%u",
             nR, k, c, m, noarg, arena_used, degraded, reloc_fixed);
}

/* 自建图
 * 目的：把"录下来的节点"直接拼成一张驱动图，一次 cuGraphLaunch。
 * 这里用已经录到的 CUfunction（torch 的 kernel），不需要自己编模块。
 * 只加 kernel 节点；memcpy/memset 节点先跳过（数量会打出来）。 */
typedef CUresult (*fn_gcreate)(CUgraph *, unsigned);
typedef CUresult (*fn_gaddk)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
                             const CUDA_KERNEL_NODE_PARAMS *);
typedef CUresult (*fn_ginst)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned);
typedef CUresult (*fn_glaunch)(CUgraphExec, CUstream);
typedef CUresult (*fn_gdestroy)(CUgraph);
typedef CUresult (*fn_gedestroy)(CUgraphExec);

static fn_gcreate  f_gcreate;
static fn_gaddk     f_gaddk;
static fn_ginst     f_ginst;
static fn_glaunch   f_glaunch;
static fn_gdestroy  f_gdestroy;
static fn_gedestroy f_gedestroy;
static CUgraph gG;
static CUgraphExec gGE;
static unsigned g_skipped;

static int bind_graph(void)
{
    if (f_gaddk) return 1;
    if (!hcu) hcu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!hcu) { logline("CR_GINSTALL no libcuda"); return 0; }
    f_gcreate  = (fn_gcreate)dlsym(hcu, "cuGraphCreate");
    f_gaddk    = (fn_gaddk)dlsym(hcu, "cuGraphAddKernelNode");
    f_ginst    = (fn_ginst)dlsym(hcu, "cuGraphInstantiate");
    f_glaunch  = (fn_glaunch)dlsym(hcu, "cuGraphLaunch");
    f_gdestroy = (fn_gdestroy)dlsym(hcu, "cuGraphDestroy");
    f_gedestroy = (fn_gedestroy)dlsym(hcu, "cuGraphExecDestroy");
    logline("CR_GINSTALL create=%p addk=%p inst=%p launch=%p", (void *)f_gcreate,
            (void *)f_gaddk, (void *)f_ginst, (void *)f_glaunch);
    return f_gaddk && f_ginst && f_glaunch;
}

int caprec_graph_build(unsigned *nodes_out)
{
    if (!nR) return -2;
    if (!bind_graph()) return -3;
    logline("CR_GSTEP create enter");
    CUresult rc = f_gcreate(&gG, 0);
    logline("CR_GSTEP create rc=%d g=%p", rc, (void *)gG);
    if (rc) { logline("CR_GFAIL create rc=%d", rc); return -4; }
    CUgraphNode prev = NULL;
    unsigned added = 0;
    g_skipped = 0;
    for (unsigned i = 0; i < nR; i++) {
        rnode *q = &R[i];
        if (q->kind != RK_KERNEL || !q->args) { g_skipped++; continue; }
        CUDA_KERNEL_NODE_PARAMS kp;
        memset(&kp, 0, sizeof(kp));
        kp.func = (CUfunction)q->func;
        kp.gridDimX = q->grid.x; kp.gridDimY = q->grid.y; kp.gridDimZ = q->grid.z;
        kp.blockDimX = q->block.x; kp.blockDimY = q->block.y; kp.blockDimZ = q->block.z;
        kp.sharedMemBytes = (unsigned)q->shm;
        kp.kernelParams = (void **)q->args;
        kp.extra = NULL;
        CUgraphNode nd = NULL;
        if (added < 4) logline("CR_GSTEP add i=%u enter", i);
        rc = f_gaddk(&nd, gG, prev ? &prev : NULL, prev ? 1 : 0, &kp);
        if (added < 4) logline("CR_GSTEP add i=%u rc=%d", i, rc);
        if (added < 4 || rc)
            logline("CR_GADD i=%u rc=%d node=%p f=%p g=%u,%u,%u", i, rc, (void *)nd,
                    q->func, q->grid.x, q->grid.y, q->grid.z);
        if (rc) { logline("CR_GFAIL add i=%u rc=%d", i, rc); return -5; }
        prev = nd;
        added++;
    }
    rc = f_ginst(&gGE, gG, NULL, NULL, 0);
    logline("CR_GINST nodes=%u skipped=%u rc=%d exec=%p", added, g_skipped, rc, (void *)gGE);
    if (nodes_out) *nodes_out = added;
    return rc ? -6 : 0;
}

int caprec_graph_launch(void)
{
    if (!gGE) return -2;
    return (int)f_glaunch(gGE, 0);
}

void caprec_graph_free(void)
{
    if (gGE && f_gedestroy) { f_gedestroy(gGE); gGE = NULL; }
    if (gG && f_gdestroy) { f_gdestroy(gG); gG = NULL; }
}

__attribute__((constructor)) static void cr_init(void)
{
    const char *p = getenv("CAPREC_LOG");
    L = fopen(p ? p : "/tmp/record_replay.log", "a");
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = cr_segv;
    sa.sa_flags = SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
}
