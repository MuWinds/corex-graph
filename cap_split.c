/* cap_split.c 
 * "自动切段"层
 *
 * 背景：
 *   1) capture 是用户态 C++ 实现，逻辑全在 libcuda；thunk(libixthunk.so) 只转发
 *      ioctl，KMD 只收 16 字节 INDIRECT 包，都不看图语义 → 只能在用户态 API
 *      边界动手，驱动内部才永远写不满那张 128 项的 0x2000 描述符表。
 *   2) 驱动允许在同一 stream 上"捕获 → End → Instantiate → 再捕获"
 *   3) 坑：厂商的 cuGraphGetNodes 会破坏调用方寄存器里存活的值
 *      所以本模块默认不调用它，节点数用自己的计数；任何驱动返回值都先落到内存再读。
 *
 * 语义：
 *   Begin   → 建会话（stream / mode / 属主线程 / 计数）
 *   记录 N 个节点（N 默认 118）→ End + 存图 + 重新 Begin
 *   End     → 密封最后一段，把第 0 段那张真图交给应用，并把第 1..n-1 段建 exec
 *   Instantiate(graph0) → 让驱动照常建 exec，同时把 graph→exec 记进序列
 *   Launch  → exec 命中序列就按顺序依次 launch 每一段真 exec
 *   Destroy → 连带销毁所有段
 *
 * 不造假句柄：应用看到的是驱动自己的图/exec，我们只维护"这些段属于同一次捕获"。
 * 只支持单流顺序捕获；嵌套捕获打 SPLIT_UNSUPPORTED 并原样放行。
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef int CUresult;
typedef void *CUstream;
typedef void *CUgraph;
typedef void *CUgraphExec;
typedef void *CUgraphNode;
typedef void *CUfunction;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

#define CUDA_ERROR_INVALID_VALUE 0x1

#define MAXC   64
#define MAXSEQ 16

typedef struct {
    CUgraph g;
    CUgraphExec e;
    unsigned nodes;
} chunk_t;

typedef struct {
    int used;
    CUgraph graph;      /* 应用看到的图 = 第 0 段 */
    CUgraphExec exec;   /* 应用为第 0 段建的 exec */
    unsigned n;
    chunk_t ch[MAXC];
} seq_t;

static seq_t SQ[MAXSEQ];

static struct {
    int active;
    int broken;
    CUstream st;
    unsigned mode;
    pthread_t owner;
    unsigned cnt;
    unsigned limit;
    unsigned nchunk;
    chunk_t ch[MAXC];
} S;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *L;
static void *hcu;

static unsigned long nrec, nseal, nlaunch, nbind;

static int opt_use_nodes, opt_trace, opt_keep;

static CUresult (*r_begin)(CUstream, int);
static CUresult (*r_end)(CUstream, CUgraph *);
static CUresult (*r_inst)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned);
static CUresult (*r_launch)(CUgraphExec, CUstream);
static CUresult (*r_sync)(CUstream);
static CUresult (*r_edestroy)(CUgraphExec);
static CUresult (*r_gdestroy)(CUgraph);
static CUresult (*r_getnodes)(CUgraph, CUgraphNode *, unsigned *);
static CUresult (*r_memset_d32a)(CUdeviceptr, unsigned, size_t, CUstream);
static CUresult (*r_memset_d8a)(CUdeviceptr, unsigned char, size_t, CUstream);
static CUresult (*r_ev_record)(CUevent, CUstream);
static CUresult (*r_wait_ev)(CUstream, CUevent, unsigned);
static CUresult (*r_launch_kernel)(CUfunction, unsigned, unsigned, unsigned,
                                   unsigned, unsigned, unsigned, unsigned,
                                   CUstream, void **, void **);
static CUresult (*r_cpy_htod)(CUdeviceptr, const void *, size_t, CUstream);
static CUresult (*r_cpy_dtoh)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (*r_cpy_dtod)(CUdeviceptr, CUdeviceptr, size_t, CUstream);

/* 驱动 API 还是 runtime API 两层的语义完全一样（Begin/End/Instantiate/Launch 只是同一套驱动入口的包装），
 * 所以会话状态只存一份，用 cb_* 指向"这一层"的真函数。runtime 层必须单独影子，
 * 因为 libcudart 走的是自己的驱动入口分派表，不经过 PLT，LD_PRELOAD 影子不到 */
static int rt_on;
static int cur_rt;                  /* 当前绑定的层：0=驱动 API，1=runtime API */
static CUresult (*cb_begin)(CUstream, int);
static CUresult (*cb_end)(CUstream, CUgraph *);
static CUresult (*cb_inst)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned);
static CUresult (*cb_launch)(CUgraphExec, CUstream);
static CUresult (*cb_sync)(CUstream);
static CUresult (*cb_ed)(CUgraphExec);
static CUresult (*cb_gd)(CUgraph);

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

static void *real_sym(const char *a, const char *b)
{
    void *p = hcu ? dlsym(hcu, a) : NULL;
    if (!p && hcu && b) p = dlsym(hcu, b);
    return p;
}

static seq_t *find_seq_graph(CUgraph g)
{
    for (int i = 0; i < MAXSEQ; i++)
        if (SQ[i].used && SQ[i].graph == g) return &SQ[i];
    return NULL;
}

static seq_t *find_seq_exec(CUgraphExec e)
{
    for (int i = 0; i < MAXSEQ; i++)
        if (SQ[i].used && SQ[i].exec == e) return &SQ[i];
    return NULL;
}

static void bind_drv(void)
{
    cur_rt = 0;
    cb_begin = r_begin; cb_end = r_end; cb_inst = r_inst;
    cb_launch = r_launch; cb_sync = r_sync;
    cb_ed = r_edestroy; cb_gd = r_gdestroy;
}

/* 密封当前段（已持锁）。返回 0 = 正常。 */
static int do_seal(CUstream st)
{
    CUgraph g = NULL;
    if (!cb_end) return -1;
    CUresult r = cb_end(st, &g);
    if (r != 0 || !g) {
        logline("SPLIT_END_FAIL rc=%d g=%p", r, (void *)g);
        S.broken = 1;
        return -1;
    }
    if (S.nchunk >= MAXC) { logline("SPLIT_TOO_MANY_CHUNKS"); S.broken = 1; return -1; }

    int idx = (int)S.nchunk;
    /* 图指针立刻落到内存：厂商函数会破坏寄存器，绝不让它跨调用留在寄存器里 */
    S.ch[idx].g = g;
    S.ch[idx].e = NULL;

    unsigned n = S.cnt;      /* 每个可记录调用恰好 1 个节点 */
    if (opt_use_nodes && r_getnodes) {
        unsigned gn = 0;
        if (r_getnodes(S.ch[idx].g, NULL, &gn) == 0) n = gn;
    }
    S.ch[idx].nodes = n;

    nseal++;
    logline("SPLIT_SEAL idx=%d nodes=%u counted=%u g=%p",
            idx, n, S.cnt, (void *)S.ch[idx].g);
    S.cnt = 0;
    S.nchunk++;
    return 0;
}

static void note_record(CUstream s)
{
    if (!S.active) return;
    /* runtime 进程里驱动层的记账一律不算，否则同一批调用会被数两遍 */
    if (rt_on && !cur_rt) return;

    pthread_mutex_lock(&mu);
    if (!S.active || S.st != s || !pthread_equal(S.owner, pthread_self())) {
        pthread_mutex_unlock(&mu);
        return;
    }
    S.cnt++;
    nrec++;
    if (!S.broken && S.cnt >= S.limit) {
        if (do_seal(s) == 0) {
            CUresult rb = cb_begin ? cb_begin(s, (int)S.mode) : -1;
            if (rb != 0) {
                logline("SPLIT_REBEGIN_FAIL rc=%d", rb);
                S.broken = 1;
            } else {
                logline("SPLIT_CHUNK_START idx=%u", S.nchunk);
            }
        }
    }
    pthread_mutex_unlock(&mu);
}

static int inited;
static int install_core(void);

static void ensure(void)
{
    if (!inited) install_core();
}

static int install_core(void)
{
    if (inited) return 1;

    hcu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!hcu) { logline("SPLIT_NO_LIBCUDA"); return 0; }

    r_begin        = real_sym("cuStreamBeginCapture_v2", "cuStreamBeginCapture");
    r_end          = real_sym("cuStreamEndCapture", NULL);
    r_inst         = (CUresult (*)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned))
                     real_sym("cuGraphInstantiate_v2", "cuGraphInstantiate");
    r_launch       = real_sym("cuGraphLaunch", NULL);
    r_sync         = real_sym("cuStreamSynchronize", NULL);
    r_edestroy     = real_sym("cuGraphExecDestroy", NULL);
    r_gdestroy     = real_sym("cuGraphDestroy", NULL);
    r_getnodes     = real_sym("cuGraphGetNodes", NULL);
    r_memset_d32a  = real_sym("cuMemsetD32Async", NULL);
    r_memset_d8a   = real_sym("cuMemsetD8Async", NULL);
    r_ev_record    = real_sym("cuEventRecord", NULL);
    r_wait_ev      = real_sym("cuStreamWaitEvent", NULL);
    r_launch_kernel= real_sym("cuLaunchKernel", NULL);
    r_cpy_htod     = real_sym("cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    r_cpy_dtoh     = real_sym("cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
    r_cpy_dtod     = real_sym("cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync");

    if (!r_begin || !r_end) { logline("SPLIT_SYM_MISS"); return 0; }

    inited = 1;
    logline("SPLIT_INSTALL limit=%u begin=%p end=%p inst=%p",
            S.limit, (void *)r_begin, (void *)r_end, (void *)r_inst);
    return 1;
}

/* 会话开启（真 Begin 已成功之后调用）。 */
static void sess_open(CUstream s, int mode)
{
    pthread_mutex_lock(&mu);
    if (S.active) {
        logline("SPLIT_UNSUPPORTED nested begin");
        pthread_mutex_unlock(&mu);
        return;
    }
    unsigned lim = S.limit;
    memset(&S, 0, sizeof(S));
    S.limit = lim ? lim : 118;
    S.active = 1;
    S.st = s;
    S.mode = (unsigned)mode;
    S.owner = pthread_self();
    logline("SPLIT_BEGIN stream=%p mode=%d limit=%u", (void *)s, mode, S.limit);
    pthread_mutex_unlock(&mu);
}

/* 会话收尾。返回 0 = 已处理；1 = 这次捕获不归本层管，调用方自己走真函数；
 * -1 = 处理失败（调用方返回 CUDA_ERROR_INVALID_VALUE）。 */
static int sess_end(CUstream s, CUgraph *pg)
{
    pthread_mutex_lock(&mu);
    if (!S.active || S.st != s || !pthread_equal(S.owner, pthread_self())) {
        pthread_mutex_unlock(&mu);
        return 1;
    }

    int rc = do_seal(s);
    if (rc != 0 || S.nchunk == 0) {
        S.active = 0;
        pthread_mutex_unlock(&mu);
        if (pg) *pg = NULL;
        return -1;
    }

    seq_t *q = NULL;
    for (int i = 0; i < MAXSEQ; i++)
        if (!SQ[i].used) { q = &SQ[i]; break; }
    if (!q) {
        logline("SPLIT_SEQ_FULL");
        S.active = 0;
        pthread_mutex_unlock(&mu);
        if (pg) *pg = NULL;
        return -1;
    }

    memset(q, 0, sizeof(*q));
    q->used = 1;
    q->graph = S.ch[0].g;
    q->n = S.nchunk;
    for (unsigned i = 0; i < S.nchunk; i++) q->ch[i] = S.ch[i];

    /* 默认不在这里建 exec：同一 context 里两个 exec 共存到 64 节点以上就会在 launch 第二段时崩，
     * 所以第 1..n-1 段改成 launch 时现场建（串行）。 */
    int post_bad = 0;
    if (opt_keep) {
        for (unsigned i = 1; i < q->n; i++) {
            CUresult ci = cb_inst(&q->ch[i].e, q->ch[i].g, NULL, NULL, 0);
            if (ci != 0 || !q->ch[i].e) {
                logline("SPLIT_POSTINST_FAIL idx=%u rc=%d e=%p",
                        i, ci, (void *)q->ch[i].e);
                post_bad = 1;
            }
        }
    }

    S.active = 0;
    if (pg) *pg = q->graph;
    logline("SPLIT_END chunks=%u nodes0=%u post_bad=%d graph=%p",
            q->n, q->ch[0].nodes, post_bad, (void *)q->graph);
    pthread_mutex_unlock(&mu);
    return 0;
}

/* 代理入口 */

CUresult cuStreamBeginCapture_v2(CUstream s, int mode)
{
    ensure();
    bind_drv();
    if (!r_begin) return CUDA_ERROR_INVALID_VALUE;
    /* runtime 层已经接管会话就不在这里再开会话，否则同一进程里会 double-open */
    if (rt_on) return r_begin(s, mode);
    CUresult r = r_begin(s, mode);
    if (r == 0) sess_open(s, mode);
    return r;
}

CUresult cuStreamBeginCapture(CUstream s, int mode)
{
    return cuStreamBeginCapture_v2(s, mode);
}

CUresult cuStreamEndCapture(CUstream s, CUgraph *pg)
{
    ensure();
    bind_drv();
    if (!r_end) return CUDA_ERROR_INVALID_VALUE;
    if (rt_on) return r_end(s, pg);   /* 会话归 runtime 层管 */
    int rc = sess_end(s, pg);
    if (rc == 1) return r_end(s, pg);
    return rc == 0 ? 0 : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuGraphInstantiate_v2(CUgraphExec *pe, CUgraph g, CUgraphNode *errNode,
                               char *errStr, unsigned flags)
{
    ensure();
    bind_drv();
    if (!r_inst) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = cb_inst(pe, g, errNode, errStr, flags);

    pthread_mutex_lock(&mu);
    seq_t *q = find_seq_graph(g);
    if (q && r == 0 && pe) {
        q->exec = *pe;
        nbind++;
        logline("SPLIT_INST_BIND chunks=%u exec=%p", q->n, (void *)*pe);
    }
    pthread_mutex_unlock(&mu);
    return r;
}

CUresult cuGraphInstantiate(CUgraphExec *pe, CUgraph g, CUgraphNode *errNode,
                            unsigned flags)
{
    ensure();
    bind_drv();
    if (!r_inst) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = cb_inst(pe, g, errNode, NULL, flags);

    pthread_mutex_lock(&mu);
    seq_t *q = find_seq_graph(g);
    if (q && r == 0 && pe) {
        q->exec = *pe;
        nbind++;
        logline("SPLIT_INST_BIND chunks=%u exec=%p", q->n, (void *)*pe);
    }
    pthread_mutex_unlock(&mu);
    return r;
}

CUresult cuGraphLaunch(CUgraphExec e, CUstream s)
{
    ensure();
    bind_drv();
    if (!r_launch) return CUDA_ERROR_INVALID_VALUE;

    pthread_mutex_lock(&mu);
    seq_t *q = find_seq_exec(e);
    if (!q) {
        pthread_mutex_unlock(&mu);
        return cb_launch(e, s);
    }
    unsigned n = q->n;
    CUresult rc = 0;
    unsigned done = 0;

    if (!opt_keep) {
        /* 串行路径：第 0 段用应用自己的 exec；其余每段现场 instantiate →
         * launch → 同步 → 销毁，任何时刻只有一个额外 exec 活着。 */
        for (unsigned i = 0; i < n; i++) {
            if (i == 0) {
                if (!q->exec) { rc = CUDA_ERROR_INVALID_VALUE; break; }
                if (opt_trace) logline("SPLIT_LCH i=0 e=%p", (void *)q->exec);
                rc = cb_launch(q->exec, s);
                if (opt_trace) logline("SPLIT_LCH_RET i=0 rc=%d", rc);
                if (rc != 0) break;
                /* 只有后面还有段才必须等这一段跑完：单段图不许加同步，
                 * 否则没切段的场景会被这条同步白白拖慢一次（实测 +1ms） */
                if (n > 1 && r_sync) rc = cb_sync(s);
                if (rc != 0) break;
                done++;
                continue;
            }
            CUgraphExec ce = NULL;
            CUresult ci = cb_inst(&ce, q->ch[i].g, NULL, NULL, 0);
            if (ci != 0 || !ce) {
                rc = ci ? ci : CUDA_ERROR_INVALID_VALUE;
                logline("SPLIT_LAZY_INST_FAIL i=%u rc=%d", i, ci);
                break;
            }
            if (opt_trace) logline("SPLIT_LCH i=%u e=%p", i, (void *)ce);
            rc = cb_launch(ce, s);
            if (opt_trace) logline("SPLIT_LCH_RET i=%u rc=%d", i, rc);
            /* 这一段的 exec 马上要销毁，必须先等它跑完，否则 destroy 会和还在跑的
             * 图撞在一起（实测会挂死） */
            if (rc == 0 && r_sync) rc = cb_sync(s);
            if (r_edestroy) cb_ed(ce);
            if (rc != 0) break;
            done++;
        }
        nlaunch++;
        pthread_mutex_unlock(&mu);
        if (rc != 0) logline("SPLIT_LAUNCH_FAIL chunks=%u done=%u rc=%d", n, done, rc);
        return rc;
    }

    for (unsigned i = 0; i < n; i++) {
        CUgraphExec ei = q->ch[i].e;
        if (i == 0) ei = q->exec;
        if (!ei) { rc = CUDA_ERROR_INVALID_VALUE; break; }
        if (opt_trace) logline("SPLIT_LCH i=%u e=%p", i, (void *)ei);
        rc = cb_launch(ei, s);
        if (opt_trace) logline("SPLIT_LCH_RET i=%u rc=%d", i, rc);
        if (rc != 0) break;
        done++;
    }
    nlaunch++;
    pthread_mutex_unlock(&mu);
    if (rc != 0) logline("SPLIT_LAUNCH_FAIL chunks=%u done=%u rc=%d", n, done, rc);
    return rc;
}

CUresult cuGraphExecDestroy(CUgraphExec e)
{
    ensure();
    bind_drv();
    pthread_mutex_lock(&mu);
    seq_t *q = find_seq_exec(e);
    if (q) {
        for (unsigned i = 0; i < q->n; i++)
            if (q->ch[i].e && r_edestroy) cb_ed(q->ch[i].e);
        memset(q, 0, sizeof(*q));
        pthread_mutex_unlock(&mu);
        logline("SPLIT_EXEC_DESTROY");
        return 0;
    }
    pthread_mutex_unlock(&mu);
    return r_edestroy ? cb_ed(e) : 0;
}

CUresult cuGraphDestroy(CUgraph g)
{
    ensure();
    bind_drv();
    pthread_mutex_lock(&mu);
    seq_t *q = find_seq_graph(g);
    if (q) {
        for (unsigned i = 0; i < q->n; i++)
            if (q->ch[i].g && r_gdestroy) cb_gd(q->ch[i].g);
        memset(q, 0, sizeof(*q));
        pthread_mutex_unlock(&mu);
        logline("SPLIT_GRAPH_DESTROY");
        return 0;
    }
    pthread_mutex_unlock(&mu);
    return r_gdestroy ? cb_gd(g) : 0;
}

/* ---- 可记录 API：先调真函数，成功后计数 ---- */

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void *src, size_t n, CUstream s)
{
    ensure();
    bind_drv();
    if (!r_cpy_htod) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = r_cpy_htod(d, src, n, s);
    if (r == 0) note_record(s);
    return r;
}

CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr d, size_t n, CUstream s)
{
    ensure();
    bind_drv();
    if (!r_cpy_dtoh) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = r_cpy_dtoh(dst, d, n, s);
    if (r == 0) note_record(s);
    return r;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr src, size_t n, CUstream s)
{
    ensure();
    bind_drv();
    if (!r_cpy_dtod) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = r_cpy_dtod(d, src, n, s);
    if (r == 0) note_record(s);
    return r;
}

CUresult cuMemsetD32Async(CUdeviceptr d, unsigned v, size_t cnt, CUstream s)
{
    ensure();
    bind_drv();
    if (!r_memset_d32a) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = r_memset_d32a(d, v, cnt, s);
    if (r == 0) note_record(s);
    return r;
}

CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t cnt, CUstream s)
{
    ensure();
    bind_drv();
    if (!r_memset_d8a) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = r_memset_d8a(d, v, cnt, s);
    if (r == 0) note_record(s);
    return r;
}

CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned by, unsigned bz, unsigned shm,
                        CUstream s, void **params, void **extra)
{
    ensure();
    bind_drv();
    if (!r_launch_kernel) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = r_launch_kernel(f, gx, gy, gz, bx, by, bz, shm, s, params, extra);
    if (r == 0) note_record(s);
    return r;
}

CUresult cuEventRecord(CUevent e, CUstream s)
{
    ensure();
    bind_drv();
    if (!r_ev_record) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = r_ev_record(e, s);
    if (r == 0) note_record(s);
    return r;
}

CUresult cuStreamWaitEvent(CUstream s, CUevent e, unsigned flags)
{
    ensure();
    bind_drv();
    if (!r_wait_ev) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = r_wait_ev(s, e, flags);
    if (r == 0) note_record(s);
    return r;
}

/* runtime API 层（libcudart）
 * 为什么必须单列一层：libcudart 走自己的驱动入口分派表，不经过 PLT 调 libcuda，所以在驱动符号上做影子永远命不中。
 *
 * 图对象的句柄两层是同一份底层对象，所以 Instantiate / Launch / Destroy 直接转给
 * 驱动层的代理即可；只有 Begin / End 和"计数"必须在这一层做。
 */

typedef struct { unsigned x, y, z; } dim3v;

static void *hrt;
static CUresult (*rt_begin)(void *, int);
static CUresult (*rt_end)(void *, void **);
static CUresult (*rt_inst)(void **, void *, void **, char *, size_t);
static CUresult (*rt_launch_c)(void *, void *);
static CUresult (*rt_edestroy_c)(void *);
static CUresult (*rt_gdestroy_c)(void *);
static CUresult (*rt_launch_kernel)(const void *, dim3v, dim3v, void **, size_t, void *);
static CUresult (*rt_memcpy_async)(void *, const void *, size_t, int, void *);
static CUresult (*rt_memset_async)(void *, int, size_t, void *);
static CUresult (*rt_event_record)(void *, void *);
static CUresult (*rt_wait_event)(void *, void *, unsigned);

static void *rt_sym(const char *a, const char *b)
{
    void *p = hrt ? dlsym(hrt, a) : NULL;
    if (!p && hrt && b) p = dlsym(hrt, b);
    return p;
}

static int rt_ensure(void)
{
    if (rt_on) return 1;
    if (!hrt) {
        hrt = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hrt) hrt = dlopen("libcudart.so.10.2", RTLD_NOW | RTLD_GLOBAL);
        if (!hrt) hrt = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_GLOBAL);
        if (!hrt) return 0;
    }
    rt_begin        = rt_sym("cudaStreamBeginCapture", NULL);
    rt_end          = rt_sym("cudaStreamEndCapture", NULL);
    rt_inst         = (CUresult (*)(void **, void *, void **, char *, size_t))
                      rt_sym("cudaGraphInstantiate", NULL);
    rt_launch_c     = rt_sym("cudaGraphLaunch", NULL);
    rt_edestroy_c   = rt_sym("cudaGraphExecDestroy", NULL);
    rt_gdestroy_c   = rt_sym("cudaGraphDestroy", NULL);
    rt_launch_kernel= (CUresult (*)(const void *, dim3v, dim3v, void **, size_t, void *))
                      rt_sym("cudaLaunchKernel", NULL);
    rt_memcpy_async = (CUresult (*)(void *, const void *, size_t, int, void *))
                      rt_sym("cudaMemcpyAsync", NULL);
    rt_memset_async = (CUresult (*)(void *, int, size_t, void *))
                      rt_sym("cudaMemsetAsync", NULL);
    rt_event_record = rt_sym("cudaEventRecord", NULL);
    rt_wait_event   = rt_sym("cudaStreamWaitEvent", NULL);

    if (!rt_begin || !rt_end) { logline("SPLIT_RT_SYM_MISS"); return 0; }
    rt_on = 1;
    logline("SPLIT_RT_INSTALL limit=%u begin=%p end=%p inst=%p launch=%p kernel=%p",
            S.limit, (void *)rt_begin, (void *)rt_end, (void *)rt_inst,
            (void *)rt_launch_c, (void *)rt_launch_kernel);
    return 1;
}

static void bind_rt(void)
{
    cur_rt = 1;
    cb_begin  = (CUresult (*)(CUstream, int))rt_begin;
    cb_end    = (CUresult (*)(CUstream, CUgraph *))rt_end;
    cb_inst   = (CUresult (*)(CUgraphExec *, CUgraph, CUgraphNode *, char *, unsigned))rt_inst;
    cb_launch = (CUresult (*)(CUgraphExec, CUstream))rt_launch_c;
    cb_sync   = r_sync;          /* cudart 侧同步仍用驱动的 stream 同步 */
    cb_ed     = (CUresult (*)(CUgraphExec))rt_edestroy_c;
    cb_gd     = (CUresult (*)(CUgraph))rt_gdestroy_c;
}

CUresult cudaStreamBeginCapture(void *s, int mode)
{
    ensure();
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    bind_rt();
    CUresult r = rt_begin(s, mode);
    if (r == 0) sess_open((CUstream)s, mode);
    return r;
}

CUresult cudaStreamEndCapture(void *s, void **pg)
{
    ensure();
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    bind_rt();
    int rc = sess_end((CUstream)s, (CUgraph *)pg);
    if (rc == 1) return rt_end(s, pg);
    return rc == 0 ? 0 : CUDA_ERROR_INVALID_VALUE;
}

CUresult cudaGraphInstantiate(void **pe, void *g, void **errNode, char *errStr, size_t bufsz)
{
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    (void)bufsz;
    CUresult r = cuGraphInstantiate_v2((CUgraphExec *)pe, (CUgraph)g,
                                       (CUgraphNode *)errNode, errStr, 0);
    return r;
}

CUresult cudaGraphLaunch(void *e, void *s)
{
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    return cuGraphLaunch((CUgraphExec)e, (CUstream)s);
}

CUresult cudaGraphExecDestroy(void *e)
{
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    return cuGraphExecDestroy((CUgraphExec)e);
}

CUresult cudaGraphDestroy(void *g)
{
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    return cuGraphDestroy((CUgraph)g);
}

CUresult cudaLaunchKernel(const void *f, dim3v grid, dim3v block, void **args,
                          size_t shm, void *s)
{
    ensure();
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    bind_rt();
    if (!rt_launch_kernel) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = rt_launch_kernel(f, grid, block, args, shm, s);
    if (r == 0) note_record((CUstream)s);
    return r;
}

CUresult cudaMemcpyAsync(void *dst, const void *src, size_t n, int kind, void *s)
{
    ensure();
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    bind_rt();
    if (!rt_memcpy_async) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = rt_memcpy_async(dst, src, n, kind, s);
    if (r == 0) note_record((CUstream)s);
    return r;
}

CUresult cudaMemsetAsync(void *d, int v, size_t n, void *s)
{
    ensure();
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    bind_rt();
    if (!rt_memset_async) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = rt_memset_async(d, v, n, s);
    if (r == 0) note_record((CUstream)s);
    return r;
}

CUresult cudaEventRecord(void *e, void *s)
{
    ensure();
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    bind_rt();
    if (!rt_event_record) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = rt_event_record(e, s);
    if (r == 0) note_record((CUstream)s);
    return r;
}

CUresult cudaStreamWaitEvent(void *s, void *e, unsigned flags)
{
    ensure();
    if (!rt_ensure()) return CUDA_ERROR_INVALID_VALUE;
    bind_rt();
    if (!rt_wait_event) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = rt_wait_event(s, e, flags);
    if (r == 0) note_record((CUstream)s);
    return r;
}

__attribute__((constructor))
static void ctor(void)
{
    S.limit = 118;
    const char *p = getenv("SPLIT_LOG");
    L = fopen(p ? p : "/tmp/cap_split.log", "a");
    const char *m = getenv("SPLIT_LIMIT");
    if (m) {
        unsigned v = (unsigned)strtoul(m, NULL, 0);
        if (v > 1 && v <= 128) S.limit = v;
    }
    opt_use_nodes = getenv("SPLIT_USE_NODES") ? 1 : 0;
    opt_trace = getenv("SPLIT_TRACE") ? 1 : 0;
    opt_keep = getenv("SPLIT_KEEP") ? 1 : 0;   /* 1 = 各段 exec 常驻（会崩，仅用于对比） */
    logline("SPLIT_LOAD limit=%u", S.limit);
}

__attribute__((destructor))
static void dtor(void)
{
    logline("SPLIT_SUMMARY record=%lu seal=%lu bind=%lu launch=%lu",
            nrec, nseal, nbind, nlaunch);
    if (L) fclose(L);
}

/* libcuda 可能在 ctor 之后才可用；借 ioctl 兜一次懒装配 */
int ioctl(int fd, unsigned long request, ...)
{
    va_list ap;
    void *arg;
    va_start(ap, request);
    arg = va_arg(ap, void *);
    va_end(ap);
    ensure();
    return (int)syscall(SYS_ioctl, fd, request, arg);
}
