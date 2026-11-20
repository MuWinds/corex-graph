/*
 * cmdbuf_grow.c 
 * 抬高 UMD cmd 块容量，消除：包/参数双向 bump 交叉 → 参数指针落到块外 → CP 页错误 → 卡挂掉。
 *
 * 为什么不换 base：静态逆向结果显示：
 * 容量与 base 都不是常量，也不是按 stride 推算，而是每块各自向驱动申请，容量取自 *(席位+0x54)。块上 base 同时存在于 +0x10 和 +0x20。
 *
 * 做法：
 *  在 0x4351d0 入口打 13 字节 inline hook：先算出席位对象、把 *(席位+0x54)
 *  改成目标容量，再走原函数。于是分配的 size、块容量 +0x0c、base 副本 +0x10、
 *  分配对象 +0x18、base +0x20 全部由驱动自己生成，天然一致。
 *  一个字节都不改块内字段，也不自建任何缓冲。
 *
 * 依据：
 *  0x435410  块 reset：+0x38=0、+0x0c=*(ring+0x54)、+0x30=-1、jmp 397530 填 base
 *  0x435460  off += n（无检查）
 *  0x435530  remain -= n（u32 回绕、无检查）
 *  0x4355f0  容量检查（存在，但 kernel launch 路径零调用）
 *  0x435680  收尾：断言 off<=cap，取 (cb+0x20, ceil32(off)) 下发
 *
 * 用法
 *    gcc -shared -fPIC -O2 -o cmdbuf_grow.so cmdbuf_grow.c -ldl
 *    CMDBUF_GROW=1 CMDBUF_SIZE=131072 LD_PRELOAD=/abs/cmdbuf_grow.so <程序>
 *  环境变量：
 *    CMDBUF_GROW  1=启用（默认 0，不启用时零行为）
 *    CMDBUF_SIZE  每块目标字节数（默认 131072 = 128 KiB）
 *    CMDBUF_LOG   日志路径（默认 /tmp/cmdbuf_grow.log）
 *    CMDBUF_PROBE 1=只打印块字段原值，不提升容量
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* 常量（全部来自静态逆向）*/
#define VMA_POOL          0x4351d0UL  /* 池建循环里逐块分配的函数 */
#define POOL_PROLOGUE     13          /* 6 条 push + mov %rdi,%rbx，正好到 0x4351dd */
#define POOL_TAIL         0x4351ddUL

#define VMA_RESET         0x435410UL  /* 块初始化（只读观察点） */
#define RESET_PROLOGUE    13
#define RESET_TAIL        0x43541dUL

#define OFF_SEAT_CAP      0x54        /* 席位对象 +0x54 = 每块容量 */

#define OFF_BLK_CAP       0x0c
#define OFF_BLK_BASE      0x20
#define OFF_BLK_OFF       0x38
#define OFF_BLK_FLAG      0x08
#define OFF_BLK_ALLOC     0x18

/* 全局 */
static FILE *L;
static int   enabled;
static int   probe_only;
static size_t blk_size = 131072;
static int   hooks_pool = 1;          /* CMDBUF_ONLY=reset 时置 0 */
static int   hooks_reset = 1;         /* CMDBUF_ONLY=pool  时置 0 */

static uint64_t mod_base;
static void    *tramp_pool;
static void    *tramp_reset;
static int      hooked;

static void logline(const char *fmt, ...)
{
    if (!L) return;
    va_list ap; va_start(ap, fmt); vfprintf(L, fmt, ap); va_end(ap);
    fputc('\n', L); fflush(L);
}

/* 找 libcuda 装载基址 */
static int ph_cb(struct dl_phdr_info *info, size_t sz, void *data)
{
    (void)sz; (void)data;
    if (info->dlpi_name && strstr(info->dlpi_name, "libcuda.so")) {
        mod_base = (uint64_t)info->dlpi_addr;
        return 1;
    }
    return 0;
}

/* hook 1：抬高每块容量
 * 注意：必须把原函数返回时的 RAX 原样透传出去，本函数处于 API 返回值链上，
 * 返回 blk 会把上层 cuCtxCreate 的返回值污染成指针 */
__attribute__((noinline))
static uint64_t hook_pool(void *blk)
{
    uint64_t (*orig)(void *) = (uint64_t (*)(void *))tramp_pool;

    if (enabled && !probe_only && blk) {
        uint64_t cont = *(uint64_t *)blk;
        uint64_t chan = cont ? *(uint64_t *)((uint8_t *)cont + 0x8) : 0;
        uint64_t dev  = chan ? *(uint64_t *)((uint8_t *)chan + 0x8) : 0;
        uint64_t seat = dev  ? *(uint64_t *)((uint8_t *)dev + 0x158) : 0;

        if (seat) {
            uint32_t old = *(uint32_t *)((uint8_t *)seat + OFF_SEAT_CAP);
            *(uint32_t *)((uint8_t *)seat + OFF_SEAT_CAP) = (uint32_t)blk_size;
            logline("CAP_SET blk=%p seat=0x%llx old=%u new=%u",
                    blk, (unsigned long long)seat, old, (unsigned int)blk_size);
        } else {
            logline("CAP_SKIP blk=%p cont=0x%llx chan=0x%llx dev=0x%llx",
                    blk, (unsigned long long)cont,
                    (unsigned long long)chan, (unsigned long long)dev);
        }
    }

    return orig(blk);   /* 原函数按（已改过的）容量现场分配，并写满 4 个字段 */
}

/* hook 2：只读观察块字段（不改任何值）*/
__attribute__((noinline))
static uint64_t hook_reset(void *blk)
{
    uint64_t (*orig)(void *) = (uint64_t (*)(void *))tramp_reset;
    uint64_t rv = orig(blk);

    if (blk) {
        uint64_t base = *(uint64_t *)((uint8_t *)blk + OFF_BLK_BASE);
        uint64_t b2   = *(uint64_t *)((uint8_t *)blk + 0x10);
        uint64_t alc  = *(uint64_t *)((uint8_t *)blk + OFF_BLK_ALLOC);
        uint32_t cap  = *(uint32_t *)((uint8_t *)blk + OFF_BLK_CAP);
        uint32_t off  = *(uint32_t *)((uint8_t *)blk + OFF_BLK_OFF);
        logline("PROBE_BLK blk=%p base20=0x%llx base10=0x%llx alloc=0x%llx cap=%u off=%u flag=%u",
                blk, (unsigned long long)base, (unsigned long long)b2,
                (unsigned long long)alc, cap, off,
                *(uint8_t *)((uint8_t *)blk + OFF_BLK_FLAG));
    }
    return rv;
}

/*  hook 3：给三个"无检查写入口"补容量闸
 * 静态依据（libcuda.so.1）：
 *   0x442740  写 0x14 字节包（头 0x13000304），尾部 jmp 435460 → off += 0x14
 *             入口 80 7f 21 00 cmpb $0,0x21(rdi) ; 74 7a je 4427c0
 *   0x4427f0  写 0x0c 字节包（头 0x0f000002），off += 0x0c
 *             入口 53 / 48 89 fb / 48 89 f7 / e8 …(call 382bf0)
 *   0x442830  写 0x20 字节包（头 0x12010207），off += 0x20
 *             入口 48 8b 7f 28 / 8b 47 38
 * 三者都直接往 base+off 落笔、绝不检查、绝不换块。
 * 0x3bdc20(obj, n)：检查 *(obj+0x28) 当前块的剩余，不够就 3baa80(obj,9)
 * 收尾提交 + 3bda60 换块 —— 就是描述符路 0x3bdc80 用的那一套。
 */
#define VMA_W1   0x442740UL
#define VMA_W2   0x4427f0UL
#define VMA_W3   0x442830UL
#define VMA_CHK  0x3bdc20UL
#define VMA_CALL382BF0 0x382bf0UL
#define R_W1     0x442746UL   /* je 落空后的主体入口 */
#define R_W1_ALT 0x4427c0UL   /* cmpb 为 0 时的去向 */
#define R_W2     0x4427fcUL
#define R_W3     0x442837UL

static void *tramp_w1, *tramp_w2, *tramp_w3;
static int   chk_on;                /* CMDBUF_CHK=1 */
static int   trace_on;              /* CMDBUF_TRACE=1 */
static int   node_on;               /* CMDBUF_NODE=1：在分派循环里补容量闸 */
static unsigned node_cost = 64;     /* 每节点预留字节，CMDBUF_NODECOST 可覆盖 */
static unsigned long nchk, nrot, nadd;

#define VMA_ADD  0x435460UL   /* off 的唯一 bump 点：121 个构包器共用的收尾 */

/* 日志侧：只负责记录，算术已经在汇编桩里做完了。
 * 关键：0x435460 返回值是 eax = 新 off，调用方用它算写入地址，所以桩里必须保真。 */
static void lg_add(void *p, unsigned n, unsigned off, unsigned long long ra)
{
    uint32_t cap = *(uint32_t *)((uint8_t *)p + 0x0c);
    if (nadd < 400)
        logline("ADD ra=0x%llx n=%u off=%u cap=%u over=%d",
                ra, n, off, cap, (int)(off > cap));
    nadd++;
}

/* 汇编桩：
 *   mov 0x38(%rdi),%eax ; add %esi,%eax ; mov %eax,0x38(%rdi)   ← 原三条，rax/flags 保真
 *   mov (%rsp),%rcx                                              ← 取调用者返回地址
 *   保存 rax/rcx/rdx/rsi/rdi/r8-r11（9 条 push，72B，入口 rsp≡8 → ≡0，可直接 call）
 *   rdi=[rsp+32] p, esi=[rsp+40] n, edx=[rsp+64] 新off, rcx=[rsp+56] ra
 *   还原后 ret，rax 仍是新 off */
static void *mk_add_stub(void)
{
    uint8_t *t = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t == MAP_FAILED) return NULL;
    uint8_t *p = t;
    *p++ = 0x8b; *p++ = 0x47; *p++ = 0x38;
    *p++ = 0x01; *p++ = 0xf0;
    *p++ = 0x89; *p++ = 0x47; *p++ = 0x38;
    *p++ = 0x48; *p++ = 0x8b; *p++ = 0x0c; *p++ = 0x24;
    *p++ = 0x50; *p++ = 0x51; *p++ = 0x52; *p++ = 0x56; *p++ = 0x57;
    *p++ = 0x41; *p++ = 0x50;
    *p++ = 0x41; *p++ = 0x51;
    *p++ = 0x41; *p++ = 0x52;
    *p++ = 0x41; *p++ = 0x53;
    *p++ = 0x48; *p++ = 0x8b; *p++ = 0x7c; *p++ = 0x24; *p++ = 0x20;  /* mov 0x20(%rsp),%rdi */
    *p++ = 0x8b; *p++ = 0x74; *p++ = 0x24; *p++ = 0x28;               /* mov 0x28(%rsp),%esi */
    *p++ = 0x8b; *p++ = 0x54; *p++ = 0x24; *p++ = 0x40;               /* mov 0x40(%rsp),%edx */
    *p++ = 0x48; *p++ = 0x8b; *p++ = 0x4c; *p++ = 0x24; *p++ = 0x38;  /* mov 0x38(%rsp),%rcx */
    {
        int32_t rel = (int32_t)((uint64_t)(uintptr_t)lg_add - (uint64_t)(uintptr_t)(p + 5));
        *p++ = 0xe8; memcpy(p, &rel, 4); p += 4;
    }
    *p++ = 0x41; *p++ = 0x5b;
    *p++ = 0x41; *p++ = 0x5a;
    *p++ = 0x41; *p++ = 0x59;
    *p++ = 0x41; *p++ = 0x58;
    *p++ = 0x5f; *p++ = 0x5e; *p++ = 0x5a; *p++ = 0x59; *p++ = 0x58;
    *p++ = 0xc3;
    __builtin___clear_cache((char *)t, (char *)p);
    return t;
}

/* 只在确实装不下时轮换，并且把 +0x90 / +0x98 一起清零。
 * 依据 0x3baa80：提交包携带 idx*64 与 seq（3baae9 / 3baafc / 3bab44），
 * 而 0x435410 换块只重置 +0x08/+0x30/+0x38/+0x0c，不动这两个字段。 */
static void chk_cap(void *obj, unsigned n)
{
    uint64_t cb = *(uint64_t *)((uint8_t *)obj + 0x28);
    if (!cb) return;
    uint32_t off = *(uint32_t *)((uint8_t *)cb + 0x38);
    uint32_t cap = *(uint32_t *)((uint8_t *)cb + 0x0c);
    nchk++;
    /* 0x4355f0 是严格小于：off+n < cap 才放行。所以这里用 >=。 */
    if ((uint64_t)off + n < (uint64_t)cap) return;

    nrot++;
    logline("CHK_ROT obj=%p cb=%p off=%u cap=%u need=%u",
            obj, (void *)cb, off, cap, n);

    /* 严格按原厂序（0x3dc20 与 0x3bab5f 的实证）：
     * 1) 0x3baa80(obj,9)：提交，同时置 cb+0x08=1、把 seq 快照到 cb+0xa0
     * 2) idx/seq 归零（435410 不管这两个字段）
     * 3) 0x3bda60(obj)：换块 → 435410 重置 +0x08/+0x30/+0x38/+0x0c
     */
    {
        void (*sub)(void *, int)   = (void (*)(void *, int))(mod_base + 0x3baa80UL);
        void (*rot)(void *)        = (void (*)(void *))(mod_base + 0x3bda60UL);
        sub(obj, 9);
        *(uint32_t *)((uint8_t *)obj + 0x90) = 0;
        *(uint64_t *)((uint8_t *)obj + 0x98) = 0;
        rot(obj);
    }
}

static uint8_t *emit_jmp(uint8_t *p, uint64_t to);

/* 分派循环里的闸。rdi = 通道对象（= rbp），obj+0x28 就是块。
 * 静态依据：
 *   38a9f0  mov 0x0(%rbp),%rax        ; rax = *(obj) = 分派表
 *   38aa03  mov %rbp,%rdi             ; 构包器第一参数 = 通道对象
 *   38aa06  call *0x18(%rax)          ; 主节点构包器
 *   38aa62  call *0x10(%rax)          ; 另一类节点构包器
 */
static void ck_node(void *obj) { chk_cap(obj, node_cost); }

/* 桩：循环头 0x38a974（9 字节 = mov (%rdx,%r14,8),%r8 ; cmpl $0x19,0x4(%r8)）
 * 放在这里换块，是因为 0x38a9bf 才用 *(表基) 算出 r15（描述符指针）；
 * 若在那之后换块，r15 就是旧块的野指针（实测 60/128/200 节点段错误就是这么来的）。
 * rbp = 通道对象（0x38aa03 就是 mov %rbp,%rdi 当构包器第一参数）。 */
static void *mk_node_stub(uint64_t resume)
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
    *p++ = 0x48; *p++ = 0x89; *p++ = 0xef;      /* mov %rbp,%rdi */
    {
        int32_t rel = (int32_t)((uint64_t)(uintptr_t)ck_node - (uint64_t)(uintptr_t)(p + 5));
        *p++ = 0xe8; memcpy(p, &rel, 4); p += 4;
    }
    *p++ = 0x41; *p++ = 0x5b;
    *p++ = 0x41; *p++ = 0x5a;
    *p++ = 0x41; *p++ = 0x59;
    *p++ = 0x41; *p++ = 0x58;
    *p++ = 0x5f; *p++ = 0x5e; *p++ = 0x5a; *p++ = 0x59; *p++ = 0x58;
    *p++ = 0x4e; *p++ = 0x8b; *p++ = 0x04; *p++ = 0xf2;              /* mov (%rdx,%r14,8),%r8 */
    *p++ = 0x41; *p++ = 0x83; *p++ = 0x78; *p++ = 0x04; *p++ = 0x19;  /* cmpl $0x19,0x4(%r8) */
    p = emit_jmp(p, resume);
    __builtin___clear_cache((char *)t, (char *)p);
    return t;
}

static void ck1(void *o) { chk_cap(o, 0x14); }
static void ck2(void *o) { chk_cap(o, 0x0c); }
static void ck3(void *o) { chk_cap(o, 0x20); }

/* 生成"保存全部易失寄存器 → 调 C 检查 → 还原 → 跳蹦床"的桩。
 * 静态依据：0x4427f0 主体第一句是 mov %rsi,%rdi（rsi 是载荷），
 *           0x442830 主体 44283b 是 mov %esi,0x4(%rax)（rsi/edx 是载荷）。
 * 所以绝不能让 C 编译器把这些寄存器清掉。 */
static void *mk_stub(void (*ck)(void *), void *tramp)
{
    uint8_t *t = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t == MAP_FAILED) return NULL;
    uint8_t *p = t;
    *p++ = 0x50;                       /* push %rax */
    *p++ = 0x51;                       /* push %rcx */
    *p++ = 0x52;                       /* push %rdx */
    *p++ = 0x56;                       /* push %rsi */
    *p++ = 0x57;                       /* push %rdi */
    *p++ = 0x41; *p++ = 0x50;          /* push %r8  */
    *p++ = 0x41; *p++ = 0x51;          /* push %r9  */
    *p++ = 0x41; *p++ = 0x52;          /* push %r10 */
    *p++ = 0x41; *p++ = 0x53;          /* push %r11 */
    /* 9 条 push 共 72 字节：入口 rsp≡8(mod16) → 现在 ≡0，可直接 call */
    {
        int32_t rel = (int32_t)((uint64_t)(uintptr_t)ck - (uint64_t)(uintptr_t)(p + 5));
        *p++ = 0xe8; memcpy(p, &rel, 4); p += 4;
    }
    *p++ = 0x41; *p++ = 0x5b;          /* pop %r11 */
    *p++ = 0x41; *p++ = 0x5a;          /* pop %r10 */
    *p++ = 0x41; *p++ = 0x59;          /* pop %r9  */
    *p++ = 0x41; *p++ = 0x58;          /* pop %r8  */
    *p++ = 0x5f;                       /* pop %rdi */
    *p++ = 0x5e;                       /* pop %rsi */
    *p++ = 0x5a;                       /* pop %rdx */
    *p++ = 0x59;                       /* pop %rcx */
    *p++ = 0x58;                       /* pop %rax */
    p = emit_jmp(p, (uint64_t)(uintptr_t)tramp);
    __builtin___clear_cache((char *)t, (char *)p);
    return t;
}

static void *stub_w1, *stub_w2, *stub_w3;

/* 生成"复刻前缀 + 绝对回跳"的蹦床 */
static uint8_t *emit_jmp(uint8_t *p, uint64_t to)
{
    int32_t rel = (int32_t)(to - (uint64_t)(uintptr_t)(p + 5));
    *p++ = 0xe9; memcpy(p, &rel, 4); p += 4;
    return p;
}

static void *mk_w1(void)
{
    uint8_t *t = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t == MAP_FAILED) return NULL;
    uint8_t *p = t;
    p[0] = 0x80; p[1] = 0x7f; p[2] = 0x21; p[3] = 0x00;   /* cmpb $0,0x21(rdi) */
    p[4] = 0x74; p[5] = 0x05;                             /* je .alt（跳过下面 5 字节） */
    p += 6;
    p = emit_jmp(p, mod_base + R_W1);
    p = emit_jmp(p, mod_base + R_W1_ALT);
    __builtin___clear_cache((char *)t, (char *)p);
    return t;
}

static void *mk_w2(void)
{
    uint8_t *t = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t == MAP_FAILED) return NULL;
    uint8_t *p = t;
    *p++ = 0x53;                                   /* push rbx */
    *p++ = 0x48; *p++ = 0x89; *p++ = 0xfb;          /* mov %rdi,%rbx */
    *p++ = 0x48; *p++ = 0x89; *p++ = 0xf7;          /* mov %rsi,%rdi */
    /* call 382bf0：重算 rel32（不能照抄原位移） */
    {
        int32_t rel = (int32_t)((mod_base + VMA_CALL382BF0) - (uint64_t)(uintptr_t)(p + 5));
        *p++ = 0xe8; memcpy(p, &rel, 4); p += 4;
    }
    p = emit_jmp(p, mod_base + R_W2);
    __builtin___clear_cache((char *)t, (char *)p);
    return t;
}

static void *mk_w3(void)
{
    uint8_t *t = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t == MAP_FAILED) return NULL;
    uint8_t *p = t;
    p[0] = 0x48; p[1] = 0x8b; p[2] = 0x7f; p[3] = 0x28;   /* mov 0x28(%rdi),%rdi */
    p[4] = 0x8b; p[5] = 0x47; p[6] = 0x38;                /* mov 0x38(%rdi),%eax */
    p += 7;
    p = emit_jmp(p, mod_base + R_W3);
    __builtin___clear_cache((char *)t, (char *)p);
    return t;
}

/* 用 5 字节相对跳转覆盖入口，余下字节补 nop */
static int patch_site(uint64_t vma, size_t cover, void *fn)
{
    uint8_t *at = (uint8_t *)(mod_base + vma);
    long ps = sysconf(_SC_PAGESIZE);
    void *page = (void *)((uintptr_t)at & ~(uintptr_t)(ps - 1));
    if (mprotect(page, (size_t)ps * 2, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        logline("GROW_MPROTECT_FAIL vma=0x%llx", (unsigned long long)vma);
        return 0;
    }
    int32_t rel = (int32_t)((uint64_t)(uintptr_t)fn - (uint64_t)(uintptr_t)(at + 5));
    at[0] = 0xe9;
    memcpy(at + 1, &rel, 4);
    for (size_t i = 5; i < cover; i++) at[i] = 0x90;
    __builtin___clear_cache((char *)at, (char *)at + cover);
    mprotect(page, (size_t)ps * 2, PROT_READ | PROT_EXEC);
    return 1;
}

static int install_chk(void)
{
    tramp_w1 = mk_w1();
    tramp_w2 = mk_w2();
    tramp_w3 = mk_w3();
    if (!tramp_w1 || !tramp_w2 || !tramp_w3) { logline("CHK_TRAMP_FAIL"); return 0; }
    stub_w1 = mk_stub(ck1, tramp_w1);
    stub_w2 = mk_stub(ck2, tramp_w2);
    stub_w3 = mk_stub(ck3, tramp_w3);
    if (!stub_w1 || !stub_w2 || !stub_w3) { logline("CHK_STUB_FAIL"); return 0; }
    if (!patch_site(VMA_W1, 6,  stub_w1)) return 0;   /* 4+2 */
    if (!patch_site(VMA_W2, 12, stub_w2)) return 0;   /* 1+3+3+5 */
    if (!patch_site(VMA_W3, 7,  stub_w3)) return 0;   /* 4+3 */
    logline("CHK_HOOK on w1=0x%llx w2=0x%llx w3=0x%llx chk=0x%llx",
            (unsigned long long)(mod_base + VMA_W1),
            (unsigned long long)(mod_base + VMA_W2),
            (unsigned long long)(mod_base + VMA_W3),
            (unsigned long long)(mod_base + VMA_CHK));
    return 1;
}

/* tramp 与打点 */
static void *make_tramp(const uint8_t *pfx, size_t n, uint64_t tail)
{
    void *t = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t == MAP_FAILED) return NULL;
    uint8_t *p = (uint8_t *)t;
    memcpy(p, pfx, n);
    p[n + 0] = 0xff; p[n + 1] = 0x25;
    p[n + 2] = 0; p[n + 3] = 0; p[n + 4] = 0; p[n + 5] = 0;
    *(uint64_t *)(p + n + 6) = tail;
    __builtin___clear_cache((char *)p, (char *)p + n + 14);
    return t;
}

static int place_hook(uint64_t vma, void *fn)
{
    uint8_t *target = (uint8_t *)(mod_base + vma);
    long ps = sysconf(_SC_PAGESIZE);
    void *page = (void *)((uintptr_t)target & ~(uintptr_t)(ps - 1));
    if (mprotect(page, (size_t)ps * 2, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        logline("GROW_MPROTECT_FAIL vma=0x%llx", (unsigned long long)vma);
        return 0;
    }
    uint64_t hop = (uint64_t)(uintptr_t)fn;
    uint8_t stub[13];
    stub[0] = 0x48; stub[1] = 0xb8;
    memcpy(stub + 2, &hop, 8);
    stub[10] = 0xff; stub[11] = 0xe0;
    stub[12] = 0x90;
    memcpy(target, stub, sizeof(stub));
    __builtin___clear_cache((char *)target, (char *)target + sizeof(stub));
    mprotect(page, (size_t)ps * 2, PROT_READ | PROT_EXEC);
    return 1;
}

static int install_hooks(void)
{
    if (hooked) return 1;
    if (!mod_base) {
        dl_iterate_phdr(ph_cb, NULL);
        if (!mod_base) return 0;
        logline("GROW_BASE_LATE 0x%llx", (unsigned long long)mod_base);
    }

    /* 0x4351d0 前 13 字节原样复刻：
     *   41 57             push %r15
     *   41 56             push %r14
     *   41 55             push %r13
     *   41 54             push %r12
     *   55                push %rbp
     *   53                push %rbx
     *   48 89 fb          mov  %rdi,%rbx
     */
    static const uint8_t pool_pfx[] = {
        0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
        0x55, 0x53, 0x48, 0x89, 0xfb
    };
    /* 0x435410 前 13 字节原样复刻：
     *   48 8b 07          mov (%rdi),%rax
     *   48 8d 57 20       lea 0x20(%rdi),%rdx
     *   be 04 00 00 00    mov $4,%esi          （= *(u32*)0x4f5888，静态已取值）
     */
    static const uint8_t reset_pfx[] = {
        0x48, 0x8b, 0x07,
        0x48, 0x8d, 0x57, 0x20,
        0xbe, 0x04, 0x00, 0x00, 0x00
    };

    tramp_pool  = make_tramp(pool_pfx,  sizeof(pool_pfx),  mod_base + POOL_TAIL);
    tramp_reset = make_tramp(reset_pfx, sizeof(reset_pfx), mod_base + RESET_TAIL);
    if (!tramp_pool || !tramp_reset) { logline("GROW_TRAMP_FAIL"); return 0; }

    if (hooks_pool)  { if (!place_hook(VMA_POOL,  (void *)&hook_pool))  return 0; }
    if (hooks_reset) { if (!place_hook(VMA_RESET, (void *)&hook_reset)) return 0; }
    if (chk_on)      { if (!install_chk()) return 0; }
    if (node_on) {
        void *s = mk_node_stub(mod_base + 0x38a97dUL);
        if (!s) { logline("NODE_STUB_FAIL"); return 0; }
        if (!patch_site(0x38a974UL, 9, s)) return 0;
        logline("NODE_GATE inst=0x%llx cost=%u",
                (unsigned long long)(mod_base + 0x38a974UL), node_cost);
    }
    if (trace_on) {
        void *st = mk_add_stub();
        if (!st || !patch_site(VMA_ADD, 5, st)) return 0;
        logline("TRACE_INSTALL 0x%llx", (unsigned long long)(mod_base + VMA_ADD));
    }

    hooked = 1;
    logline("GROW_HOOK base=0x%llx pool=0x%llx reset=0x%llx size=%zu mode=%s",
            (unsigned long long)mod_base,
            (unsigned long long)(mod_base + VMA_POOL),
            (unsigned long long)(mod_base + VMA_RESET),
            blk_size, probe_only ? "probe" : "cap");
    return 1;
}

/* ioctl 拦截：只为尽早装 hook */
int ioctl(int fd, unsigned long request, ...)
{
    static int (*real_ioctl)(int, unsigned long, ...);
    va_list ap; void *arg;
    va_start(ap, request); arg = va_arg(ap, void *); va_end(ap);
    (void)arg;
    if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");

    if ((enabled || probe_only) && !hooked) install_hooks();
    return real_ioctl(fd, request, arg);
}

__attribute__((constructor))
static void ctor(void)
{
    const char *pb = getenv("CMDBUF_PROBE");
    probe_only = pb && atoi(pb) == 1;
    const char *tr = getenv("CMDBUF_TRACE");
    trace_on = tr && atoi(tr) == 1;
    const char *nd = getenv("CMDBUF_NODE");
    node_on = nd && atoi(nd) == 1;
    const char *nc = getenv("CMDBUF_NODECOST");
    if (nc) { node_cost = (unsigned)strtoul(nc, NULL, 0); if (!node_cost) node_cost = 64; }
    enabled = getenv("CMDBUF_GROW") && atoi(getenv("CMDBUF_GROW")) == 1;
    if (!enabled && !probe_only && !trace_on && !node_on) return;

    const char *p = getenv("CMDBUF_LOG");
    L = fopen(p ? p : "/tmp/cmdbuf_grow.log", "a");

    const char *s = getenv("CMDBUF_SIZE");
    if (s) blk_size = (size_t)strtoul(s, NULL, 0);
    if (blk_size < 8192) blk_size = 8192;
    if (blk_size > (1u << 20)) blk_size = 1u << 20;

    const char *only = getenv("CMDBUF_ONLY");
    if (only && strcmp(only, "pool") == 0)   hooks_reset = 0;
    if (only && strcmp(only, "reset") == 0)  hooks_pool = 0;
    if (only && strcmp(only, "none") == 0)   { hooks_pool = 0; hooks_reset = 0; }

    const char *ck = getenv("CMDBUF_CHK");
    chk_on = ck && atoi(ck) == 1;

    dl_iterate_phdr(ph_cb, NULL);
    logline("GROW_INIT pid=%d size=%zu base=0x%llx probe=%d",
            (int)getpid(), blk_size, (unsigned long long)mod_base, probe_only);
    if (!mod_base) logline("GROW_NO_LIBCUDA");
    else install_hooks();
}
