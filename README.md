# graph —— 在自己程序里用上 CUDA Graph（capture + replay）

这套东西的用途只有一句话：**让 CoreX（BI-V100）上的程序能真的用 CUDA Graph**。
厂商驱动单张图最多 128 个节点，超了不是被拦就是崩，所以大模型/大 step 的捕获根本走不通。
这里在 capture 期把大图自动切成每段 ≤118 节点的多段图，`Instantiate` / `Launch` / `Destroy`
仍然是你程序里那几个原生调用，内部按段 replay —— 你的代码不需要知道"切了"这件事。

接入只有一个 `.so` + 三个环境变量，不改你程序一行代码。

---

## 1. 你的程序侧要做的（就四步）

以 runtime API 为例，标准 CUDA Graph 用法即可：

```c
cudaStream_t st;  cudaStreamCreate(&st);

/* 1) 开始捕获 */
cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);

/* 2) 捕获期：kernel / memcpy / memset / event 照常提交，别同步、别切流 */
for (int i = 0; i < N; i++) my_kernel<<<grid, block, 0, st>>>(...);

/* 3) 结束捕获，拿到图 */
cudaGraph_t g = NULL;
cudaStreamEndCapture(st, &g);

/* 4) instantiate 一次，然后每轮只 replay */
cudaGraphExec_t ge = NULL;
cudaGraphInstantiate(&ge, g, NULL, NULL, 0);
for (int it = 0; it < M; it++) {
    cudaGraphLaunch(ge, st);       /* 就是它变快了：N 次提交压成 1 次 */
    cudaStreamSynchronize(st);
}
cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
```

驱动 API 完全对称，参数一一对应：

| runtime | driver |
|---|---|
| `cudaStreamBeginCapture(st, mode)` | `cuStreamBeginCapture_v2(st, mode)` |
| `cudaStreamEndCapture(st, &g)` | `cuStreamEndCapture(st, &g)` |
| `cudaGraphInstantiate(&ge, g, ...)` | `cuGraphInstantiate_v2(&ge, g, ...)` |
| `cudaGraphLaunch(ge, st)` | `cuGraphLaunch(ge, st)` |
| `cudaGraphExecDestroy(ge)` / `cudaGraphDestroy(g)` | `cuGraphExecDestroy` / `cuGraphDestroy` |

两条入口这层都截了（runtime 侧 11 个 `cuda*` 代理、driver 侧 8 个可记录 API + 5 个图 API），
所以你用哪套都行，混用也行。

## 2. 接上

```bash
# 编（计算节点，cuda.h 在 /usr/local/corex/include）
gcc -shared -fPIC -O2 -I/usr/local/corex/include -o cap_split.so cap_split.c -ldl -lpthread

# 跑
LD_PRELOAD=/abs/path/cap_split.so \
SPLIT_LOG=/tmp/cap_split.log \
SPLIT_LIMIT=118 \
python your_train.py        # 或 ./your_binary
```

`N ≤ 118` 时这层等于透明，行为与不加完全一致（实测 K=100 的 capture+replay 耗时与原生一致）；
`N > 118` 时它替你把图切开，绕过 128 上限。

## 3. 确认真的用上了（必看日志）

```
SPLIT_BEGIN  stream=0x...  mode=1  limit=118
SPLIT_SEAL   idx=0 nodes=118 counted=118   ← 切段发生在 capture 中途
SPLIT_CHUNK_START idx=1                     ← 自动重开 capture
SPLIT_SEAL   idx=1 nodes=118 counted=118
SPLIT_END    chunks=3 nodes0=118 post_bad=0 graph=0x...
SPLIT_INST_BIND chunks=3 exec=0x...         ← Instantiate 认到了这张图
SPLIT_SUMMARY record=300 seal=3 bind=1 launch=20
```

- `chunks=1` → 没切，说明你的图本来就 ≤118，这层只是透传。
- `record` = capture 期数到的可记录调用数；**`record=0` 说明压根没进这层**，graph 也就没被接管。
- `bind=0` 而 `chunks>0` → Instantiate 没走到影子入口，Launch 只会跑第 0 段（数据会缺，务必看这个）。
- 校验数据别只看最终缓冲：捕获的节点会按段顺序重放，用"第 i 个节点向 `buf+i*4` 写 i、回放后逐槽比对"的方式验。

## 4. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SPLIT_LOG` | `/tmp/cap_split.log` | 日志路径 |
| `SPLIT_LIMIT` | `118` | 每段节点上限（1..128）。**别设 128**，计数有 ±1 误差，留余量 |
| `SPLIT_TRACE` | 关 | 逐段打 launch 点，崩的时候定位崩在第几段 |
| `SPLIT_USE_NODES` | 关 | 用 `cuGraphGetNodes` 复核真实节点数（调试用，见第 6 节第 1 条） |
| `SPLIT_KEEP` | 关 | 各段 exec 常驻。**会崩**，只用于复现已知问题 |

## 5. 用之前得知道的语义变化

1. **一次 `Launch` 从"异步提交"变成"提交到完成"**。段间要 `cuStreamSynchronize`，因为你不能同时
   让两个 exec 活着跑（见第 6 节第 2 条）。所以 replay 返回时，前 n-1 段已经跑完了。
2. **段边界的同步是固定开销**（本机约 1.5 ms/次）。实测 300 节点切成 3 段：
   CPU 提交从 `0.752 µs/节点` 降到 `0.45 µs/节点`（−40%），但整轮 1848 µs → 3237 µs。
   → **每段 GPU 工作量远大于一次同步时才净赚**；图很轻又必须切段，整轮会更慢。
3. **每轮 `Launch` 都要现场 instantiate 中间段**（`instantiate → launch → 同步 → 销毁`），
   这个开销也在每轮里，别把它算漏。
4. capture 期只能提交**被记录的 API**（kernel launch、`memcpy*Async`、`memset*Async`、`eventRecord`、
   `streamWaitEvent`）。用了没覆盖的入口，计数会偏少，段就可能超 128 而崩。

## 6. 复用时必须记住的几条（都是实测踩出来的）

1. `cuGraphGetNodes` 会破坏调用方寄存器里存活的值，所以默认不调它，节点数靠自己计数。
2. 同 context 两个 exec 共存超过 64 节点，launch 第二段必崩 → 只能串行，一次一个额外 exec。
3. **临时建的段 exec 必须先同步再销毁**：`launch → destroy` 连着写会挂死（`exit=124`，永不返回）。
4. 单流、顺序捕获才支持；嵌套捕获会打 `SPLIT_UNSUPPORTED` 原样放行（此时回到厂商 128 上限，大概率崩）。
5. `cuMemcpyDtoHAsync` 是 `(dstHost, srcDevice, size, stream)`，参数写反表现为"rc 全 0、数据全 0"。

## 7. 自检

```bash
gcc -O2 -o test2s tests/test2s.c -ldl
./test2s 0                  # 打印 TEST2S_DONE 才是好卡
```

地址级依据、行为学分类、历史作业溯源在 `../../docs/`（`COREX-GRAPH-PATH.md`、`CAP-SPLIT.md`、`GRAPH-CHANNEL.md`）。
