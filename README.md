# graph —— 给自己程序接上 CUDA Graph 自动切段

这个仓库是给 Iluvatar CoreX（BI-V100，CoreX 3.2.1）用的一个 `LD_PRELOAD` 补丁：
厂商驱动单张图最多 128 个节点，超了要么拦、要么崩。它在你程序无感知的情况下把一张大图
在 capture 期自动切成多段（每段 ≤118 节点），`Instantiate`/`Launch`/`Destroy` 照原样调，
内部按段串行 replay。

接入不改一行代码，只加环境变量。

---

## 1. 编译（计算节点上，`cuda.h` 在 `/usr/local/corex/include`）

```bash
gcc -shared -fPIC -O2 -I/usr/local/corex/include -o cap_split.so cap_split.c -ldl -lpthread
```

产物是一个 `.so`，和你的程序放一起就行。

## 2. 接上

```bash
LD_PRELOAD=/abs/path/cap_split.so \
SPLIT_LOG=/tmp/cap_split.log \
SPLIT_LIMIT=118 \
python your_train.py        # 或 ./your_binary
```

就这三行。你的程序里 `cudaStreamBeginCapture / cudaStreamEndCapture / cudaGraphInstantiate /
cudaGraphLaunch`（或对应的 `cu*` 驱动接口）都不用改。

## 3. 确认真的接上了（必看）

跑完 `grep` 一下日志，出现这几行就是生效了：

```
SPLIT_BEGIN  stream=0x...  mode=1  limit=118
SPLIT_SEAL   idx=0 nodes=118 counted=118     ← 切了第一段
SPLIT_CHUNK_START idx=1                       ← 自动重开 capture
SPLIT_SEAL   idx=1 nodes=118 counted=118
SPLIT_END    chunks=3 nodes0=118 post_bad=0 graph=0x...
SPLIT_INST_BIND chunks=3 exec=0x...           ← Instantiate 认到了这张图
SPLIT_SUMMARY record=300 seal=3 bind=1 launch=20
```

判读方式：

- `chunks` = 这次捕获被切成几段，`nodes` 每段必须 ≤`SPLIT_LIMIT`。
- `record` = capture 期间被数到的可记录调用数；`record=0` 说明**根本没进这层**（见下面第 5 节的排查第一条）。
- `bind=0` 但 `chunks>0`：Instantiate 没走到影子入口，`Launch` 时会退化成只跑第 0 段。
- 最后跑一次健康检查程序，确认卡没被搞挂。

单段（节点数 ≤118）的情况日志里 `chunks=1`，性能与不加这个 `.so` 完全一致。

## 4. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SPLIT_LOG` | `/tmp/cap_split.log` | 日志路径 |
| `SPLIT_LIMIT` | `118` | 每段节点上限（1..128）。**别设 128**，计数有 ±1 误差，留余量 |
| `SPLIT_TRACE` | 关 | 逐段打 launch 点，崩的时候用它定位崩在第几段 |
| `SPLIT_USE_NODES` | 关 | 用驱动的 `cuGraphGetNodes` 复核每段真实节点数（只调试用，见第 6 节第 1 条） |
| `SPLIT_KEEP` | 关 | 各段 exec 常驻。**会崩**，只用于复现已知问题，不要在生产开 |

## 5. 没生效怎么查

1. `record=0` → 你的程序走的是**没被影子到的那条入口**。这层同时截 `libcuda` 和
   `libcudart` 两条（`cudaStreamBeginCapture` 等 11 个 runtime 入口已覆盖）。如果你用的是
   `dlsym(handle, "cuGraphLaunch")` 直接取函数指针、自己封装一层，就会绕过。
   先 `SPLIT_LOG` 看有没有 `SPLIT_RT_INSTALL` / `SPLIT_INSTALL` 这两行，没有就是没加载上。
2. 有 `SPLIT_LOAD` 但没有 `SPLIT_BEGIN` → 你的捕获不是从被影子的 `Begin` 开始的
   （比如用了 `cudaStreamBeginCaptureToGraph` 之类没覆盖的入口）。
3. `SPLIT_UNSUPPORTED nested begin` → 你在同线程里嵌套捕获，这层不支持，会原样放行。

## 6. 复用时必须记住的几条（都是实测踩出来的）

1. `cuGraphGetNodes` 会破坏调用方寄存器里存活的值，所以默认不调它，节点数靠自己计数。
   要复核就开 `SPLIT_USE_NODES=1`，并且只做对照，别放进生产路径。
2. 同 context 两个 exec 共存超过 64 节点，launch 第二段必崩。所以第 0 段用你自己的 exec，
   其余段是 `instantiate → launch → 同步 → 销毁`，一次只有一个额外 exec 活着。
3. **临时建的段 exec 必须先同步再销毁**。`launch → destroy` 直接连着写会挂死（`exit=124`，永不返回）。
4. 因为第 2 条，一次 `Launch` 的语义变成了“提交到完成”（段间有 `cuStreamSynchronize`）。
   每段 GPU 工作量远大于一次同步开销时才是净赚；图很轻又必须切段时整轮会更慢。
5. `cuMemcpyDtoHAsync` 是 `(dstHost, srcDevice, size, stream)`，参数写反表现为“rc 全 0、数据全 0”。

## 7. 自检（可选）

仓库带了几支裸驱动的小程序，用来在目标机上确认这套东西成立：

```bash
gcc -O2 -o test2s ../src/tests/test2s.c -ldl
./test2s 0                   # 探活：打印 TEST2S_DONE 就是好卡
```

历史作业与地址级依据在 `../../docs/`（`COREX-GRAPH-PATH.md`、`CAP-SPLIT.md`、`GRAPH-CHANNEL.md`）。
