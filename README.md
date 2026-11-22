# corex-graph

一句话：让天数智芯天垓 BI-V100 能跑起来 cuda graph，并尽可能减少 kernel launch 开销
## 目录

```
src/
  record_replay.c        唯一的主程序：截 launch/copy 入口，记录节点表，回放时逐条重发
  record_replay.map       导出符号白名单（LD_PRELOAD 用）
  README.md              本文
  tests/                 作业入口、bench、离线分析、符号表
```

`tests/` 里只有三类东西值得留：

| 类别 | 文件 | 用途 |
|---|---|---|
| 回放 bench | `bench_rec6.py`（`bench_rec*.py` 是它的迭代前身） | 输出 `R6_EAGER` / `R6_REC nodes=` / `R6_RP1 SAME=` / `R6_EAGER_MS` / `R6_RP_MS` |
| 集群入口 | `run_rec3_sbx.sh`、`run_job_dump.sh`、`run_job_dump43.sh`、`run_job_v43.sh`、`run_job_v44.sh`、`run_job_v44d.sh`、`cardprobe.py` | 提交作业、落 IR、复现 631→502→481 |
| 离线分析 | `analyze_ir.py` + `syms_ops.txt`(v42) / `syms_v43.txt`(v43) | 用 `CR_MAP` 还原 kernel 名，按"纯搬运"判据清点节点 |

连同它们的文档一起退到了 `../../.retired/`，清单在 `../../.retired/RETIRED.md`。**那里不是结论源。**

## 编和挂

```bash
gcc -shared -fPIC -O2 -I/usr/local/corex/include -o record_replay.so record_replay.c \
    -ldl -lpthread -Wl,--version-script=record_replay.map

LD_PRELOAD=/abs/path/record_replay.so CAPREC_LOG=/tmp/caprec.log python your_decode.py
```

## 环境变量

| 变量 | 作用 |
|---|---|
| `CAPREC_LOG` | 日志路径 |
| `CAPREC_CHUNK` | 回放栅栏粒度。**256 是已验证稳定配方**（3 道栅栏、连跑 3 次全 `SAME=True`）；0 会算错，2 道栅栏不稳 |
| `CAPREC_NOREPLAY=1` | 只录不放：数节点、落 IR，不碰回放风险 |
| `CAPREC_DUMP=1` | 落 `CR_DUMP` / `CR_HASH` / `CR_PTR` / `CR_MAP` 逐节点明细（图优化分析靠它） |
| `CAPREC_TRACE`、`CAPREC_SYNC_EACH` | 逐节点打点 / 每节点同步，只在定位崩点时开，会毁掉步耗 |
| `CAPREC_RELOC`、`CAPREC_RESTORE` | 重定位与状态恢复开关，回放要重入时才需要 |

## 看日志

```
CR_INSTALL launch=0x... cpy=0x... set=0x... sync=0x...
CR_BEGIN
CR_END   nodes=481 arena=...
CR_RP_SKIPPED n=481            ← CAPREC_NOREPLAY=1 的预期表现
```

判读要点：

- **`CR_INSTALL` 之后没有 `CR_BEGIN`** → 压根没进这层（`LD_PRELOAD` 或版本没生效），不是"图太短"。
- **`nodes=` 与上一步对不上** → 先看是不是 `CAPREC_NOREPLAY` / `CAPREC_CHUNK` 换了。
- `CR_DUMP` 的 `f=` 必须配对应版本的符号表解析（v42 用 `syms_ops.txt`、v43 用 `syms_v43.txt`），
  **用错表会把所有 kernel 认成别的名字**。
- 回放是否可信只看两件事：token 逐位（`R6_RP1 SAME=True`）和帧级粒度（`CAPREC_CHUNK`）。
