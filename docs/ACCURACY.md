# 精度基准

这里只有一个数字口径：**我们自己合成的确定性片段**。它不是对 Klangio、Songscription、AnthemScore 或 Basic Pitch 的同口径对比，也不该被读成那种对比。真实的对手分数来自各自的私有测试集，我们既拿不到它们的音表，也没有 MIDI 渲染器（本机只有 `ffmpeg`，没有 fluidsynth/timidity 和 soundfont），所以无法在同一 ground truth 上给它们打分。任何人复现下面的数字都不需要下载音频。

## 口径

`scripts/bench_synth.py` 用固定种子生成一段钢琴音频，生成器自己持有音表，那就是 ground truth：

```bash
PYTHONPATH=src .venv/bin/python scripts/bench_synth.py --seed 1 --out bench/clip1.wav
```

- 21.0 秒，`bpm=96`，4/4，A 小调音阶
- 117 个音：右手 54（每八分音符一个，12% 休止）、左手 63（每小节第 1、3 拍块状和弦）
- 右手锁在 MIDI `[60,79]`，左手在 `45±`，都落在三档的骨架音域内
- 音表写在同目录的 `clip1.truth.json`

打分：

```bash
PYTHONPATH=src .venv/bin/python scripts/note_f1.py bench/clip1.truth.json out/clip1-easy.arr.pkl --tolerance 0.30
PYTHONPATH=src .venv/bin/python scripts/note_f1.py bench/clip1.truth.json out/clip1-easy.arr.pkl --tolerance 0.30 --role rh
```

一个预测音命中，当且仅当：音高**完全相同**、onset 相距 ≤ 0.30 秒、贪心一对一。0.30 秒不是宽松：分离加转写本身的 onset 抖动就在这个量级，压到 0.05 秒测的已经不是排谱而是强迫对齐。音级（pitch class）版本同样跑一份，音高差一个八度时它不会扣分——八度错误是我们已知的短板（见下），所以两个数都要看，不能只报高的那个。

预测音的音高要先减掉 `arr.meta['transpose']`，因为成品是移调到档位调号之后的结果，而真值在源调。

## 当前数字

`--tolerance 0.30`，`easy` 档，21 秒分离口径：

| 指标 | 数值 |
| --- | --- |
| note-F1（全体） | **0.5604**（precision 0.6444 / recall 0.4957，命中 58/117，预测 90） |
| pitch-class F1 | **0.6667** |
| note-F1（右手） | 0.6538 |
| note-F1（左手） | 0.4660 |

三档 `easy`/`medium`/`hard` 的数字**完全相同**。这是设计，不是巧合：三档只差 `global.target_key` 与 `global.allowed_black_keys`，调号与黑键数量变了，音高集合和节奏不变。`transpose` 分别是 `+2`、`+5`、`-2`，黑键数 `15/50`（右手）、`10/40`（左手）。任何一档的数字和其它档对不上，就是档位同构被破坏。

## 演进

每一行都是同一个片段、同一份真值、同一个容差，只改了一处代码：

| 阶段 | note-F1 | pc-F1 | 改了什么 |
| --- | --- | --- | --- |
| 混音口径（作废） | 0.1441 | 0.3136 | 没做人声分离，口径不对，不再引用 |
| 分离口径基线 | 0.3448 | 0.4368 | `--no-separate` 去掉后的真实起点 |
| 补 onset 切分 | 0.3892 | 0.4973 | `cqt_melody` 接上 `_cut_at_onsets`，旋律音数 22 → 59 |
| 去重窗口收紧到槽 | 0.4925 | 0.5829 | `_dedupe_beats` 从「每拍一个」改成「每 16 分槽一个」 |
| 合并窗口收紧到槽 | **0.5604** | **0.6667** | `_collapse_beat_onsets` 同样从拍改到槽，右手 note-F1 0.2927 → 0.6538 |

## 已知短板

- **八度错误**。`cqt_melody` 的谐波累加方向反了：`sal[: n_bins - shift] += w * C[shift:]` 把高 bin 的能量加到低 bin，造成系统性低八度偏置。半音差直方图上 `-12` 出现 5 次。pc-F1 明显高于 note-F1 的一部分原因就在这里。
- **速度检测双向误判**。真值 96，`track_beats` 报 191.41、`analyse` 报 47.85。强制成 96 后 note-F1 只从 0.5604 到 0.5714，不是主因，但页面标题会印出 `bpm=48`，是可见的错。
- **左手偏弱**。左手 note-F1 0.4660，低于右手 0.6538。左手是折叠出来的，真值里的块状和弦本来就不该被逐音还原，这个数字的含义要弱于右手。
- **没有 MIDI 渲染器**，所以无法在别人的测试集上算同口径分数。

改动任何一环，上面的表都要重跑；数字掉了就是掉了，不要换容差、不要换口径、不要只报 pc-F1。
