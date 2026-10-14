table: qtable.npy | states updated at least once: 32/96 | RL disagrees with sensor rule in 6 of them

| green side | red side | green time | spillback | Q(keep) | Q(switch) | sensor rule | RL choice |
|---|---|---|---|---|---|---|---|
| jam (>0.45) | jam (>0.45) | <20 s | yes | 7.63 | 8.52 | keep | SWITCH |
| light (0.12-0.25) | jam (>0.45) | <20 s | no | 7.32 | 8.18 | keep | SWITCH |
| light (0.12-0.25) | jam (>0.45) | 20-40 s | no | 7.56 | 8.24 | keep | SWITCH |
| queue (0.25-0.45) | jam (>0.45) | <20 s | no | 7.78 | 8.32 | keep | SWITCH |
| jam (>0.45) | jam (>0.45) | >=40 s | no | 9.28 | 8.96 | SWITCH | keep |
| empty (<0.12) | light (0.12-0.25) | <20 s | no | 5.82 | 5.79 | SWITCH | keep |
