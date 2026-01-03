# Eagle data flow

## iter 1

```cpp
TARGET_forward([852, 279, 1156, 5779, 10250, 5109, 25])
  -> hidden_states<[7, 2048]>, aux_hidden_states<3x[7, 2048]>
select using logits_indices<[6]>: hidden_states[6] -> sampled_hidden_states<[1, 2048]> -> logits<[1, 151936]>
normal_sample(logits<[1, 151936]>) -> [[220]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[7, 6144]>) -> hidden_states<[7, 2048]>
  DRAFT_forward([279, 1156, 5779, 10250, 5109, 25, 220], hidden_states<[7, 2048]>) -> last_hidden_states<[7, 2048]>, hidden_states<[7, 2048]>
  select using last_token_indices<[6]>:
    last_hidden_states[6] -> sampled_hidden_states<[1, 2048]> -> logits<[1, 151936]>
    hidden_states[6] -> hidden_states<[1, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[1, 151936]>) -> [220]
  ····················
  DRAFT_forward([220], hidden_states<[1, 2048]>) -> last_hidden_states<[1, 2048]>, hidden_states<[1, 2048]>
  last_hidden_states<[1, 2048]> -> logits<[1, 151936]>
  greedy sample: argmax(logits<[1, 151936]>) -> [279]
  ····················
  DRAFT_forward([279], hidden_states<[1, 2048]>) -> last_hidden_states<[1, 2048]>, hidden_states<[1, 2048]>
  last_hidden_states<[1, 2048]> -> logits<[1, 151936]>
  greedy sample: argmax(logits<[1, 151936]>) -> [279]
propose DRAFT tokens: [[220, 279, 279]]
append [220] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220]
set req-0.spec_token_ids to [220, 279, 279]
```

## iter 2

```cpp
TARGET_forward([220, 220, 279, 279, 785, 6722, 315, 9625, 374, 12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25])
  -> hidden_states<[32, 2048]>, aux_hidden_states<3x[32, 2048]>
select using logits_indices<[0, 1, 2, 3, 8, 20, 31]>: hidden_states[0, 1, 2, 3, 8, 20, 31] -> sampled_hidden_states<[7, 2048]> -> logits<[7, 151936]>
rejection_sample(logits<[7, 151936]>) -> [[17, -1, -1, -1], [12095, -1, -1, -1], [1052, -1, -1, -1], [220, -1, -1, -1]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[32, 6144]>) -> hidden_states<[32, 2048]>
  DRAFT_forward([17, 279, 279, 785, 6722, 315, 9625, 374, 12095, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220], hidden_states<[32, 2048]>) -> last_hidden_states<[32, 2048]>, hidden_states<[32, 2048]>
states<[32, 2048]>
  select using last_token_indices<[0, 8, 20, 31]>:
    last_hidden_states[0, 8, 20, 31] -> sampled_hidden_states<[4, 2048]> -> logits<[4, 151936]>
    hidden_states[0, 8, 20, 31] -> hidden_states<[4, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[4, 151936]>) -> [11, 374, 1052, 16]
  ····················
  DRAFT_forward([11, 374, 1052, 16], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  last_hidden_states<[4, 2048]> -> logits<[4, 151936]>
  greedy sample: argmax(logits<[4, 151936]>) -> [279, 12095, 1052, 15]
  ····················
  DRAFT_forward([279, 12095, 1052, 15], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  last_hidden_states<[4, 2048]> -> logits<[4, 151936]>
  greedy sample: argmax(logits<[4, 151936]>) -> [279, 12095, 11, 271]
propose DRAFT tokens: [[11, 279, 279], [374, 12095, 12095], [1052, 1052, 11], [16, 15, 271]]
append [17] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220, 17]
append [12095] to req-1 -> [785, 6722, 315, 9625, 374, 12095]
append [1052] to req-2 -> [12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052]
append [220] to req-3 -> [852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220]
set req-0.spec_token_ids to [11, 279, 279]
set req-1.spec_token_ids to [374, 12095, 12095]
set req-2.spec_token_ids to [1052, 1052, 11]
set req-3.spec_token_ids to [16, 15, 271]
```

## iter 3

```cpp
TARGET_forward([17, 11, 279, 279, 12095, 374, 12095, 12095, 1052, 1052, 1052, 11, 220, 16, 15, 271])
  -> hidden_states<[16, 2048]>, aux_hidden_states<3x[16, 2048]>
select using logits_indices<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: hidden_states[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] -> sampled_hidden_states<[16, 2048]> -> logits<[16, 151936]>
rejection_sample(logits<[16, 151936]>) -> [[11, 220, -1, -1], [13, -1, -1, -1], [572, -1, -1, -1], [16, 11, -1, -1]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[16, 6144]>) -> hidden_states<[16, 2048]>
  DRAFT_forward([11, 220, 279, 12095, 13, 12095, 12095, 1052, 572, 1052, 11, 220, 16, 11, 271, 3041], hidden_states<[16, 2048]>) -> last_hidden_states<[16, 2048]>, hidden_states<[16, 2048]>
  select using last_token_indices<[1, 4, 8, 13]>:
    last_hidden_states[1, 4, 8, 13] -> sampled_hidden_states<[4, 2048]> -> logits<[4, 151936]>
    hidden_states[1, 4, 8, 13] -> hidden_states<[4, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[4, 151936]>) -> [220, 9625, 264, 220]
  ····················
  DRAFT_forward([220, 9625, 264, 220], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  last_hidden_states<[4, 2048]> -> logits<[4, 151936]>
  greedy sample: argmax(logits<[4, 151936]>) -> [18, 13, 264, 16]
  ····················
  DRAFT_forward([18, 13, 264, 16], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  last_hidden_states<[4, 2048]> -> logits<[4, 151936]>
  greedy sample: argmax(logits<[4, 151936]>) -> [11, 12095, 1052, 11]
propose DRAFT tokens: [[220, 18, 11], [9625, 13, 12095], [264, 264, 1052], [220, 16, 11]]
append [11, 220] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220, 17, 11, 220]
append [13] to req-1 -> [785, 6722, 315, 9625, 374, 12095, 13]
append [572] to req-2 -> [12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052, 572]
append [16, 11] to req-3 -> [852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11]
set req-0.spec_token_ids to [220, 18, 11]
set req-1.spec_token_ids to [9625, 13, 12095]
set req-2.spec_token_ids to [264, 264, 1052]
set req-3.spec_token_ids to [220, 16, 11]
```

## iter 4

```cpp
TARGET_forward([220, 220, 18, 11, 13, 9625, 13, 12095, 572, 264, 264, 1052, 11, 220, 16, 11])
  -> hidden_states<[16, 2048]>, aux_hidden_states<3x[16, 2048]>
select using logits_indices<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: hidden_states[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] -> sampled_hidden_states<[16, 2048]> -> logits<[16, 151936]>
rejection_sample(logits<[16, 151936]>) -> [[18, -1, -1, -1], [576, -1, -1, -1], [264, 3908, -1, -1], [220, 16, 16, -1]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[16, 6144]>) -> hidden_states<[16, 2048]>
  DRAFT_forward([18, 18, 11, 13, 576, 13, 12095, 572, 264, 3908, 1052, 11, 220, 16, 16, 3041], hidden_states<[16, 2048]>) -> last_hidden_states<[16, 2048]>, hidden_states<[16, 2048]>
  select using last_token_indices<[0, 4, 9, 14]>:
    last_hidden_states[0, 4, 9, 14] -> sampled_hidden_states<[4, 2048]> -> logits<[4, 151936]>
    hidden_states[0, 4, 9, 14] -> hidden_states<[4, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[4, 151936]>) -> [11, 374, 3908, 11]
  ····················
  DRAFT_forward([11, 374, 3908, 11], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  last_hidden_states<[4, 2048]> -> logits<[4, 151936]>
  greedy sample: argmax(logits<[4, 151936]>) -> [11, 374, 5193, 220]
  ····················
  DRAFT_forward([11, 374, 5193, 220], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  last_hidden_states<[4, 2048]> -> logits<[4, 151936]>
  greedy sample: argmax(logits<[4, 151936]>) -> [220, 304, 264, 220]
propose DRAFT tokens: [[11, 11, 220], [374, 374, 304], [3908, 5193, 264], [11, 220, 220]]
append [18] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220, 17, 11, 220, 18]
append [576] to req-1 -> [785, 6722, 315, 9625, 374, 12095, 13, 576]
append [264, 3908] to req-2 -> [12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052, 572, 264, 3908]
append [220, 16, 16] to req-3 -> [852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16]
set req-0.spec_token_ids to [11, 11, 220]
set req-1.spec_token_ids to [374, 374, 304]
set req-2.spec_token_ids to [3908, 5193, 264]
set req-3.spec_token_ids to [11, 220, 220]
```

## iter 5

```cpp
TARGET_forward([18, 11, 11, 220, 576, 374, 374, 304, 3908, 3908, 5193, 264, 16, 11, 220, 220])
  -> hidden_states<[16, 2048]>, aux_hidden_states<3x[16, 2048]>
select using logits_indices<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: hidden_states[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] -> sampled_hidden_states<[16, 2048]> -> logits<[16, 151936]>
rejection_sample(logits<[16, 151936]>) -> [[11, 220, -1, -1], [6722, -1, -1, -1], [3743, -1, -1, -1], [11, 220, 16, -1]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[16, 6144]>) -> hidden_states<[16, 2048]>
  DRAFT_forward([11, 220, 220, 576, 6722, 374, 304, 3908, 3743, 5193, 264, 16, 11, 220, 16, 3041], hidden_states<[16, 2048]>) -> last_hidden_states<[16, 2048]>, hidden_states<[16, 2048]>
  select using last_token_indices<[1, 4, 8, 14]>:
    last_hidden_states[1, 4, 8, 14] -> sampled_hidden_states<[4, 2048]> -> logits<[4, 151936]>
    hidden_states[1, 4, 8, 14] -> hidden_states<[4, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[4, 151936]>) -> [220, 315, 6941, 16]
  ····················
  DRAFT_forward([220, 315, 6941, 16], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  last_hidden_states<[4, 2048]> -> logits<[4, 151936]>
  greedy sample: argmax(logits<[4, 151936]>) -> [19, 279, 6941, 16]
  ····················
  DRAFT_forward([19, 279, 6941, 16], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  last_hidden_states<[4, 2048]> -> logits<[4, 151936]>
  greedy sample: argmax(logits<[4, 151936]>) -> [11, 315, 6941, 11]
propose DRAFT tokens: [[220, 19, 11], [315, 279, 315], [6941, 6941, 6941], [16, 16, 11]]
append [11, 220] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220, 17, 11, 220, 18, 11, 220]
append [6722] to req-1 -> [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722]
append [3743] to req-2 -> [12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052, 572, 264, 3908, 3743]
append [11, 220] to req-3 -> [852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16, 11, 220]
set req-0.spec_token_ids to [220, 19, 11]
set req-1.spec_token_ids to [315, 279, 315]
set req-2.spec_token_ids to [6941, 6941, 6941]
```

## iter 6

```cpp
TARGET_forward([220, 220, 19, 11, 6722, 315, 279, 315, 3743, 6941, 6941, 6941])
  -> hidden_states<[12, 2048]>, aux_hidden_states<3x[12, 2048]>
select using logits_indices<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]>: hidden_states[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] -> sampled_hidden_states<[12, 2048]> -> logits<[12, 151936]>
rejection_sample(logits<[12, 151936]>) -> [[20, -1, -1, -1], [315, 279, 3639, -1], [6941, 444, -1, -1]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[12, 6144]>) -> hidden_states<[12, 2048]>
  DRAFT_forward([20, 19, 11, 6722, 315, 279, 3639, 3743, 6941, 444, 6941, 16], hidden_states<[12, 2048]>) -> last_hidden_states<[12, 2048]>, hidden_states<[12, 2048]>
  select using last_token_indices<[0, 6, 9]>:
    last_hidden_states[0, 6, 9] -> sampled_hidden_states<[3, 2048]> -> logits<[3, 151936]>
    hidden_states[0, 6, 9] -> hidden_states<[3, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[3, 151936]>) -> [11, 374, 444]
  ····················
  DRAFT_forward([11, 374, 444], hidden_states<[3, 2048]>) -> last_hidden_states<[3, 2048]>, hidden_states<[3, 2048]>
  last_hidden_states<[3, 2048]> -> logits<[3, 151936]>
  greedy sample: argmax(logits<[3, 151936]>) -> [220, 374, 13]
  ····················
  DRAFT_forward([220, 374, 13], hidden_states<[3, 2048]>) -> last_hidden_states<[3, 2048]>, hidden_states<[3, 2048]>
  last_hidden_states<[3, 2048]> -> logits<[3, 151936]>
  greedy sample: argmax(logits<[3, 151936]>) -> [220, 304, 13]
propose DRAFT tokens: [[11, 220, 220], [374, 374, 304], [444, 13, 13]]
append [20] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220, 17, 11, 220, 18, 11, 220, 20]
append [315, 279, 3639] to req-1 -> [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 279, 3639]
append [6941, 444] to req-2 -> [12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052, 572, 264, 3908, 3743, 6941, 444]
set req-1.spec_token_ids to [374, 374, 304]
set req-2.spec_token_ids to [444, 13, 13]
```

## iter 7

```cpp
TARGET_forward([444, 444, 13, 13, 3639, 374, 374, 304])
  -> hidden_states<[8, 2048]>, aux_hidden_states<3x[8, 2048]>
select using logits_indices<[0, 1, 2, 3, 4, 5, 6, 7]>: hidden_states[0, 1, 2, 3, 4, 5, 6, 7] -> sampled_hidden_states<[8, 2048]> -> logits<[8, 151936]>
rejection_sample(logits<[8, 151936]>) -> [[10524, -1, -1, -1], [4180, -1, -1, -1]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[8, 6144]>) -> hidden_states<[8, 2048]>
  DRAFT_forward([10524, 13, 13, 3639, 4180, 374, 304, 3743], hidden_states<[8, 2048]>) -> last_hidden_states<[8, 2048]>, hidden_states<[8, 2048]>
  select using last_token_indices<[0, 4]>:
    last_hidden_states[0, 4] -> sampled_hidden_states<[2, 2048]> -> logits<[2, 151936]>
    hidden_states[0, 4] -> hidden_states<[2, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[2, 151936]>) -> [13, 374]
  ····················
  DRAFT_forward([13, 374], hidden_states<[2, 2048]>) -> last_hidden_states<[2, 2048]>, hidden_states<[2, 2048]>
  last_hidden_states<[2, 2048]> -> logits<[2, 151936]>
  greedy sample: argmax(logits<[2, 151936]>) -> [13, 304]
  ····················
  DRAFT_forward([13, 304], hidden_states<[2, 2048]>) -> last_hidden_states<[2, 2048]>, hidden_states<[2, 2048]>
  last_hidden_states<[2, 2048]> -> logits<[2, 151936]>
  greedy sample: argmax(logits<[2, 151936]>) -> [13, 279]
propose DRAFT tokens: [[13, 13, 13], [374, 304, 279]]
append [4180] to req-1 -> [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 279, 3639, 4180]
append [10524] to req-2 -> [12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052, 572, 264, 3908, 3743, 6941, 444, 10524]
```
