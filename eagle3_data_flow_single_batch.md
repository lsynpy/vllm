# Eagle Data Flow for Single Batch

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
TARGET_forward([220, 220, 279, 279])
  -> hidden_states<[4, 2048]>, aux_hidden_states<3x[4, 2048]>
select using logits_indices<[0, 1, 2, 3]>: hidden_states[0, 1, 2, 3] -> sampled_hidden_states<[4, 2048]> -> logits<[4, 151936]>
rejection_sample(logits<[4, 151936]>) -> [[17, -1, -1, -1]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[4, 6144]>) -> hidden_states<[4, 2048]>
  DRAFT_forward([17, 279, 279, 10250], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  select using last_token_indices<[0]>:
    last_hidden_states[0] -> sampled_hidden_states<[1, 2048]> -> logits<[1, 151936]>
    hidden_states[0] -> hidden_states<[1, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[1, 151936]>) -> [11]
  ····················
  DRAFT_forward([11], hidden_states<[1, 2048]>) -> last_hidden_states<[1, 2048]>, hidden_states<[1, 2048]>
  last_hidden_states<[1, 2048]> -> logits<[1, 151936]>
  greedy sample: argmax(logits<[1, 151936]>) -> [279]
  ····················
  DRAFT_forward([279], hidden_states<[1, 2048]>) -> last_hidden_states<[1, 2048]>, hidden_states<[1, 2048]>
  last_hidden_states<[1, 2048]> -> logits<[1, 151936]>
  greedy sample: argmax(logits<[1, 151936]>) -> [279]
propose DRAFT tokens: [[11, 279, 279]]
append [17] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220, 17]
set req-0.spec_token_ids to [11, 279, 279]
```

## iter 3

```cpp
TARGET_forward([17, 11, 279, 279])
  -> hidden_states<[4, 2048]>, aux_hidden_states<3x[4, 2048]>
select using logits_indices<[0, 1, 2, 3]>: hidden_states[0, 1, 2, 3] -> sampled_hidden_states<[4, 2048]> -> logits<[4, 151936]>
rejection_sample(logits<[4, 151936]>) -> [[11, 220, -1, -1]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[4, 6144]>) -> hidden_states<[4, 2048]>
  DRAFT_forward([11, 220, 279, 10250], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  select using last_token_indices<[1]>:
    last_hidden_states[1] -> sampled_hidden_states<[1, 2048]> -> logits<[1, 151936]>
    hidden_states[1] -> hidden_states<[1, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[1, 151936]>) -> [18]
  ····················
  DRAFT_forward([18], hidden_states<[1, 2048]>) -> last_hidden_states<[1, 2048]>, hidden_states<[1, 2048]>
  last_hidden_states<[1, 2048]> -> logits<[1, 151936]>
  greedy sample: argmax(logits<[1, 151936]>) -> [11]
  ····················
  DRAFT_forward([11], hidden_states<[1, 2048]>) -> last_hidden_states<[1, 2048]>, hidden_states<[1, 2048]>
  last_hidden_states<[1, 2048]> -> logits<[1, 151936]>
  greedy sample: argmax(logits<[1, 151936]>) -> [220]
propose DRAFT tokens: [[18, 11, 220]]
append [11, 220] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220, 17, 11, 220]
set req-0.spec_token_ids to [18, 11, 220]
```

## iter 4

```cpp
TARGET_forward([220, 18, 11, 220])
  -> hidden_states<[4, 2048]>, aux_hidden_states<3x[4, 2048]>
select using logits_indices<[0, 1, 2, 3]>: hidden_states[0, 1, 2, 3] -> sampled_hidden_states<[4, 2048]> -> logits<[4, 151936]>
rejection_sample(logits<[4, 151936]>) -> [[18, 11, 220, 20]]
Do 3 DRAFT forward passes:
  ····················
  combine_hidden_states(aux_hidden_states<[4, 6144]>) -> hidden_states<[4, 2048]>
  DRAFT_forward([18, 11, 220, 20], hidden_states<[4, 2048]>) -> last_hidden_states<[4, 2048]>, hidden_states<[4, 2048]>
  select using last_token_indices<[3]>:
    last_hidden_states[3] -> sampled_hidden_states<[1, 2048]> -> logits<[1, 151936]>
    hidden_states[3] -> hidden_states<[1, 2048]> (for next draft forward input)
  greedy sample: argmax(logits<[1, 151936]>) -> [11]
  ····················
  DRAFT_forward([11], hidden_states<[1, 2048]>) -> last_hidden_states<[1, 2048]>, hidden_states<[1, 2048]>
  last_hidden_states<[1, 2048]> -> logits<[1, 151936]>
  greedy sample: argmax(logits<[1, 151936]>) -> [220]
  ····················
  DRAFT_forward([220], hidden_states<[1, 2048]>) -> last_hidden_states<[1, 2048]>, hidden_states<[1, 2048]>
  last_hidden_states<[1, 2048]> -> logits<[1, 151936]>
  greedy sample: argmax(logits<[1, 151936]>) -> [220]
propose DRAFT tokens: [[11, 220, 220]]
append [18, 11, 220, 20] to req-0 -> [852, 279, 1156, 5779, 10250, 5109, 25, 220, 17, 11, 220, 18, 11, 220, 20]
```
