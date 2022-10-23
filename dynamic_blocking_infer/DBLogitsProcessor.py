from transformers import LogitsProcessor, LogitsProcessorList
import torch
import random

random.seed(1)

class DBLogitsProcessor(LogitsProcessor):
  def __init__(self, input_ids, threshold_p = 0.3):
    super(DBLogitsProcessor, self).__init__()
    tokens = input_ids[0].tolist()
    self.block_table = {}
    bos_token = 0
    eos_token = 2
    for i, token in enumerate(tokens):
      p = random.random()
      if p < threshold_p:
        if i == 0:
          self.block_table[bos_token] = token
        else:
          self.block_table[tokens[i-1]] = token
    p = random.random()
    if p < threshold_p:
      self.block_table[token] = eos_token
    # print(self.block_table)
  
  def __call__(self, input_ids, logits):
    for beam_idx in range(4):
      for k, v in self.block_table.items():
        if input_ids[beam_idx, -1] == k:
          # print("Block")
          logits[beam_idx, v] = -float('inf')
          # id = torch.argmax(logits, dim=-1)
          # print(f"Next token id will be {id[beam_idx]}")
    # print(input_ids[:, -3:-1])
    # id = torch.argmax(logits, dim=-1)
    # print(f"Next token id will be {id}")
    return logits