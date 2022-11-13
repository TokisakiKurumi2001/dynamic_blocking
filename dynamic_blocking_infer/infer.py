from transformers import T5Tokenizer, T5ForConditionalGeneration, LogitsProcessorList
from DBLogitsProcessor import DBLogitsProcessor
import pandas as pd
import re
import torch

class DBInference:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("model")
        self.model = T5ForConditionalGeneration.from_pretrained("model").to(device=torch.device('cuda:1'))

    def __normalize_output(self, sent: str) -> str:
        """
        Lowercase and capitalize the sentence
        """
        return sent.lower().capitalize()

    def __call__(self, file_name: str):
        with open(file_name, "r") as file:
            outs = []
            ins = []
            limit = 10000
            for i, line in enumerate(file):
                if i % limit == 0:
                    if len(ins) != 0:
                        df = pd.DataFrame({"Input": ins, "Output": outs})
                        df.to_csv(f"infer_result/segment_{int(i/limit)}.csv", index=False)
                        ins = []
                        outs = []
            
                line = re.sub("\n", "", line)
                input_ids = self.tokenizer(line, return_tensors="pt").input_ids
                input_ids = input_ids.to(device=torch.device('cuda:1'))
                outputs = self.model.generate(
                    input_ids=input_ids,
                    num_beams=4,
                    logits_processor=LogitsProcessorList(
                        [DBLogitsProcessor(input_ids, threshold_p=0.3)
                    ]))
                output = self.__normalize_output(self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
                ins.append(line)
                outs.append(output)
            if len(ins) != 0:
                df = pd.DataFrame({"Input": ins, "Output": outs})
                df.to_csv(f"infer_result/segment_{int(i/limit)+1}.csv", index=False)

if __name__ == "__main__":
    db_inference = DBInference()
    db_inference("data_vi.txt")