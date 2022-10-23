from transformers import BartTokenizer, BartForConditionalGeneration, LogitsProcessorList
from DBLogitsProcessor import DBLogitsProcessor
import pandas as pd
import re

class DBInference:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained("model/v2")
        self.model = BartForConditionalGeneration.from_pretrained("model/v2")

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
                outputs = self.model.generate(
                    input_ids=input_ids,
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
    db_infernce = DBInference()
    db_infernce("predictions.txt")