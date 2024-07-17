"""
This script defines a PromptTemplate class that assists in generating 
conversation/prompt templates. The script facilitates formatting prompts 
for inference and training by combining various context elements and user inputs.
"""

import dataclasses 
from typing import Dict, List, Union

@dataclasses.dataclass
class PromtTemplete:
    """A class that manages prompt templates"""
    
    # The name of this templete
    name: str
    # The templete of the system prompt
    system_templete: str = "{system_message}"
    # The templete for the system context
    context_templete: str = "{user_context}\n{news_context}"
    # The templete for the conversastion history
    chat_history_templete: str = "{chat_history}"
    # The templete of the user question
    question_templete: str = "{question}"
    # The templete of the system answer
    answer_templete: str = "{answer}"
    # The sytem message
    system_message: str = ""
    # Separator
    sep: str = "\n"
    eos: str = ""
    
    @property
    def input_varabels(self) -> List[str]:
        return ["user_context", "news_context", "chat_history", "question", "answer"]
    
    @property
    def train_raw_templete(self):
        system = self.system_templete.format(system_message=self.system_message)
        context = f"{self.sep}{self.context_templete}"
        chat_history = f"{self.sep}{self.chat_history_templete}"
        question = f"{self.sep}{self.question_templete}"
        answer = f"{self.sep}{self.answer_templete}"
        
        return f"{system}{context}{chat_history}{question}{answer}"
    
    @property
    def infer_raw_templete(self):
        system = self.system_templete.format(system_message=self.system_message)
        context = f"{self.sep}{self.context_templete}"
        chat_history = f"{self.sep}{self.chat_history_templete}"
        question = f"{self.sep}{self.question_templete}"
        return f"{system}{context}{chat_history}{question}"
    
    def format_train(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        
        promt = self.trian_raw_templete.format(
            user_context=sample["user_context"],
            news_context=sample["news_context"],
            chat_history=sample.get("chat_history", ""),
            question=sample["question"],
            answer=sample["answer"],
        )
        return {"promt": promt, "payload": sample}
    
    def format_infer(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        prompt = self.infer_raw_template.format(
            user_context=sample["user_context"],
            news_context=sample["news_context"],
            chat_history=sample.get("chat_history", ""),
            question=sample["question"],
        )
        return {"prompt": prompt, "payload": sample}
    
# Global Templete registry
templetes: Dict[str, PromtTemplete] = {}

def register_llm_templete(templete: PromtTemplete):
    """Register a new template to the global templates registry"""
    templetes[templete.name] = templete


def get_llm_templete(name: str) -> PromtTemplete:
    """Returns the templete assigned to the given name"""
    return templetes[name]

#### Register Templetes ####
# - FALCON (spec: https://huggingface.co/tiiuae/falcon-7b/blob/main/tokenizer.json)
register_llm_templete(
    PromtTemplete(
        name="falcon",
        system_template=">>INTRODUCTION<< {system_message}",
        system_message="You are a helpful assistant, with financial expertise.",
        context_template=">>DOMAIN<< {user_context}\n{news_context}",
        chat_history_template=">>SUMMARY<< {chat_history}",
        question_template=">>QUESTION<< {question}",
        answer_template=">>ANSWER<< {answer}",
        sep="\n",
        eos="<|endoftext|>",
    )
)
    
    