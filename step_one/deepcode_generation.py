import os
from utils.chat_log import ChatLog
from utils.chatlog2code import Chatlog2Code


class DLcodeGeneration:
    """Implement the dlcode generation component."""

    def __init__(self, count_num, save_code_path, save_log_path):
        """Initialize the instance."""
        self.count_num = count_num
        self.save_code_path = save_code_path
        self.save_log_path = save_log_path

    def prompt2llm(
        self,
        model,
        tokenizer,
        prompt,
        chat2llm,
        temperature,
        top_p,
        max_new_tokens,
    ):
        """Send a prompt to the language model."""

        # get conversation from LLM
        response = chat2llm.chat(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        return response

    def save(self, prompting, response):
        # save chat logs
        if not os.path.exists(self.save_log_path):
            os.makedirs(self.save_log_path)
        chatloger = ChatLog(self.save_log_path, prompting, response)
        _, log_file = chatloger.save_chat_log(self.count_num)

        if not os.path.exists(self.save_code_path):
            os.makedirs(self.save_code_path)

        chatlog2coder = Chatlog2Code(
            log_file_path=log_file, save_code_path=self.save_code_path
        )
        matched_code = chatlog2coder.read_yamldata_to_pycode()

        return matched_code
