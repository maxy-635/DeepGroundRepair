import os
from utils.chat_log import ChatLog
from utils.match_and_save_code import MatchSaveCode


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
        chatloger.save_chat_log(self.count_num)

        if not os.path.exists(self.save_code_path):
            os.makedirs(self.save_code_path)

        match_saver = MatchSaveCode(target_folder=self.save_code_path, response=response)
        matched_code = match_saver.match_code()
        match_saver.save_code(matched_code, "generated_code.py")

        return matched_code
