import os
import logging

__all__ = [
    "Logger"
    ]


class Logger:
    def __init__(self, log_path):
        log_name = os.path.basename(log_path)
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_name = log_name if log_name else "train.log"
        self.log_path = log_path

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 配置文件 handler
        file_handler = logging.FileHandler(self.log_path, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

        # 配置屏幕 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(log_formatter)

        # 添加 handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
    