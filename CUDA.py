import re
import subprocess
import time

import random
import os
from enum import IntEnum
from oracle import oracle_report

import logging

logger = logging.getLogger()

main_code = """

int main(){
return 0;
}

"""


class LEVEL(IntEnum):
    INFO = 1  # most important message (are bugs found?, overview, etc)
    TRACE = 2
    VERBOSE = 3  # most verbose messages (including validation messages, etc)

class Logger:
    # logging structure breakdown into validation, and sample generation,
    # and any potential bugs are logged always in main log.txt
    # TODO: support logging levels
    def __init__(self, basedir, file_name: str, level: LEVEL = LEVEL.INFO):
        self.logfile = os.path.join(basedir, file_name)
        self.level = level

    @staticmethod
    def format_log(msg, level: LEVEL = LEVEL.VERBOSE):
        return f"[{level.name}] {msg}"

    def logo(self, msg, level: LEVEL = LEVEL.VERBOSE):
        try:
            with open(self.logfile, "a+", encoding="utf-8") as logfile:
                logfile.write(self.format_log(msg, level))
                logfile.write("\n")
            if level <= self.level:
                print(self.format_log(msg, level))
        except Exception as e:
            pass

class CudaTargetController():
    def __init__(self, **kwargs):
        self.folder = kwargs['working_dir']
        self.g_logger = Logger(self.folder, "log_generation.txt", level=kwargs["verbose_level"])
        self.v_logger = Logger(self.folder, "log_validation.txt", level=kwargs["verbose_level"])
        # main logger for system messages
        self.m_logger = Logger(self.folder, "log.txt")

        self.exec_list_path = kwargs['exec_list_path']
        self.round_list = kwargs['round_list']
        self.round_dict = {}
        self.exec_list = self.read_exec_list_path(self.exec_list_path)

        self.target = kwargs['target']
        self.timelimit_per_program = kwargs['timelimit_per_program']
        self.status_file = f"{self.folder}/status.json"
        self.initialize()
        self.resume()

    def initialize(self):
        self.m_logger.logo(
            f"Initializing ... the exec list to be test is {self.exec_list}", level=LEVEL.INFO
        )
        current_directory = os.getcwd()
        # print("当前工作路径:", current_directory)

    def resume(self):
        print("the save dir is : ", self.folder)

    def read_exec_list_path(self, path_list:list) -> list:
        # 存放不同的harness测试结果的目录
        self.m_logger.logo("Reading harness list file", level=LEVEL.INFO)
        all_lines = []

        # 遍历当前目录中的所有文件
        for i, path in enumerate(path_list):
            for entry in os.listdir(path):
                # 构建完整的文件路径
                full_path = os.path.join(path, entry)

                # 检查是否是文件
                if os.path.isfile(full_path):
                    # 检查文件名是否不包含 '.'
                    if ('.' not in entry) and ('_wrap' in entry):    ###   考虑都加上 ，把原始harness, sep, wrap都测试
                        all_lines.append(full_path)
                        self.round_dict[full_path] = self.round_list[i]

        self.m_logger.logo("Done", level=LEVEL.INFO)
        return all_lines




class CudaAPIController():
    def __init__(self, **kwargs):
        self.cpu_semaphore = kwargs['cpu_semaphore']
        self.timelimit_per_program = kwargs['timelimit_per_program']
        self.api = kwargs['api']
        self.working_dir = kwargs['working_dir']
        self.round = kwargs['round']
        # self.m_logger = Logger(self.working_dir, "log.txt")
    def start_process(self):
        with self.cpu_semaphore:
            time.sleep(1+random.randint(0, 1))

            fuzz_out_dir_thread = self.working_dir + f'/{os.path.basename(self.api)}_round{self.round}'
            subprocess.run(["rm", "-rf", fuzz_out_dir_thread], check=True)
            subprocess.run(["mkdir", fuzz_out_dir_thread], check=True)
            subprocess.run(["cp", self.api, fuzz_out_dir_thread], check=True)

            if 'nvjpeg' in fuzz_out_dir_thread:
                fuzz_commond = f'{fuzz_out_dir_thread}/{os.path.basename(self.api)} ./nvjpeg/fuzz_output_threads/OIP.jpg'
                fuzz_commond = [f'{fuzz_out_dir_thread}/{os.path.basename(self.api)}', './nvjpeg/fuzz_output_threads/OIP.jpg']
            else:
                fuzz_commond = f'{fuzz_out_dir_thread}/{os.path.basename(self.api)}'
            result = oracle_report(fuzz_commond, self.timelimit_per_program)
            # if result >1:
            #     self.m_logger.logo(
            #         f" *********!!!!!********* {self.api} report:  {result}", level=LEVEL.INFO
            #     )
        print(f"pre fuzz done round {self.round}:{self.api}, result: {result}", flush=True)
        logging.info(f"pre fuzz done round {self.round}:{self.api}, result: {result}")
        return result


if '__name__' == '__main__':
    print(1)
