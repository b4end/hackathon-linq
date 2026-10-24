"""Настройки сервиса: окружение / .env"""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    login: str = field(default_factory=lambda: _env('LINQ_LOGIN', 'hackathonXXXXX'))
    rabbit_host: str = field(default_factory=lambda: _env('LINQ_RABBIT_HOST', '10.0.114.30'))
    rabbit_port: int = field(default_factory=lambda: int(_env('LINQ_RABBIT_PORT', '5672')))
    rabbit_user: str = field(default_factory=lambda: _env('LINQ_RABBIT_USER', 'guest'))
    rabbit_pass: str = field(default_factory=lambda: _env('LINQ_RABBIT_PASS', 'guest'))
    exchange: str = field(default_factory=lambda: _env('LINQ_EXCHANGE', 'tpu.direct'))
    input_queue: str = field(default_factory=lambda: _env('LINQ_INPUT_QUEUE', 'input'))
    reply_prefix: str = field(default_factory=lambda: _env('LINQ_REPLY_PREFIX', 'pokus'))
    timeout_s: float = field(default_factory=lambda: float(_env('LINQ_TIMEOUT_S', '60')))
    program_dir: str = field(default_factory=lambda: _env('LINQ_PROGRAM_DIR', ''))

    @property
    def programs_root(self) -> str:
        """Общая папка на их стороне, откуда воркеры читают .tpu-программы."""
        return self.program_dir or f'/auto/cache/{self.login}'

    def target_uri(self, program_file: str) -> str:
        return f'file://{self.programs_root}/{program_file}?raw=False'