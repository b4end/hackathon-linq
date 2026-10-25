"""Боевой бэкенд: RabbitMQ -> их TPU-воркеры.
Логика сообщений 1:1 из их example.ipynb, плюс реконнект с бэкoffом и ретраями.
Требует VPN и пакет tpu_pycloud_format в корне репо (см. README)."""
import json
import time
import uuid

import pika

from linq.config import Settings
from linq.registry import PROGRAMS, validate_inputs
from tpu_pycloud_format import (
    MESSAGE_ID_ATTR_NAME,
    RESPONSE_ROUTING_KEYS_ATTR_NAME,
    TARGET_URI_ATTR_NAME,
    PyTask,
    deserialize,
    serialize,
)

# ошибки, после которых имеет смысл повторять попытку
CONNECTION_ERRORS = (pika.exceptions.AMQPError, OSError)


class CloudBackend:
    name = 'cloud'

    def __init__(self, settings: Settings, max_attempts: int = 4, backoff_base: float = 1.5):
        self.s = settings
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._conn = None
        self._ch = None
        self.last_worker = None      # инфо о воркере из последнего ответа (для бенча)
        self.last_latency_s = 0.0

    # --- соединение -----------------------------------------------------------
    def _channel(self):
        if self._conn is None or self._conn.is_closed:
            self._conn = pika.BlockingConnection(pika.ConnectionParameters(
                self.s.rabbit_host,
                self.s.rabbit_port,
                credentials=pika.PlainCredentials(self.s.rabbit_user, self.s.rabbit_pass),
            ))
            self._ch = self._conn.channel()
        return self._ch

    def _reset(self) -> None:
        try:
            if self._conn is not None and not self._conn.is_closed:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        self._ch = None

    # --- инференс ---------------------------------------------------------------
    def run(self, program: str, tensors: dict) -> dict:
        spec = PROGRAMS[program]
        validate_inputs(spec, tensors)
        last_err = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._run_once(spec, tensors)
            except CONNECTION_ERRORS as err:
                last_err = err
                self._reset()
                if attempt < self.max_attempts:
                    time.sleep(min(10.0, self.backoff_base ** attempt))
        raise last_err

    def _run_once(self, spec, tensors: dict) -> dict:
        ch = self._channel()
        task_id = uuid.uuid4().hex
        reply_queue = f'{self.s.reply_prefix}_{task_id[:8]}'

        # личная очередь ответов
        ch.queue_declare(queue=reply_queue, auto_delete=True)
        ch.queue_bind(exchange=self.s.exchange, queue=reply_queue, routing_key=reply_queue)

        started = time.monotonic()
        ch.basic_publish(
            exchange=self.s.exchange,
            routing_key=self.s.input_queue,
            body=serialize(PyTask(tensors)),
            properties=pika.BasicProperties(headers={
                TARGET_URI_ATTR_NAME: self.s.target_uri(spec.file),
                RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([reply_queue]),
                MESSAGE_ID_ATTR_NAME: task_id,
            }),
        )

        method = props = body = None
        for method, props, body in ch.consume(reply_queue, auto_ack=True, inactivity_timeout=1.0):
            if method is None:
                if time.monotonic() - started > self.s.timeout_s:
                    raise TimeoutError(f'нет ответа за {self.s.timeout_s} с')
                continue
            ch.cancel()
            break
        if body is None:
            raise TimeoutError('очередь ответа закрыта пустой')
        self.last_latency_s = time.monotonic() - started

        if props.headers.get(MESSAGE_ID_ATTR_NAME) != task_id:
            raise RuntimeError('ответ не на нашу задачу')
        data, metadata, worker = deserialize(body)
        self.last_worker = worker
        if not data:
            raise RuntimeError(f'ошибка воркера: {worker.get("error_message")}')
        return data

    def close(self) -> None:
        self._reset()