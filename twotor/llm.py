import asyncio
import json
import queue
import threading

import httpx

from .conversation import system_prompt


class DeepSeek:
    def __init__(self, config, transport=None):
        self.config = config
        self.transport = transport

    def stream(self, history, instruction, cancel: threading.Event):
        cfg = self.config
        if not cfg.api_key:
            raise RuntimeError("Fill DEEPSEEK_API_KEY in .env, then restart 2tor.")
        messages = [{"role": "system", "content": system_prompt(cfg.correction)}]
        messages.extend(history[-20:])
        if instruction:
            messages.append({"role": "user", "content": instruction})
        timeout = httpx.Timeout(connect=10, read=15, write=10, pool=10)
        tokens = queue.Queue(maxsize=32)
        stopped = threading.Event()
        errors = []

        async def produce():
            async with (
                httpx.AsyncClient(timeout=timeout, transport=self.transport) as client,
                client.stream(
                    "POST",
                    cfg.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + cfg.api_key},
                    json={
                        "model": cfg.model,
                        "messages": messages,
                        "stream": True,
                        "thinking": {"type": "disabled"},
                        "max_tokens": 180,
                    },
                ) as response,
            ):
                if response.status_code != 200:
                    hints = {
                        401: "API key invalid",
                        402: "Insufficient API balance",
                        429: "API rate limit; try again shortly",
                    }
                    raise RuntimeError(
                        hints.get(response.status_code, f"DeepSeek HTTP {response.status_code}")
                    )
                complete = False
                async for line in response.aiter_lines():
                    if cancel.is_set():
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        complete = True
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError("DeepSeek returned a stream error; retry your message.")
                    for choice in chunk.get("choices", []):
                        token = choice.get("delta", {}).get("content")
                        if token:
                            while not cancel.is_set() and not stopped.is_set():
                                try:
                                    tokens.put_nowait(token)
                                    break
                                except queue.Full:
                                    await asyncio.sleep(0.02)
                if not complete and not cancel.is_set():
                    raise RuntimeError("DeepSeek connection ended before the reply was complete.")

        async def run():
            task = asyncio.create_task(produce())
            try:
                while not task.done():
                    await asyncio.wait({task}, timeout=0.05)
                    if cancel.is_set() or stopped.is_set():
                        task.cancel()
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=lambda: asyncio.run(run()), name="2tor-http", daemon=True)
        worker.start()
        try:
            while worker.is_alive() or not tokens.empty():
                if cancel.is_set():
                    return
                try:
                    yield tokens.get(timeout=0.05)
                except queue.Empty:
                    pass
            if errors and not cancel.is_set():
                raise errors[0]
        finally:
            stopped.set()
            worker.join(timeout=1)
