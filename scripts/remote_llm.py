import asyncio
from typing import Any, Dict

import aiohttp


class RemoteLLM:
    """
    Async wrapper around the remote Qwen2.5-7B-Instruct endpoint.
    """

    def __init__(self, endpoint: str = "http://143.110.210.212/v1/chat/completions"):
        self.endpoint = endpoint

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        text = data["choices"][0]["message"]["content"]
        return text.strip()

    def chat_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Convenience synchronous wrapper (runs the async call in an event loop).
        """
        return asyncio.run(
            self.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

