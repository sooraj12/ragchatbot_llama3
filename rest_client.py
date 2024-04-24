import aiohttp
import sys
import asyncio

args = sys.argv[1:]

if len(args) != 1:
    print("Usage: Provide question")
    sys.exit(1)

question = args[0]
print(question)


async def make_api_call():
    try:
        data = {"input": {"question": question}}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8080/chat/stream", json=data
            ) as response:
                print(await response.text())
    except Exception as e:
        print(e)


asyncio.run(make_api_call())
