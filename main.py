import asyncio

from infrastructure.elasticsearch.client import es


async def main() -> None:
    info = await es.info()
    print(info)

    await es.close()


if __name__ == "__main__":
    asyncio.run(main())