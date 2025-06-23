from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load environment variables once
load_dotenv('.env', override=True)
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

async def scrape_channel(channel_username, media_dir, csv_path):
    client = TelegramClient(f'session_{channel_username}', api_id, api_hash)
    await client.start()
    entity = await client.get_entity(channel_username)
    channel_title = entity.title
    print(f"Scrapping data from {channel_title}")

    total = await client.get_messages(entity, limit=0)
    total_count = total.total if hasattr(total, 'total') else 10000

    pbar = tqdm(total=total_count, desc=f"Scraping {channel_title}", unit='message')

    with open(csv_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        async for message in client.iter_messages(entity, limit=10):
            media_path = None
            views = getattr(message, "views", None)
            # if message.media and hasattr(message.media, 'photo'):
            #     filename = f"{channel_username}_{message.id}.jpg"
            #     media_path = os.path.join(media_dir, filename)
            #     await client.download_media(message.media, media_path)
            writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path, views])
            pbar.update(1)
        pbar.close()
    await client.disconnect()

def run_scraper(channel_username, media_dir, csv_path):
    import asyncio
    asyncio.run(scrape_channel(channel_username, media_dir, csv_path))

def main():
    media_dir = 'data/photos'
    os.makedirs(media_dir, exist_ok=True)
    csv_path = 'data/telegram_data.csv'
    # Write header if file does not exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path', 'Views'])

    channels = [
            '@Shageronlinestore', '@Leyueqa', '@sinayelj', '@marakibrand', '@qnashcom', '@MerttEka'  
        ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        for channel in channels:
            executor.submit(run_scraper, channel, media_dir, csv_path)

if __name__ == "__main__":
    main()