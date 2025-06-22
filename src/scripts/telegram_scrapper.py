from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables once
load_dotenv('.env', override=True)
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir):
    """    
    Scrapes messages from a Telegram channel and writes them to a CSV file.
    Args:
        client (TelegramClient): The Telegram client instance.
        channel_username (str): The username of the channel to scrape.
        writer (csv.writer): The CSV writer object to write data.
        media_dir (str): Directory to save media files.
    """
    entity = await client.get_entity(channel_username)
    channel_title = entity.title  # Extract the channel's title
    print(f"Scrapping data from {channel_title}")

    total = await client.get_messages(entity, limit=0)  # Get total messages count
    total_count = total.total if hasattr(total, 'total') else 10000

    pbar = tqdm(total=total_count, desc=f"Scraping {channel_title}", unit='message')

    async for message in client.iter_messages(entity, limit=10000):
        media_path = None
        if message.media and hasattr(message.media, 'photo'):
            # Create a unique filename for the photo
            filename = f"{channel_username}_{message.id}.jpg"
            media_path = os.path.join(media_dir, filename)
            # Download the media to the specified directory if it's a photo
            await client.download_media(message.media, media_path)
        
        # Write the channel title along with other data
        writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])
        pbar.update(1)
    pbar.close()

# Initialize the client once
client = TelegramClient('scraping_session', api_id, api_hash)

async def main():
    await client.start()
    
    # Create a directory for media files
    media_dir = 'data/photos'
    os.makedirs(media_dir, exist_ok=True)

    # Open the CSV file and prepare the writer
    with open('data/telegram_data.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])  # Include channel title in the header
        
        # List of channels to scrape
        channels = [
            '@Leyueqa', '@sinayelj', '@marakibrand', '@qnashcom', '@MerttEka'  
        ]
        
        # Iterate over channels and scrape data into the single CSV file
        for channel in channels:
            await scrape_channel(client, channel, writer, media_dir)
            print(f"Scraped data from {channel}")

with client:
    client.loop.run_until_complete(main())
