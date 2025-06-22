# 🚀 Telegram Commerce NER

A Natural Language Processing (NLP) project for extracting commerce-related entities from Telegram messages.

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Telegram Scraping](#telegram-scraping-using-telethon)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)



## 📝 Overview

Telegram Commerce NER provides robust tools and machine learning models to automatically identify and extract commerce-related entities—such as products, prices, quantities, and seller information—from Telegram messages. This project is ideal for automating data extraction for e-commerce analytics, chatbots, business intelligence, and more.

**Key Use Cases:**
- Automate product and price extraction from chat logs
- Power intelligent chatbots for commerce
- Enable business analytics on Telegram commerce data

## ✨ Features

- 🤖 **Pre-trained NER models** tailored for commerce domains
- 🔗 **Seamless integration** with Telegram chat data (JSON format)
- 🏷️ **Customizable entity types** to fit your business needs
- 📊 **Evaluation scripts** with detailed metrics (precision, recall, F1-score)
- 🛠️ **Support for fine-tuning** on your own datasets
- 📁 **Sample data and configuration files** included

## ⚡ Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/telegram-commerce-ner.git
    cd telegram-commerce-ner
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
## 📥 Telegram Scraping Using Telethon

To collect Telegram messages for entity extraction, you can use the [Telethon](https://github.com/LonamiWebs/Telethon) library.

**Example: Scrape messages from a channel**

```python
from telethon import TelegramClient

api_id = 'YOUR_API_ID'
api_hash = 'YOUR_API_HASH'
channel = 'https://t.me/your_channel'

with TelegramClient('anon', api_id, api_hash) as client:
    for message in client.iter_messages(channel, limit=1000):
        print(message.text)
```

- Replace `YOUR_API_ID` and `YOUR_API_HASH` with your Telegram API credentials.
- The script prints messages; you can modify it to save messages in JSON format for use with this project.
- See [Telethon documentation](https://docs.telethon.dev/) for more details.

## 🚦 Usage

1. **Prepare your Telegram message data** in the supported JSON format (see [`data/README.md`](data/README.md) for details and examples).
2. **Run the entity extraction script:**

    ```bash
    python src/scripts/telegram_scrapper.py
    ```

3. **Review the extracted entities** in the output file. Entities will be structured by message and entity type.

4. **(Optional) Customize entity types** by editing the list of channels.

## 🏋️ Model Training

To train a new model or fine-tune an existing one on your own data:

```bash
python src/scripts/fine_tune_ner.py
```


- Training logs and checkpoints will be saved in the `models/` directory.


## 🤝 Contributing

Contributions are welcome! 🚀

- Open [issues](https://github.com/yourusername/telegram-commerce-ner/issues) for bug reports or feature requests.
- Submit [pull requests](https://github.com/yourusername/telegram-commerce-ner/pulls) for improvements.
- Please follow the [contribution guidelines](CONTRIBUTING.md) if available.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Made with ❤️ for the open-source community.