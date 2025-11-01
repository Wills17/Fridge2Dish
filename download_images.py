
from bing_image_downloader import downloader

labels = ["egg", "tomato", "onion", "carrot", "milk", "bread", 
          "banana", "apple", "cheese", "potato", "butter", "lemon", "yogurt"]

for label in labels:
    downloader.download(label, limit=80, output_dir='dataset', adult_filter_off=True)
