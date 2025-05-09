import os
import subprocess

downloads = [
    {
        "filename": "massive-stock-news-analysis-db-for-nlpbacktests.zip",
        "url": "https://www.kaggle.com/api/v1/datasets/download/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    },
    {
        "filename": "massive-yahoo-finance-dataset.zip",
        "url": "https://www.kaggle.com/api/v1/datasets/download/iveeaten3223times/massive-yahoo-finance-dataset"
    }
]

for file in downloads:
    file_path = os.path.join(os.getcwd(), file["filename"])

    if os.path.exists(file_path):
        print(f"✅ {file['filename']} already exists.")
    else:
        print(f"⬇️  Downloading {file['filename']}...")
        subprocess.run(["curl", "-L", "-o", file_path, file["url"]], check=True)
        print(f"✅ Downloaded {file['filename']}")
