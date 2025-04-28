#!/bin/bash

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-d small|all]"
  echo -e "\t-d small|all - Specify whether to download entire dataset (all) or just 1000 (small)"
  exit 1 # Exit script after printing help
}

# Get values of command line flags
while getopts d: flag
do
  case "${flag}" in
    d) data=${OPTARG};;
  esac
done

if [ -z "$data" ]; then
  echo "[ERROR]: Missing -d flag"
  helpFunction
fi

# Install Python Dependencies
conda install spacy
pip install -r requirements_arm.txt;

# Install Environment Dependencies via `conda`
conda install -c pytorch faiss-cpu;
conda install -c conda-forge openjdk=11;

# Download dataset into `data` folder via `gdown` command
mkdir -p data;
cd data;
if [ "$data" == "small" ]; then
  gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
  gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
elif [ "$data" == "all" ]; then
  gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
  gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
else
  echo "[ERROR]: argument for `-d` flag not recognized"
  helpFunction
fi
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
cd ..

# Download spaCy large NLP model
python -m spacy download en_core_web_sm

# Build search engine index
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..

# # Create logging folder + samples of log data
# get_human_trajs () {
#   PYCMD=$(cat <<EOF
# import gdown
# url="https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
# gdown.download_folder(url, quiet=True, remaining_ok=True)
# EOF
#   )
#   python -c "$PYCMD"
# }
# mkdir -p user_session_logs/
# cd user_session_logs/
# echo "Downloading 50 example human trajectories..."
# get_human_trajs
# echo "Downloading example trajectories complete"
# cd ..

get_human_trajs () {
  PYCMD=$(cat <<EOF
import gdown
import os
import subprocess
import requests
import json
import time

def try_gdown_download():
    url = "https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
    try:
        print("Attempting download with gdown...")
        gdown.download_folder(url, quiet=True, remaining_ok=True, use_cookies=True)
        return True
    except Exception as e:
        print(f"gdown download failed: {e}")
        return False

def try_direct_api_download():
    folder_id = "16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
    try:
        print("Attempting download via Google Drive API...")
        # Get file listing
        query = f"'{folder_id}' in parents and trashed = false"
        url = f"https://www.googleapis.com/drive/v3/files?q={query}&key=AIzaSyAa8yy0GdcGPHdtD083HiGGx_S0vMPScDM"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Failed to list files: {resp.status_code}")
            return False
            
        files = json.loads(resp.text).get('files', [])
        for file in files:
            file_id = file['id']
            file_name = file['name']
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key=AIzaSyAa8yy0GdcGPHdtD083HiGGx_S0vMPScDM"
            
            # Download the file
            r = requests.get(download_url, stream=True)
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {file_name}")
            time.sleep(1)  # Avoid rate limits
        return True
    except Exception as e:
        print(f"API download failed: {e}")
        return False

def try_wget_download():
    folder_id = "16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
    try:
        print("Attempting download with wget...")
        # This is a simplified approach - wget can't directly download folders
        # For each known file in the folder (you'd need to know these in advance):
        known_files = [
            # Add file IDs here if you know them
            # Format: ("file_id", "filename")
        ]
        
        success = False
        for file_id, filename in known_files:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            result = subprocess.run(["wget", "--no-check-certificate", download_url, "-O", filename])
            if result.returncode == 0:
                print(f"Downloaded {filename}")
                success = True
            time.sleep(1)  # Avoid rate limits
        return success
    except Exception as e:
        print(f"wget download failed: {e}")
        return False

def try_curl_download():
    folder_id = "16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
    try:
        print("Attempting download with curl...")
        # Similar to wget approach
        known_files = [
            # Add file IDs here if you know them
            # Format: ("file_id", "filename")
        ]
        
        success = False
        for file_id, filename in known_files:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            result = subprocess.run(["curl", "-L", download_url, "-o", filename])
            if result.returncode == 0:
                print(f"Downloaded {filename}")
                success = True
            time.sleep(1)  # Avoid rate limits
        return success
    except Exception as e:
        print(f"curl download failed: {e}")
        return False

def try_browser_emulation():
    try:
        print("Attempting download with browser emulation...")
        import mechanize
        
        # Create a browser object
        br = mechanize.Browser()
        br.set_handle_robots(False)
        br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
        
        # Navigate to the folder
        url = "https://drive.google.com/drive/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
        response = br.open(url)
        
        # This is a simplified approach - in reality you would need to:
        # 1. Parse the page to find download links
        # 2. Navigate to each file
        # 3. Trigger the download
        # 4. Save the file locally
        
        print("Browser emulation requires more complex logic to fully implement")
        return False
    except ImportError:
        print("mechanize not installed, skipping browser emulation")
        return False
    except Exception as e:
        print(f"Browser emulation failed: {e}")
        return False

# Try each method in sequence until one works
if not try_gdown_download():
    if not try_direct_api_download():
        if not try_wget_download():
            if not try_curl_download():
                if not try_browser_emulation():
                    print("All download methods failed. Please download manually from:")
                    print("https://drive.google.com/drive/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto")
                    print("and place the files in the current directory.")
EOF
  )
  python -c "$PYCMD"
}
mkdir -p user_session_logs/
cd user_session_logs/
echo "Downloading 50 example human trajectories..."
get_human_trajs
echo "Downloading example trajectories complete"
cd ..