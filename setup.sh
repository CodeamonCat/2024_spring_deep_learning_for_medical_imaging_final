#!/bin/bash

# Set the Kaggle username and dataset name
username="paultimothymooney"
dataset="chest-xray-pneumonia"

# Prompt the user to set up API credentials
read -p "Enter your Kaggle username: " kaggle_username
read -s -p "Enter your Kaggle API key: " kaggle_api_key
echo

# Create the kaggle.json file with the provided credentials
echo "{\"username\":\"$kaggle_username\",\"key\":\"$kaggle_api_key\"}" > kaggle.json

# Install Kaggle API
pip install --user kaggle

# Create the .kaggle directory and move the API credentials file
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle

# Change the file permission to ensure it's not readable by other users
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
kaggle datasets download $username/$dataset

# Unzip the dataset
unzip $dataset.zip

# Remove the zip file
rm $dataset.zip

# Remove the unnecessary files
rm -r chest_xray/chest_xray
rm -r chest_xray/__MACOSX

# Ask the user if they want to run train.py and pretrained.py
read -p "Do you want to run train.py and pretrained.py? (y/n): " run_scripts

if [[ $run_scripts == "y" ]]; then
    # Run train.py
    python train.py

    # Run pretrained.py
    python pretrained.py
fi