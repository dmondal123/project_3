# API Creation

This engineering task involves creating an API for music genre classification using MIDI embeddings. It first detects whether music is present or not using the YAMNet Model by setting the  threshold for an audio file being music being 0.6 as below that, it detects noise like mic distortions and sirens as music as well. If music is not present, then it says "no music detected" and otherwise it moves on to embedding generation using MIDI2VEC and predicts the genre of the song. Some MIDI files are provided for testing in the test_songs folder. 

Note: This API is made using Streamlit and a Tensorflow Model which isn't compatible with MAC M1 Pro Chip hence it is made on Google Colab instead of a python file.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)

## Requirements

Note: Please make sure gensim version 3.8.3 is correctly installed in your system before running the files.

- Python 3.11 or later
- NodeJS
- Required packages:
  - `streamlit`
  - `timidity`
  - `gensim==3.8.3`
  - `nodevectors=2.8`
  - `tensorflow_hub`
  - `scipy`

You can install the required packages using:

```bash
pip install -q streamlit
npm install localtunnel
sudo apt-get install timidity
pip install gensim==3.8.3
pip install nodevectors==2.8
pip install tensorflow_hub
pip install scipy
```

## Usage

# Running a Streamlit App in Google Colab

This guide explains how to run a Streamlit app in Google Colab and access it using a generated URL. Follow the steps below to get started:

1. **Set up your Google Colab environment:**
    - Make sure you have a Colab notebook open and have installed the required packages, including Streamlit and Localtunnel. Upload utilities.py and the model which is birnn1.h5 in the /contents folder on Google Colab.

2. **Running the code**
    - Load the safeapi.ipynb file on /contents folder of Google Colab and run each cell.

3. **Run the following command in a Colab cell:**
    ```bash
    !streamlit run app.py &> /content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com
    ```

4. **Understand the output:**
    - The command runs your Streamlit app and redirects logs to `/content/logs.txt`.
    - It also establishes a local tunnel to expose your app to the internet.
    - The command outputs your IP address (e.g., `34.106.130.179`) and a Localtunnel URL (e.g., `https://pretty-carrots-vanish.loca.lt`).

5. **Access the app:**
    - The provided IP address (e.g., `34.106.130.179`) serves as the password to access your app.
    - The URL (e.g., `https://pretty-carrots-vanish.loca.lt`) is the link where you can access your Streamlit app.

6. **Check logs:**
    - If needed, you can review the logs at `/content/logs.txt` to troubleshoot any issues.

With these instructions, you should be able to successfully run your Streamlit app in Google Colab and access it using the generated Localtunnel URL.
