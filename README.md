# Kannada Sign Language Recognition

Real-time recognition of Kannada sign language gestures from a webcam, built as a final-year engineering project.

The app tracks your hand with MediaPipe, classifies the gesture with an SVM, and shows the matching Kannada letter or word on screen in Kannada script. You can build a sentence one sign at a time and send it to a Telegram chat.

## What it recognises

The trained model (`model.pkl`) knows **89 signs**:

- **Vowels (15):** ಅ ಆ ಇ ಈ ಉ ಊ ಋ ಎ ಏ ಐ ಒ ಓ ಔ ಅಂ ಅಃ
- **Consonants (34):** ಕ ಖ ಗ ಘ ಙ ಚ ಛ ಜ ಝ ಞ ಟ ಠ ಡ ಢ ಣ ತ ಥ ದ ಧ ನ ಪ ಫ ಬ ಭ ಮ ಯ ರ ಲ ವ ಶ ಷ ಸ ಹ ಳ
- **Words (40):** ನಮಸ್ಕಾರ, ಧನ್ಯವಾದಗಳು, ವಂದನೆಗಳು, ಅಮ್ಮ, ಅಪ್ಪ, ಪೋಷಕರು, ಶಿಕ್ಷಕರು, ವಿದ್ಯಾರ್ಥಿಗಳು, ಪ್ರಾಂಶುಪಾಲರು, ಕಾಲೇಜು, ಇಂಜಿನಿಯರಿಂಗ್, ವಿಟಿಯು, ಫಲಿತಾಂಶ, ಗುರಿ, ಪ್ರಯತ್ನ, ಕೆಲಸ, ನಂಬಿಕೆ, ಆಶೀರ್ವಾದ, ಶುಭ, ಮುಂಜಾನೆ, ಹೇಗಿದೀರ, ಏನು ಮಾಡುತ್ತಿದಿಯ, ... (full list in the `names` dictionary in `recognition1.py`)

## How it works

```
Webcam frame ──► MediaPipe Hands ──► 21 landmarks × (x, y, z) = 63 features ──► SVM (RBF) ──► Kannada text overlay
```

1. **`dataset.py`**: captures webcam images for one sign into `DATASET/<sign>/` (set `Class` in the script).
2. **`to_csv.py`**: runs MediaPipe on every image in `DATASET/` and writes the 63 landmark values plus the label to `dataset.csv`.
3. **`train.ipynb`**: trains an SVM (`C=10, gamma=0.1, kernel='rbf'`) on `dataset.csv`, evaluates it, and saves `model.pkl`.
4. **`recognition1.py`**: the live app.

`kannada_font.py` is a small test that checks Kannada text renders correctly on the video feed.

## Setup

Requires Python 3.8–3.10 and a webcam.

```bash
pip install -r requirements.txt
```

**Kannada font:** the app needs a Kannada font saved as `kannada.ttf` in the project folder. It isn't included in this repo for licensing reasons. Download [Noto Sans Kannada](https://fonts.google.com/noto/specimen/Noto+Sans+Kannada) (free, OFL licence), then copy one of its `.ttf` files here and rename it to `kannada.ttf`.

**Telegram (optional):** to send recognised sentences to Telegram, create a bot with [@BotFather](https://t.me/BotFather) and set:

```bash
# Windows PowerShell
$env:TELEGRAM_BOT_TOKEN = "<your bot token>"
$env:TELEGRAM_CHAT_ID   = "<your chat id>"
```

If these aren't set, the app still runs and just skips sending.

## Usage

```bash
python recognition1.py
```

| Key | Action |
|-----|--------|
| `a` | Add the currently recognised sign to the sentence |
| `b` | Add a space |
| `e` | Show the sentence on screen, send it to Telegram, and start a new one |
| `q` | Quit |

The frame updates about once per second so you have time to hold each sign.

## Training your own model

```bash
python dataset.py     # repeat for each sign, changing Class each time
python to_csv.py      # builds dataset.csv from DATASET/
# then run train.ipynb
```

`dataset.csv` in this repo is only a small sample (three classes) that shows the format. The full training images for the 89-sign model aren't included, but the trained `model.pkl` is, so the app works without retraining.

> `model.pkl` is a pickled scikit-learn model. If loading it gives a version error with a newer scikit-learn, retrain it with `train.ipynb` or install the older scikit-learn it was saved with (around 1.2, early 2023).

## Tech stack

Python · OpenCV · MediaPipe · scikit-learn · Pillow · pandas · telepot
