# Soundification
Note: Repo is not completed
## Sound-Classification using Pytorch
The master branch works with **PyTorch 1.1 to 1.3.1.**
![](https://miro.medium.com/fit/c/1838/551/0*FPuNYV2HfUF8FCtb.png)
## Introduction
This repository is open-sourced toolbox based on pytorch which can be used for Sound Classification like Emotion recognition, Urbansound 8k, FSD

## Feature
* It is fast as uses Pytorch and GPUs.
* Dataset Management easy.
* Can run on colab.
* It is platform friendly.
* Multi Feature data extraction like MFCC, Chroma, STFT.
* Log files and Tensorboard
## TO-DO:
- [ ] Train the model on multiple dataset.
- [ ] Make model.yaml
- [ ] Distributed Training multi GPU
- [ ] Make a code to compute MFCC faster.

### Getting Started
To Install the repo and train it on your own Custom Sound-Classifcation Dataset use 


## Emotion Detetion
## MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation using Pytorch.
### Dataset
Multimodal EmotionLines Dataset (MELD) has been created by enhancing and extending EmotionLines dataset. MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.

Example Dialogue
![](https://github.com/SenticNet/MELD/raw/master/images/emotion_shift.jpeg)

### Dataset Statistics
| Statistics                      | Train   | Dev     | Test    |
|---------------------------------|---------|---------|---------|
| # of modality                   | {a,v,t} | {a,v,t} | {a,v,t} |
| # of unique words               | 10,643  | 2,384   | 4,361   |
| Avg. utterance length           | 8.03    | 7.99    | 8.28    |
| Max. utterance length           | 69      | 37      | 45      |
| Avg. # of emotions per dialogue | 3.30    | 3.35    | 3.24    |
| # of dialogues                  | 1039    | 114     | 280     |
| # of utterances                 | 9989    | 1109    | 2610    |
| # of speakers                   | 260     | 47      | 100     |
| # of emotion shift              | 4003    | 427     | 1003    |
| Avg. duration of an utterance   | 3.59s   | 3.59s   | 3.58s   |

Please visit https://affective-meld.github.io for more details.

### Dataset Distribution

|          | Train | Dev | Test |
|----------|-------|-----|------|
| Anger    | 1109  | 153 | 345  |
| Disgust  | 271   | 22  | 68   |
| Fear     | 268   | 40  | 50   |
| Joy      | 1743  | 163 | 402  |
| Neutral  | 4710  | 470 | 1256 |
| Sadness  | 683   | 111 | 208  |
| Surprise | 1205  | 150 | 281  |

The repo is maintained by [RAVISING-H](https://github.com/ravising-h/)
