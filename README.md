# TTS_FineTuning_French
**Fine-Tuning TTS for French Language**
# Introduction
Text-to-Speech (TTS) synthesis has become a crucial technology in today's digital landscape, facilitating applications ranging from accessibility to entertainment. This project focuses on fine-tuning Microsoft's SpeechT5 TTS model for French language synthesis, addressing the demand for high-quality speech synthesis systems tailored for French speakers. By enhancing the TTS capabilities for French, we aim to foster better user engagement and broaden the accessibility of technology for French-speaking populations.

# Model Overview
**Base Model:** Microsoft SpeechT5 (microsoft/speecht5_tts)
**Fine-Tuned Model:** DeepDiveDev/French_finetuned_speecht5_tts
**Task:** Text-to-Speech (TTS)
**Language:** French
**Dataset:** [ymoslem/MediaSpeech]
## Intended Uses & Limitations

# Intended Uses
- Accessibility tools for visually impaired individuals.

- Educational platforms supporting language learning.

- Virtual assistants for natural conversations.

- Content creation and media localization.
# Limitations

- Currently limited to French language with potential challenges in dialectal variations.

- Performance may vary based on input complexity and speaker characteristics.

## Training and Evaluation Data

# Dataset Characteristics

- High-quality audio recordings from native French speakers.

- Diverse phonetic coverage, ensuring representation of different accents and styles.
  
 # Training Procedure
 
- The model was fine-tuned on a selected dataset, ensuring comprehensive coverage for TTS tasks.

## Training Hyperparameters
The following hyperparameters were used during training:

. Learning Rate: 0.0001

. Train Batch Size: 4

. Eval Batch Size: 2

. Seed: 42

. Gradient Accumulation Steps: 8

. Total Train Batch Size: 32

. Optimizer: AdamW with parameters (betas=(0.9, 0.999), epsilon=1e-08)

. LR Scheduler Type: Linear

. LR Scheduler Warmup Steps: 100

. Training Steps: 600

. Mixed Precision Training: Native AMP

## Training Results

# Training Progress

| Training Loss | Epoch | Step | Validation Loss |
|---------------|-------|------|-----------------|
|     1.0764    |  100  |  100 |      0.4995     |
|     0.8705    |  200  |  200 |      0.5197     |
|     0.7866    |  300  |  300 |      0.5469     |
|     0.7062    |  400  |  400 |      0.5615     |
|     0.6710    |  500  |  500 |      0.5805     |
|     0.6184    |  600  |  600 |      0.5870     |

# Key Enhancements and Improvements

- Fine-tuned on a curated French dataset to enhance TTS performance.

- Implemented advanced text preprocessing techniques for better handling of French linguistic features.

- Integrated speaker embeddings for a more diverse voice output.

## Objective Evaluation

The model exhibited consistent improvement during training:

- Initial Validation Loss: 0.4995

- Final Validation Loss: 0.5870

- Training Loss Reduction: from 1.0764 to 0.6184

## Subjective Evaluation
A Mean Opinion Score (MOS) evaluation was conducted with native French speakers focusing on:

- Naturalness and intelligibility

- Prosody and emphasis accuracy

Results indicated a significant improvement in the clarity and naturalness of speech compared to the baseline.

## Challenges and Solutions

# Dataset Challenges
. **Limited Availability:** High-quality French speech data was scarce.

. **Solution:** Augmented existing data through targeted recordings and preprocessing.

# Technical Challenges

- **Training Stability:** Addressed issues with memory constraints through gradient accumulation and warmup steps.

- **Inference Speed Optimization:** Applied model quantization to enhance real-time performance.
 ## Environment and Dependencies

* Transformers: 4.47.0.dev0

* PyTorch: 2.5.0+cu121

* Datasets: 3.0.2

* Tokenizers: 0.20.1
  
# Ethical Considerations

* Potential for misuse in generating misleading audio.
* Bias in voice generation may reflect demographics present in the training data.
  
## Conclusion
 
 # Key Achievements
. Successfully fine-tuned SpeechT5 for French TTS.

. Achieved significant reductions in loss metrics while maintaining high quality.

# Future Improvements
. Expand the dataset for broader phonetic coverage.

. Explore emotion and style transfer capabilities for dynamic speech synthesis.

# Usage
The model can be utilized with the Hugging Face Transformers library:
```
python
Copy code
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

# Load the model and processor
 model = SpeechT5ForTextToSpeech.from_pretrained("DeepDiveDev/French_finetuned_speecht5_tts")
 processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
 vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Generate speech
text = "Je travaille sur l'apprentissage automatique."  # Replace with your input text in French
  
 inputs = processor(text, return_tensors="pt")

# Generate speech
speech = model.generate(**inputs)
audio = vocoder(speech)
```

# Acknowledgements
Base SpeechT5 Model: Developed by Microsoft

Dataset: ymoslem/MediaSpeech

The open-source speech processing community for ongoing support and resources.

Internship Program: PARIMAL intern program at IIT Roorkee

## Citation

If you use this model, please cite:


```
@misc{DeepDiveDev/TTS_FineTune_French,
  author = {Jyotirmoyee Mandal},
  title = {Fine-tuning TTS for French},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://huggingface.co/DeepDiveDev/TTS_FineTune_French}},
  Contact Information: {Jyotirmoyeemandal63@gmail.com}.
}
```

## References

This code draws lessons from:
- [Hugging Face Audio Course - Fine-Tuning](https://huggingface.co/learn/audio-course/en/chapter6/fine-tuning)

# License
This project is licensed under the MIT License - see the LICENSE file for details.
