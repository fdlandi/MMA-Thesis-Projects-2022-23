# MMA-Thesis-Projects-2022-23

## Table of contents
   1. [Who We Are](#who-we-are)
   2. [Speech-related Projects](#speech-related-projects)
   3. [Video-related Projects](#video-related-projects)
   4. [Join Us](#join-us)

## Who We Are
We are the MultiModal Amsterdam (MMA) Team, from the Huawei Technologies Amsterdam Research Center. Our group counts seven full-time engineers/researchers working on the audio-visual aspects of deep learning. All of us have a PhD and/or multiple years of experience in Artificial Intelligence, Computer Science, or closely-related topics. For this reason, we can grant you high-quality supervision and mentorship throughout your Thesis experience. Read more about our currently available projects in speech and video topics below.

## Speech-related Projects

### End-to-End Speech Recognition/Translation

In recent months, multilingual Automatic Speech Recognition and Speech Translation have become major topics with some great proposals on how to handle them. Multi-tasking ASR unites speech recognition and speech translation into a sequence-to-sequence speech-to-text task. The only difference between speech recognition and speech translation is that while speech recognition converts speech signals into a text sequence in the same language, speech translation converts them into a different one. The most significant speech translation model is the Whisper model, which is a multi-task trained model released by OpenAI [1]. The model cannot only transcribe speech signals from multiple languages, but also translate them into English. Such models are much easier to deploy than multiple separate models for individual languages and such multilingual models perform well for general recognitions. However, for specific tasks there still exists a large gap in performance between a multilingual model and a dedicated model.

**Topic 1: Finetune pretrained model using real-world speech data.** Multilingual models are trained on an extensive amount of data, which can differ from the field user data. Field user data is likely to show much bigger variation in speakers, accents, and background noises. Most of the research so far has focused on adapting models to a single domain and single language, which multiplies the models used in production. The major task with regards to this topic is to use field data to finetune the pretrained model to boost the performance of both multilingual speech recognition and speech translation on field data at the same time [2], while avoiding catastrophic forgetting [3].

**Topic 2: Patch end-to-end speech translation model for new words.** A major issue in machine translation is keeping up with new name entities that emerge daily as a language evolves. Since those phrases are novel, it is very likely that they are not included in the original training data. This issue has become more severe to speech translation models. The goal of this proposal is to investigate an adaptation approach to ‘patch’ the model to ensure that the model can be up-to-date without retraining the model from scratch [4].

**Topic 3: External Language Model fusion for End-to-End ASR.** It is much harder for an End-to-End model to do domain-shifting than for hybrid systems without finetuning using audio inputs. Without a clear separation of acoustic and language model, LM fusion remains challenging for E2E systems. The goal is to explore the best way to close the gap between a dedicated model using an external in-domain Language Model and an End-To-End model’s decoder adaptation [5,6].

The perfect candidate for this MS Thesis project is familiar with language modeling. Excellent knowledge of Transformer-based E2E architectures is required (theory + code). Knowledge of Automatic Speech Recognition toolkits is a big plus. The candidate will work on a brand-new speech recognition system, bringing it closer to real-life scenarios.

**References**
* [1] Radford, Alec, et al. Robust speech recognition via large-scale weak supervision. Tech. Rep., Technical report, OpenAI, 2022.
* [2] https://huggingface.co/blog/fine-tune-whisper
* [3] Majumdar, Somshubra, et al. "Damage Control During Domain Adaptation for Transducer Based Automatic Speech Recognition." arXiv preprint arXiv:2210.03255 (2022).
* [4] Joshi, Raviraj, and Anupam Singh. "A Simple Baseline for Domain Adaptation in End to End ASR Systems Using Synthetic Data." Proceedings of The Fifth Workshop on e-Commerce and NLP (ECNLP 5). 2022.
* [5] Z. Meng et al., "Internal Language Model Estimation for Domain-Adaptive End-to-End Speech Recognition," 2021 IEEE Spoken Language Technology Workshop (SLT), 2021.
* [6] Tsunoo, Emiru, et al. "Residual Language Model for End-to-end Speech Recognition." arXiv preprint arXiv:2206.07430 (2022).

## Video-related Projects

### Referring Video Object Segmentation

Referring video object segmentation aims at segmenting objects and people in a video using a natural language expression as query. In particular, textual expressions are employed to identify the specific object/animal/person to be segmented in the video. The challenging nature of this task arises from the need to fuse multiple modalities (video + text) effectively. On the other hand, there is also the need to maintain the temporal consistency of the video modality when dealing with the stream of rgb frames. The ideal candidate for this MS Thesis project is familiar with multimodal fusion techniques and recent architectures for Computer Vision and Natural Language Processing. As a consequence, an excellent knowledge of Transformers architectures is required (theory + code). Previous experience on videos and/or time-series is a plus. The candidate will carry on cutting-edge research on the topic of referring video object segmentation with a strong orientation towards future publications.

**References**
* Seonguk Seo, Joon-Young Lee, Bohyung Han, “URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark”, European Conference on Computer Vision (ECCV), 2020. ([paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600205.pdf))
* https://github.com/wjn922/ReferFormer
* https://youtube-vos.org/

### Image/Video Captioning

Image/Video Captioning aims at generating a natural language description for an arbitrary image/video. This is done in two phases: 1) recognizing the content of the input image/video, and 2) decoding the correspondence sentence. In a similar fashion, neural architectures for captioning are divided in two functional blocks: image/video encoder and language decoder. With the advent of Transformers, these two components are apparently merged into a single unit, yet the challenge remains the same: how can we deal with these two modalities more effectively in order to produce better description? The ideal candidate for this MS Thesis project is familiar with vision-and-language tasks and recent architectures for Computer Vision and Natural Language Processing. An excellent knowledge of Transformers architectures is required (theory + code). Previous experience on videos and/or time-series is a plus. The candidate will carry on cutting-edge research on the topic of image or video captioning with a strong orientation towards future publications.

**References**
* Xu K, Ba J, Kiros R, Cho K, Courville A, Salakhudinov R, Zemel R, Bengio Y. “Show, attend and tell: Neural image caption generation with visual attention”. International Conference on Machine Learning (ICML), 2015. ([paper](https://proceedings.mlr.press/v37/xuc15.pdf))
* Lin K, Li L, Lin CC, Ahmed F, Gan Z, Liu Z, Lu Y, Wang L. “SwinBERT: End-to-end transformers with sparse attention for video captioning”. Computer Vision and Pattern Recognition (CVPR) 2022. ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_SwinBERT_End-to-End_Transformers_With_Sparse_Attention_for_Video_Captioning_CVPR_2022_paper.pdf), [github](https://github.com/microsoft/SwinBERT))

### Image/Video Synthesis from Textual Prompt

Very recently, the vision-and-language research community has delivered impressive results in the field of image generation from text, a.k.a. text-to-image translation. As a consequence, this topic has attracted increasing attention over the last few months, with even more impressive results in the form of image editing or video generation from textual prompts. Together with impressive results, the recent advancements bring new and challenging research questions: how can we build better models that can imagine and draw, similar to how a human would do? The ideal candidate for this MS Thesis project is familiar with vision-and-language tasks and recent architectures for Computer Vision and Natural Language Processing. An excellent knowledge of Transformers architectures is required (theory + code). Previous experience on diffusion models or generative models is a plus. The candidate will carry on cutting-edge research on the topic of image or video synthesis with a strong orientation towards future publications.

**References**
* https://openai.com/blog/dall-e/
* Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B. "High-resolution image synthesis with latent diffusion models". In Computer Vision and Pattern Recognition (CVPR), 2022. ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf), [github](https://github.com/CompVis/stable-diffusion))


## Join Us
We look forward to welcoming you into our team! If you feel one (or more) of our research topics fits you, please contact us and we will arrange an interview with you.
If the interview is successful, you will have the opportunity to work with us and carry on your Thesis project in our Amsterdam Research Center.

### Contact Information
**Sun Yang** (MMA Team Leader): [sunyang48@huawei.com](mailto:sunyang48@huawei.com)

**Federico Landi** (MMA Video): [federico.landi@huawei-partners.com](mailto:federico.landi@huawei-partners.com)

*Please, in your email always CC also **Bryan van Reijen** ([bryan.vanreijen@huawei.com](mailto:bryan.vanreijen@huawei.com)) and **Xiaoyi Sun** ([xiaoyi.sun1@huawei.com](mailto:xiaoyi.sun1@huawei.com)).*
