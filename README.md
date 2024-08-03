# [Comply AI](https://complyai.live/)
Every e-commerce merchant has to comply with multiple central & state regulations to ensure product & service safety for the buyer. All these information guidelines and documents are scattered throughout different platforms. Because of that, it can be very overwhelming for an entrepreneur just starting out. Therefore we built a centralized AI Based Legal Compliance & Product Safety Engine where one can search their queries, and questions about everything related to Legal Compliance & Product Safety in India for their businesses. Also, we have provided a Rich Image processing feature by using which people can validate their product packaging against the applicable legal compliances.

## Authors 

- Partha - [@iamparthaonline](https://github.com/iamparthaonline)
- Smit - [@smitFromWolkus](https://github.com/smitFromWolkus)
- Sai - [@saisharmavakkalanka](https://github.com/saisharmavakkalanka)
- Vedant - [@VedantS20](https://github.com/VedantS20)


## Installation 

Docker should be already installed in your system 

After cloning the repo

```
cd BFB-Hackathon-ONDC-2024
docker compose up
```

## Design and Architecture
![Architecture](https://github.com/iamparthaonline/BFB-Hackathon-ONDC-2024/blob/main/design-architecture.png)


## Tech Stack and Deployment Details

1. Website/App
    - Meteor.js
    - Vue.js

2. Model API's
    - FastAPI
    - Langchain

3. LLM's
   - Text Generation (Finetuned mistralai/Mistral-7B-v0.1)

4. Deployment
   - Google Cloud Platform

![Tech](https://github.com/iamparthaonline/BFB-Hackathon-ONDC-2024/blob/main/tech.png)


## Modal Fine Tunning 
We have used [RAFT](https://arxiv.org/pdf/2403.10131) (Retrieval Augmented Fine-Tuning) to Generate the dataset to fine tune Mistral 7B 

Dataset Link : [https://huggingface.co/datasets/Vedant20/ComplyAI_dataset/viewer](https://huggingface.co/datasets/Vedant20/ComplyAI_dataset/viewer)

![image](https://github.com/user-attachments/assets/8e901047-4b89-4afe-95b0-7ddd459916ec)




## Checkout The Application

[Website Link](https://complyai.live/)


