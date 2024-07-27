import axios from 'axios';

Meteor.methods({
  async ASR_NMT_handler(sourceLanguage, audioContent) {
    const res = Meteor.call('ASR_NMT_ConfigCall', sourceLanguage);
    if (!res.isSuccess) return res;
    const {pipelineResponseConfig, pipelineInferenceAPIEndPoint} = res.data;

    const asrServiceId = pipelineResponseConfig.find(task => task.taskType === 'asr').config[0].serviceId;
    const translationServiceId = pipelineResponseConfig.find(task => task.taskType === 'translation').config[0]
      .serviceId;

    const auth = {
      key: pipelineInferenceAPIEndPoint.inferenceApiKey.name,

      value: pipelineInferenceAPIEndPoint.inferenceApiKey.value,
    };
    const {callbackUrl} = pipelineInferenceAPIEndPoint;
    const data = Meteor.call(
      'ASR_NMT_ComputeCall',
      callbackUrl,
      auth,
      sourceLanguage,
      asrServiceId,
      translationServiceId,
      audioContent,
    );
    return data;
  },
  async ASR_NMT_ConfigCall(sourceLanguage) {
    const configCallconfig = {
      pipelineTasks: [
        {
          taskType: 'asr',
          config: {
            language: {
              sourceLanguage,
            },
          },
        },
        {
          taskType: 'translation',
          config: {
            language: {
              sourceLanguage,
              targetLanguage: 'en',
            },
          },
        },
      ],
      pipelineRequestConfig: {
        pipelineId: '64392f96daac500b55c543cd',
      },
    };
    const res = Meteor.call('createPostRequest', Meteor.settings.public.API_ENDPOINT, configCallconfig, {
      headers: {
        userID: Meteor.settings.public.userID,
        ulcaApiKey: Meteor.settings.public.ulcaApiKey,
      },
    });
    return res;
  },
  async ASR_NMT_ComputeCall(callbackUrl, auth, sourceLanguage, asrServiceId, translationServiceId, audioContent) {
    const computeCallConfig = {
      pipelineTasks: [
        {
          taskType: 'asr',
          config: {
            language: {
              sourceLanguage,
            },
            serviceId: asrServiceId,
            audioFormat: 'wav',
            samplingRate: 16000,
          },
        },
        {
          taskType: 'translation',
          config: {
            language: {
              sourceLanguage,
              targetLanguage: 'en',
            },
            serviceId: translationServiceId,
          },
        },
      ],
      inputData: {
        audio: [
          {
            audioContent,
          },
        ],
      },
    };
    const res = Meteor.call('createPostRequest', callbackUrl, computeCallConfig, {
      headers: {
        Authorization: auth.value,
      },
    });
    return res;
  },
  async ASR_handler(sourceLanguage, audioContent) {
    const res = Meteor.call('ASR_ConfigCall', sourceLanguage);
    if (!res.isSuccess) return res;
    const {pipelineResponseConfig, pipelineInferenceAPIEndPoint} = res.data;
    const asrServiceId = pipelineResponseConfig.find(task => task.taskType === 'asr').config[0].serviceId;

    const auth = {
      key: pipelineInferenceAPIEndPoint.inferenceApiKey.name,

      value: pipelineInferenceAPIEndPoint.inferenceApiKey.value,
    };
    const {callbackUrl} = pipelineInferenceAPIEndPoint;
    const data = Meteor.call('ASR_ComputeCall', callbackUrl, auth, sourceLanguage, asrServiceId, audioContent);
    return data;
  },
  async ASR_ConfigCall(sourceLanguage) {
    const configCallconfig = {
      pipelineTasks: [
        {
          taskType: 'asr',
          config: {
            language: {
              sourceLanguage,
            },
          },
        },
      ],
      pipelineRequestConfig: {
        pipelineId: '64392f96daac500b55c543cd',
      },
    };
    const res = Meteor.call('createPostRequest', Meteor.settings.public.API_ENDPOINT, configCallconfig, {
      headers: {
        userID: Meteor.settings.public.userID,
        ulcaApiKey: Meteor.settings.public.ulcaApiKey,
      },
    });
    return res;
  },
  async ASR_ComputeCall(callbackUrl, auth, sourceLanguage, asrServiceId, audioContent) {
    const computeCallConfig = {
      pipelineTasks: [
        {
          taskType: 'asr',
          config: {
            language: {
              sourceLanguage,
            },
            serviceId: asrServiceId,
            audioFormat: 'wav',
            samplingRate: 16000,
          },
        },
      ],
      inputData: {
        audio: [
          {
            audioContent,
          },
        ],
      },
    };
    const res = Meteor.call('createPostRequest', callbackUrl, computeCallConfig, {
      headers: {
        Authorization: auth.value,
      },
    });
    return res;
  },
  async NMT_handler(sourceLanguage, targetLanguage, textContent) {
    const res = Meteor.call('NMT_ConfigCall', sourceLanguage, targetLanguage);
    if (!res.isSuccess) return res;
    const {pipelineResponseConfig, pipelineInferenceAPIEndPoint} = res.data;

    const translationServiceId = pipelineResponseConfig.find(task => task.taskType === 'translation').config[0]
      .serviceId;

    const auth = {
      key: pipelineInferenceAPIEndPoint.inferenceApiKey.name,

      value: pipelineInferenceAPIEndPoint.inferenceApiKey.value,
    };
    const {callbackUrl} = pipelineInferenceAPIEndPoint;
    const data = Meteor.call(
      'NMT_ComputeCall',
      callbackUrl,
      auth,
      sourceLanguage,
      targetLanguage,
      translationServiceId,
      textContent,
    );
    return data;
  },
  async NMT_ConfigCall(sourceLanguage, targetLanguage) {
    const configCallconfig = {
      pipelineTasks: [
        {
          taskType: 'translation',
          config: {
            language: {
              sourceLanguage,
              targetLanguage,
            },
          },
        },
      ],
      pipelineRequestConfig: {
        pipelineId: '64392f96daac500b55c543cd',
      },
    };
    const res = Meteor.call('createPostRequest', Meteor.settings.public.API_ENDPOINT, configCallconfig, {
      headers: {
        userID: Meteor.settings.public.userID,
        ulcaApiKey: Meteor.settings.public.ulcaApiKey,
      },
    });
    return res;
  },
  async NMT_ComputeCall(callbackUrl, auth, sourceLanguage, targetLanguage, translationServiceId, textContent) {
    const computeCallConfig = {
      pipelineTasks: [
        {
          taskType: 'translation',
          config: {
            language: {
              sourceLanguage,
              targetLanguage,
            },
            serviceId: translationServiceId,
          },
        },
      ],
      inputData: {
        input: textContent.map(text => {
          return {source: text.source};
        }),
      },
    };
    const res = Meteor.call('createPostRequest', callbackUrl, computeCallConfig, {
      headers: {
        Authorization: auth.value,
      },
    });
    return res;
  },
  async createPostRequest(apiEndPoint, requestdata, headers) {
    try {
      const res = await axios.post(apiEndPoint, requestdata, headers);
      const responseIs = {isSuccess: true, message: 'Success', data: res.data};
      return responseIs;
    } catch (err) {
      const errorIs = {isSuccess: false, message: 'Failed', err};
      return errorIs;
    }
  },
});
