<template>
  <app-wrapper ref="appWrapper">
    <div class="product-packaging-scanner-container">
      <nav-back @goback="goback" />
      <h1 v-if="status === 'INITIAL'">{{ _('query_page_title') }}</h1>
      <h1 v-else>{{ _('query_page_answer_to_your_query') }}</h1>
      <div v-if="status === 'INITIAL'">
        <p class="description">
          {{ _('query_page_description') }}
          <br />
          {{ _('query_page_description_example') }}
        </p>

        <div class="actions">
          <div :class="`action-item ${tab === 'WRITE' ? 'active' : ''}`" @click="changeTab('WRITE')" v-ripple>
            <div class="action-img"><img src="/keyboard.png" /></div>
            <h2>{{ _('query_page_i_want_to_write_button') }}</h2>
          </div>
          <div :class="`action-item ${tab === 'TALK' ? 'active' : ''}`" @click="changeTab('TALK')" v-ripple>
            <div class="action-img"><img src="/microphone.png" /></div>
            <h2>{{ _('query_page_i_want_to_talk_button') }}</h2>
          </div>
        </div>
        <div v-if="tab === 'WRITE'" class="write-container">
          <h2>{{ _('query_page_write_in_your_lang_text') }}</h2>

          <validation-observer ref="observer" v-slot="{invalid}">
            <form @submit.prevent="submit" class="mt-4">
              <validation-provider v-slot="{errors}" name="Query" rules="required|max:500">
                <v-textarea
                  filled
                  auto-grow
                  v-model="name"
                  outlined
                  :counter="500"
                  :error-messages="errors"
                  :label="_('query_page_enter_your_query_text')"
                  required
                  @input="$refs.observer.validate()"
                ></v-textarea>
                <v-select
                  class="mt-3"
                  :items="lang"
                  v-model="defaultSelected"
                  item-text="name"
                  item-value="value"
                  :label="_('query_page_language_text')"
                  outlined
                ></v-select>
              </validation-provider>
              <div class="text-center mt-2">
                <v-btn
                  block
                  class="mr-4"
                  type="submit"
                  color="#7930f0"
                  :loading="loadingSubmitBtn"
                  :disabled="invalid || loading"
                >
                  {{ _('query_page_submit_button') }}
                </v-btn>
                <!-- <v-btn @click="clear" :disabled="loading" color="#ccc">Clear</v-btn> -->
              </div>
            </form>
          </validation-observer>
        </div>
        <div v-else-if="tab === 'TALK'" class="write-container talk-container">
          <h2>{{ _('query_page_speak_in_your_lang_text') }}</h2>
          <div class="recorder-wrapper">
            <div class="recorder-time-action">
              <div class="recorder-time">{{ timerString }}</div>
              <v-btn
                block
                class="mr-4"
                type="submit"
                color="red"
                :loading="loading"
                :disabled="recording !== 'RECORDING'"
                outlined
                @click="cancel"
              >
                {{ _('query_page_cancel_button') }}
              </v-btn>
            </div>
            <div v-ripple class="record-action" v-if="recording === 'INITIAL'" @click="startRecording">
              <img src="/start-record.png" alt="" />
              <span>{{ _('query_page_start_recording_text') }}</span>
            </div>
            <div v-ripple class="record-action" v-else-if="recording === 'RECORDING'" @click="save">
              <img src="/stop-record.png" alt="" />
              <span>{{ _('query_page_finish_recording') }}</span>
            </div>
            <div class="record-action" v-else-if="recording === 'PROCESSING'">
              <img src="/stop-record.png" alt="" />
              <span>{{ _('query_page_processing') }}</span>
            </div>
          </div>
          <figure v-if="audioURL" class="my-2">
            <audio controls :src="audioURL"></audio>
            <div class="my-2">
              <v-btn type="submit" color="red" outlined @click="reset">{{ _('query_page_reset') }}</v-btn>
            </div>
          </figure>
          <v-select
            class="mt-3"
            :items="lang"
            v-model="defaultSelected"
            item-text="name"
            item-value="value"
            :label="_('query_page_language_text')"
            outlined
          ></v-select>
          <v-btn
            block
            class="mr-4"
            type="submit"
            color="#7930f0"
            :loading="loadingSubmitBtn"
            :disabled="disabledAudioSubBtn"
            @click="searchQueryWithAudio"
          >
            {{ _('query_page_submit') }}
          </v-btn>
        </div>
      </div>
      <div v-else-if="status === 'LOADING' || status === 'SUCCESS' || status === 'REPORT'" class="scanning-image">
        <div class="original-query mb-3">
          <h2>{{ _('query_page_your_query_text') }}</h2>
          <figure v-if="audioURL" class="my-2">
            <figcaption>{{ _('query_page_listen_your_query_text') }}</figcaption>
            <audio controls :src="audioURL"></audio>
          </figure>

          <p>"{{ audioURL ? ASTValue : name }}"</p>
        </div>
        <div v-if="status === 'REPORT'" class="report-paragraph">
          <v-alert v-if="timeTaken && !loading" dense text type="success">
            {{ _('query_page_processed_your_query_in') }}
            <strong>{{ parseInt(timeTaken) }} {{ _('query_page_second') }}</strong>
          </v-alert>

          <results
            v-if="response"
            :aiResponse="aiResponse"
            :semanticSearchResponse="semanticSearchResponse"
            :defaultSelected="defaultSelected"
            :loading="langChangeLoading"
            @changeLang="getResponseInSelectedLang"
          ></results>
          <v-btn block @click="status = 'INITIAL'" class="mt-3" outlined color="#4b06ba">
            {{ _('result_page_ask_another_question') }}
          </v-btn>
        </div>
        <v-overlay v-if="status === 'LOADING' || status === 'SUCCESS'">
          <div class="loader-container" v-if="status === 'LOADING'">
            <h2>{{ _('query_page_please_wait_text') }}</h2>
            <img src="/ai.gif" class="loader" alt="" />
            <p>
              {{ audioURL ? 'Processing your audio text query' : 'Processing your text query' }}
              <br />
              {{ _('query_page_processing_image_disclaimer') }}
            </p>
          </div>
          <div class="loader-container" v-else>
            <h2>{{ _('query_page_processing_complete_text') }}</h2>
            <img src="/verified.gif" class="loader" alt="" />
            <p>{{ _('query_page_completed_scanning_text') }}</p>
          </div>
        </v-overlay>
      </div>
      <div v-else>
        <h4>
          {{ _('query_page_umm_text') }}
          <br />
          {{ _('query_page_error_message') }}
        </h4>

        <v-btn block @click="status = 'INITIAL'" class="mb-3" outlined color="#4b06ba">
          {{ _('query_page_try_again_button') }}
        </v-btn>
        <h3>{{ _('query_page_image_upload_instruction_text') }}</h3>
        <ul>
          <li>
            <strong>{{ _('query_page_image_upload_instruction_point1_title') }}</strong>
            {{ _('query_page_image_upload_instruction_point1_description') }}
          </li>
          <li>
            <strong>{{ _('query_page_image_upload_instruction_point2_title') }}</strong>
            {{ _('query_page_image_upload_instruction_point2_description') }}
          </li>
          <li>
            <strong>{{ _('query_page_image_upload_instruction_point3_title') }}</strong>
            {{ _('query_page_image_upload_instruction_point3_description') }}
          </li>
        </ul>
      </div>
    </div>
    <p class="my-3">
      {{ _('query_page_bhashini_text') }}
      <a href="https://bhashini.gov.in/">Bhashini</a>
    </p>
  </app-wrapper>
</template>

<script>
  import {required, digits, max} from 'vee-validate/dist/rules';
  import {extend, ValidationObserver, ValidationProvider, setInteractionMode} from 'vee-validate';
  import {MediaRecorder, register} from 'extendable-media-recorder';
  import {connect} from 'extendable-media-recorder-wav-encoder';
  import appWrapper from '../../../components/appWrapper.vue';
  import navBack from '../../../components/navBack.vue';
  import results from '../../home/results.vue';

  setInteractionMode('eager');

  extend('digits', {
    ...digits,
    message: '{_field_} needs to be {length} digits. ({_value_})',
  });

  extend('required', {
    ...required,
    message: '{_field_} can not be empty',
  });

  extend('max', {
    ...max,
    message: '{_field_} may not be greater than {length} characters',
  });

  export default {
    components: {
      ValidationProvider,
      ValidationObserver,
      appWrapper,
      navBack,
      results,
    },
    props: {},
    data: () => ({
      name: '',
      status: 'INITIAL', // LOADING || SUCCESS
      tab: 'WRITE',
      recording: 'INITIAL', // RECORDING || PROCESSING
      timeTaken: Number,
      response: false,
      semanticSearchResponse: '',
      aiResponse: [],
      loading: false, // to avoid error msg
      invalid: false, // to avoid error msg
      defaultSelected: localStorage.getItem('language') || 'en',
      lang: [
        {name: 'English', value: 'en'},
        {name: 'हिंदी', value: 'hi'},
        {name: 'ગુજરાતી', value: 'gu'},
        {name: 'मराठी', value: 'mr'},
        {name: 'ಕನ್ನಡ', value: 'kn'},
        {name: 'తెలుగు', value: 'te'},
        {name: 'தமிழ்', value: 'ta'},
      ],
      constraints: {audio: true},
      timerString: '00:00',
      mediaRecorder: null,
      minute: 0,
      second: 0,
      chunks: [],
      audioContent: null,
      saveAudioString: false,
      loadingSubmitBtn: false,
      audioURL: null,
      audioSearch: false,
      ASTValue: null,
      langChangeLoading: false,
      disableSubmitBtn: false,
    }),
    async mounted() {
      this.port = await connect();
      await register(this.port);
    },
    computed: {
      disabledAudioSubBtn() {
        if (this.recording === 'INITIAL' && this.audioContent) return false;
        return true;
      },
    },
    methods: {
      goback() {
        if (this.status === 'REPORT') {
          this.status = 'INITIAL';
          this.audioContent = null;
          this.audioURL = null;
          this.timerString = '00:00';
        } else this.$router.go(-1);
      },

      async submit() {
        this.loadingSubmitBtn = true;
        this.status = 'LOADING';
        this.$refs.observer.validate();
        if (this.defaultSelected !== 'en') {
          await Meteor.call('NMT_handler', this.defaultSelected, 'en', [{source: this.name.trim()}], (err, data) => {
            if (!err) {
              console.log('data', data);
              this.loadingSubmitBtn = false;
              if (data.isSuccess) {
                console.log(
                  '🚀 ~ awaitMeteor.call ~ data.data.pipelineResponse[1].output[0].source:',
                  data.data.pipelineResponse[0].output[0].source,
                );
                this.name = data.data.pipelineResponse[0].output[0].source;
                this.searchQuery(data.data.pipelineResponse[0].output[0].target);
                // this.ASTValue = data.data.pipelineResponse[1].output[0].source;
                // this.searchQuery(data.data.pipelineResponse[1].output[0].target);
              }
            } else {
              this.loadingSubmitBtn = false;
              this.status = 'ERROR';
              console.log('error', err);
            }
          });
        } else {
          this.searchQuery(this.name);
        }
      },
      clear() {
        this.name = '';
        this.$refs.observer.reset();
      },
      changeTab(tab) {
        this.tab = tab;
      },
      async searchQuery(queryText) {
        this.loadingSubmitBtn = false;
        this.status = 'LOADING';
        this.semanticSearchResponse = '';
        this.aiResponse = [];
        this.response = false;
        const startTime = +new Date();
        const response = await fetch(`${Meteor.settings.public.API_HOST}/generate-AI-response`, {
          method: 'POST',
          cache: 'no-cache',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({queryText, k: 25}),
        });
        const bareText = await response.json();
        this.responseSrc = bareText
          .split('\n')
          .filter(text => text.trim())
          .map(text => {
            return {source: text.trim()};
          });
        console.log('🚀 ~ searchQuery ~  this.responseSrc:', this.responseSrc);
        await this.getResponseInSelectedLang(this.defaultSelected);
        const semanticSearch = await fetch(
          `${Meteor.settings.public.API_HOST}/semantic-search?query=${queryText}&k=25`,
          {
            method: 'POST',
            cache: 'no-cache',
            headers: {
              'Content-Type': 'application/json',
            },
          },
        );

        this.semanticSearchResponse = await semanticSearch.json();
        this.semanticSearchResponse.sort((a, b) => a.document.similarity_score - b.document.similarity_score);
        this.status = 'REPORT';
        this.response = true;
        this.timeTaken = (+new Date() - startTime) / 1000;
      },
      startRecording() {
        if (navigator.mediaDevices.getUserMedia) {
          this.audioURL = null;
          navigator.mediaDevices.getUserMedia(this.constraints).then(this.onSuccess, this.onError);
        } else {
          this.recording = 'INITIAL';
          console.log('MediaDevices.getUserMedia() not supported on your browser!');
        }
      },
      async stopRecording() {
        await this.mediaRecorder.stop();
        console.log('🚀 ~ stopRecording ~ this.recording:', this.recording);
      },
      onSuccess(stream) {
        this.timerString = '00:00';
        setTimeout(() => {
          this.stopWatch();
        }, 1000);
        this.recording = 'RECORDING';
        this.mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/wav'});

        this.mediaRecorder.start();
        this.mediaRecorder.onstop = async () => {
          this.recording = 'PROCESSING';
          const tracks = stream.getAudioTracks();
          tracks.forEach(track => track.stop());
          if (this.saveAudioString) {
            const blob = new Blob(this.chunks, {type: 'audio/wav'});
            console.log('🚀 ~ this.mediaRecorder.onstop= ~ this.mediaRecorder.mimeType:', this.mediaRecorder.mimeType);
            this.audioURL = window.URL.createObjectURL(blob);
            console.log('🚀 ~ onSuccess ~ audioURL:', this.audioURL);

            const audioString = await this.audioToBase64(blob);
            [, this.audioContent] = audioString.split(',');

            console.log('🚀 ~ this.mediaRecorder.onstop= ~ audioString:', this.audioContent);
          }
          this.chunks = [];
          this.recording = 'INITIAL';
          console.log('🚀 ~ this.mediaRecorder.onstop= ~ this.recording:', this.recording);
        };

        this.mediaRecorder.ondataavailable = e => {
          this.chunks.push(e.data);
        };
      },
      onError(err) {
        this.recording = 'INITIAL';
        console.log('The following error occured: ', err);
      },
      stopWatch() {
        if (this.recording !== 'RECORDING') {
          this.minute = 0;
          this.second = 0;
          return;
        }
        this.second += 1;

        if (this.second === 60) {
          this.minute += 1;
          this.second = 0;
        }

        if (this.minute === 60) {
          this.minute = 0;
          this.second = 0;
        }

        let minString = this.minute;
        let secString = this.second;

        if (this.minute < 10) {
          minString = `0${minString}`;
        }

        if (this.second < 10) {
          secString = `0${secString}`;
        }
        this.timerString = `${minString}:${secString}`;
        setTimeout(this.stopWatch, 1000);
      },
      audioToBase64(blob) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onerror = reject;
          reader.onload = e => resolve(e.target.result);
          reader.readAsDataURL(blob);
        });
      },
      cancel() {
        this.saveAudioString = false;
        this.minute = 0;
        this.second = 0;
        this.timerString = '00:00';
        this.audioContent = null;
        this.stopRecording();
      },
      save() {
        this.saveAudioString = true;
        this.stopRecording();
      },
      reset() {
        this.saveAudioString = false;
        this.minute = 0;
        this.second = 0;
        this.timerString = '00:00';
        this.audioContent = null;
        this.audioURL = null;
      },
      async searchQueryWithAudio() {
        this.loadingSubmitBtn = true;
        this.status = 'LOADING';
        this.audioSearch = true;
        if (this.defaultSelected !== 'en') {
          await Meteor.call('ASR_NMT_handler', this.defaultSelected, this.audioContent, (err, data) => {
            if (!err) {
              console.log('data', data);
              this.loadingSubmitBtn = false;
              if (data.isSuccess) {
                console.log(
                  '🚀 ~ awaitMeteor.call ~ data.data.pipelineResponse[1].output[0].target:',
                  data.data.pipelineResponse[1].output[0].target,
                );
                this.ASTValue = data.data.pipelineResponse[1].output[0].source;
                this.searchQuery(data.data.pipelineResponse[1].output[0].target);
              }
            } else {
              this.loadingSubmitBtn = false;
              this.status = 'ERROR';

              console.log('error', err);
            }
          });
        } else {
          await Meteor.call('ASR_handler', this.defaultSelected, this.audioContent, (err, data) => {
            if (!err) {
              console.log('data', data);
              this.loadingSubmitBtn = false;
              if (data.isSuccess) {
                console.log(
                  '🚀 ~ awaitMeteor.call ~ data.data.pipelineResponse[0].output[0].source',
                  data.data.pipelineResponse[0].output[0].source,
                );

                this.ASTValue = data.data.pipelineResponse[0].output[0].source;
                this.searchQuery(this.ASTValue);
              }
            } else {
              this.loadingSubmitBtn = false;
              this.status = 'ERROR';
              console.log('error', err);
            }
          });
        }
      },
      async getResponseInSelectedLang(lang) {
        console.log('🚀 ~ getResponseInSelectedLang ~ lang:', lang);
        this.langChangeLoading = true;
        if (lang === 'en') {
          this.aiResponse = this.responseSrc;
          this.langChangeLoading = false;
          console.log('🚀 ~ getResponseInSelectedLang ~ this.loadingSubmitBtn:', this.loadingSubmitBtn);
          return;
        }
        await Meteor.call('NMT_handler', 'en', lang, this.responseSrc, (err, data) => {
          if (!err) {
            console.log('data', data);
            if (data.isSuccess) {
              console.log(
                '🚀 ~ awaitMeteor.call ~ data.data.pipelineResponse[1].output:',
                data.data.pipelineResponse[0].output,
              );
              this.aiResponse = data.data.pipelineResponse[0].output;
              this.langChangeLoading = false;
              // this.status = 'REPORT';
            }
          } else {
            this.langChangeLoading = false;
            this.status = 'ERROR';

            console.log('error', err);
          }
        });
      },
    },
  };
</script>

<style lang="scss" scoped>
  .product-packaging-scanner-container {
    .scanning-image {
      text-align: center;
      margin: 20px 0;
      position: relative;
      .report-paragraph {
        background: #fff;
        padding: 16px;
        border-radius: 12px;
        font-size: 16px;
        display: block;
        text-align: left;
        font-weight: 400;
        @media only screen and (min-width: 600px) {
          margin: 0px -100px;
          p {
            font-size: 16px;
          }
        }
        @media only screen and (max-width: 600px) {
          max-width: calc(100dvw - 40px);
        }
        h3 {
          margin-top: 0;
        }
      }
      .uploaded-img {
        width: 300px;
        height: 400px;
        border-radius: 12px;
        object-fit: cover;
      }
      .success-report {
        width: 200px;
        height: 200px;
        position: absolute;
        opacity: 0.1;
        display: flex;
        align-self: center;
        bottom: 20px;
        right: 20px;
        filter: grayscale(1);
      }
      .loader-container {
        padding: 20px;
        background-color: #fff;
        border-radius: 12px;
        width: 300px;
        p,
        h2 {
          color: #444;
        }
        p {
          font-size: 12px;
        }
        .loader {
          width: 100px;
          height: 100px;
          object-fit: cover;
        }
      }
    }
    .description {
      font-size: 14px;
      margin: 16px 0;
    }
    ul {
      margin: 20px 0;
      font-size: 12px;
    }
    h3,
    h4 {
      margin: 20px 0;
    }
    .actions {
      display: flex;
      gap: 16px;
      align-items: center;
      margin: 24px 0;

      .action-item {
        background: #fff;
        border-radius: 12px;
        padding: 8px 16px;
        cursor: pointer;
        box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
        &:hover {
          box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;
        }
        display: flex;
        align-items: center;
        gap: 8px;
        h2 {
          font-size: 12px;
          margin: 0;
        }
        .action-img {
          padding: 10px;
          background-color: #e1e0fd;
          border-radius: 50%;
          display: inline-flex;
          align-items: center;
          justify-content: space-around;
          img {
            width: 16px;
            height: 16px;
          }
        }

        &.active {
          background-color: #7930f0;
          h2 {
            color: #fff;
          }
          .action-img {
            background-color: #fff;
          }
        }
      }
    }
    .write-container {
      background-color: #fff;
      .v-input .v-label {
        color: #7930f0;
      }
      h2 {
        color: #444;
        text-align: center;
        margin: 16px 0;
      }
      &.talk-container {
        text-align: center;
        .recorder-wrapper {
          display: flex;
          align-items: center;
          justify-content: space-around;
        }
        .recorder-time {
          font-size: 24px;
          font-family: courier;
          margin: 20px 0;
          padding: 0 40px;
        }
        .record-action {
          display: flex;
          align-items: center;
          flex-direction: column;
          background: #e1e0fd;
          display: inline-flex;
          border-radius: 12px;
          padding: 12px;
          gap: 12px;
          margin: 20px 0;
          img {
            width: 40px;
          }
          span {
            color: #7930f0;
            font-weight: bold;
          }
        }
      }
      border-radius: 12px;
      padding: 8px 16px;
      box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
      button {
        color: #fff;
      }
    }
    .original-query {
      background-color: #fff;
      border-radius: 12px;
      padding: 24px;
      p {
        font-size: 16px;
      }
    }
  }
</style>
