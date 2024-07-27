<template>
  <app-wrapper>
    <div>
      <nav-back @goback="$router.go(-1)" />
      <div v-if="!openCamera" class="product-packaging-scanner-container">
        <h1>{{ _('scanner_page_title') }}</h1>
        <div v-if="status === 'INITIAL'">
          <p class="description">
            {{ _('scanner_page_image_upload_instruction') }}
          </p>
          <ul>
            <li>
              <strong>{{ _('scanner_page_image_quality_instruction_point1_title') }}</strong>
              {{ _('scanner_page_image_quality_instruction_point1_description') }}
            </li>
            <li>
              <strong>{{ _('scanner_page_image_quality_instruction_point2_title') }}</strong>
              {{ _('scanner_page_image_quality_instruction_point2_description') }}
            </li>
            <li>
              <strong>{{ _('scanner_page_image_quality_instruction_point3_title') }}</strong>
              {{ _('scanner_page_image_quality_instruction_point3_description') }}
            </li>
          </ul>
          <div class="actions">
            <input type="file" @change="uploadImage" ref="file" multiple accept="image/*" style="display: none" />
            <div class="action-item" @click="$refs.file.click()" v-ripple>
              <div class="action-img"><img src="/photo.png" /></div>
              <h2>{{ _('scanner_page_upload_image_button_title') }}</h2>
              <p>{{ _('scanner_page_upload_image_button_description') }}</p>
            </div>
            <div class="action-item" v-ripple @click="openCamera = true">
              <div class="action-img"><img src="/camera.png" /></div>
              <h2>{{ _('scanner_page_open_camera_button_title') }}</h2>
              <p>{{ _('scanner_page_open_camera_button_description') }}</p>
            </div>
          </div>
        </div>
        <div v-else-if="status === 'LOADING' || status === 'SUCCESS' || status === 'REPORT'" class="scanning-image">
          <img
            :src="imageBase64"
            :class="`uploaded-img ${status === 'LOADING' || status === 'SUCCESS' ? '' : 'success-report'}`"
            alt=""
          />

          <div v-if="status === 'REPORT'" class="report-paragraph">
            <h3>{{ _('scanner_page_report_for_your_package_text') }}</h3>
            <p style="white-space: pre-line">
              {{ responseText }}
            </p>
            <v-btn block @click="status = 'INITIAL'" class="mt-3" outlined color="#4b06ba">
              {{ _('scanner_page_check_another_text') }}
            </v-btn>
          </div>
          <v-overlay v-if="status === 'LOADING' || status === 'SUCCESS'">
            <div class="loader-container" v-if="status === 'LOADING'">
              <h2>{{ _('scanner_page_please_wait_text') }}</h2>
              <img src="/search.gif" class="loader" alt="" />
              <p>
                {{ _('scanner_page_processing_image_text') }}
                <br />
                {{ _('scanner_page_note_text') }}
              </p>
            </div>
            <div class="loader-container" v-else>
              <h2>{{ _('scanner_page_process_completed_text') }}</h2>
              <img src="/verified.gif" class="loader" alt="" />
              <p>{{ _('scanner_page_scanning_completed_text') }}</p>
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
      <CameraComponent v-else @imageClick="getImageClickedFromCamera" />
    </div>
  </app-wrapper>
</template>

<script>
  import appWrapper from '../../../components/appWrapper.vue';
  import navBack from '../../../components/navBack.vue';
  import CameraComponent from './takePhoto.vue';

  export default {
    components: {appWrapper, navBack, CameraComponent},
    data() {
      return {
        status: 'INITIAL', // LOADING || SUCCESS
        imageBase64: undefined,
        responseText: undefined,
        openCamera: false,
      };
    },

    methods: {
      process() {
        this.status = 'LOADING';
        setTimeout(() => {
          this.status = 'SUCCESS';
        }, 3000);
        setTimeout(() => {
          this.status = 'REPORT';
        }, 6000);
      },

      async uploadImage(event) {
        this.imageBase64 = undefined;
        const base64 = await this.convertBase64(event.target.files[0]);
        this.imageBase64 = base64;
        this.uploadImageToImageKit();
      },
      convertBase64(file) {
        return new Promise((resolve, reject) => {
          const fileReader = new FileReader();
          fileReader.readAsDataURL(file);

          fileReader.onload = () => {
            resolve(fileReader.result);
          };

          fileReader.onerror = error => {
            reject(error);
          };
        });
      },
      uploadImageToImageKit() {
        this.status = 'LOADING';
        Meteor.call('uploadImage', this.imageBase64, (error, response) => {
          if (response && !error) {
            Meteor.call('getAIResponseForScannedImage', response.url, (err, res) => {
              if (res) {
                this.status = 'SUCCESS';
                setTimeout(() => {
                  this.responseText = res;
                  this.imageURL = response.url;
                  this.status = 'REPORT';
                }, 2000);
              } else this.status = 'ERROR';
            });
          }
        });
      },
      getImageClickedFromCamera(image) {
        this.openCamera = false;
        this.imageBase64 = image;
        this.uploadImageToImageKit();
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
          p {
            font-size: 16px;
          }
        }
        h3 {
          margin-top: 0;
          margin-bottom: 210px;
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
        display: flex;
        align-self: center;
        top: 60px;
        right: calc(50% - 100px);
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
        padding: 16px;
        cursor: pointer;
        min-width: 135px;
        box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
        &:hover {
          box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;
        }
        h2 {
          font-size: 16px;
        }
        .action-img {
          padding: 10px;
          background-color: #e1e0fd;
          border-radius: 50%;
          display: inline-flex;
          align-items: center;
          justify-content: space-around;
          margin-bottom: 14px;
          img {
            width: 24px;
            height: 24px;
          }
        }
      }
    }
  }
</style>
