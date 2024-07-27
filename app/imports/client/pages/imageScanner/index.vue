<template>
  <v-card
    style="position: relative"
    height="100%"
    width="100%"
    color="#E3F2FD"
    class="d-flex flex-column justify-start align-center"
  >
    <v-btn fab absolute plain style="left: 0; top: 0" to="/">
      <v-icon>mdi-arrow-left</v-icon>
    </v-btn>
    <ImageCard v-if="!responseText" :showLoader="showLoader" :imageBase64="imageBase64" />
    <ResponseCard v-else :imageSourceURL="imageURL" :responseText="responseText" />
    <div v-if="imageBase64">
      <v-btn @click="uploadImageToImageKit" tile color="success" class="mt-6">
        <v-icon left>mdi-creation</v-icon>
        Scan Image
      </v-btn>
      <v-btn @click="goToHome" tile color="red" dark class="mt-6 ml-2">
        <v-icon left>mdi-reload</v-icon>
        Retake
      </v-btn>
    </div>

    <input type="file" @change="uploadImage" ref="file" multiple accept="image/*" style="display: none" />
    <div v-if="!isCameraActive && !imageBase64" style="position: absolute; bottom: 20px">
      <v-fab-transition>
        <v-btn
          @click="$refs.file.click()"
          key="mdi-share-variant"
          color="success"
          fab
          large
          dark
          class="v-btn--example"
        >
          <v-icon>mdi-upload</v-icon>
        </v-btn>
      </v-fab-transition>
      <v-fab-transition v-if="isCordova">
        <v-btn key="mdi-share-variant" @click="startCamera" color="primary" fab large dark class="v-btn--example ml-6">
          <v-icon>mdi-camera</v-icon>
        </v-btn>
      </v-fab-transition>
    </div>
    <div v-if="isCameraActive" style="position: absolute; bottom: 20px">
      <div v-if="!imageBase64">
        <v-fab-transition>
          <v-btn @click="goToHome" color="secondary" fab large dark class="v-btn--example">
            <v-icon>mdi-arrow-left</v-icon>
          </v-btn>
        </v-fab-transition>
        <v-fab-transition>
          <v-btn
            @click="takePicture"
            key="mdi-share-variant"
            color="secondary"
            fab
            large
            dark
            class="ml-6 v-btn--example"
          >
            <v-icon>mdi-camera</v-icon>
          </v-btn>
        </v-fab-transition>
        <v-fab-transition>
          <v-btn
            key="mdi-share-variant"
            @click="flipCamera"
            color="secondary"
            fab
            large
            dark
            class="v-btn--example ml-6"
          >
            <v-icon>mdi-camera-flip-outline</v-icon>
          </v-btn>
        </v-fab-transition>
      </div>
    </div>
  </v-card>
</template>

<script>
  import ImageCard from './imageCard.vue';
  import ResponseCard from './responseCard.vue';

  export default {
    components: {ImageCard, ResponseCard},
    data() {
      return {
        isCameraActive: false,
        imageBase64: undefined,
        showLoader: false,
        responseText: undefined,
        imageURL: undefined,
        isCordova: Meteor.isCordova,
      };
    },
    methods: {
      startCamera() {
        if (Meteor.isCordova) {
          let options = {
            x: 30,
            y: 40,
            width: window.screen.width * 0.85,
            height: window.screen.height * 0.67,
            camera: CameraPreview.CAMERA_DIRECTION.BACK,
            toBack: false,
            tapPhoto: true,
            tapFocus: false,
            previewDrag: false,
            storeToFile: false,
            disableExifHeaderStripping: false,
          };

          CameraPreview.startCamera(options, succ => {
            if (succ) {
              this.isCameraActive = true;
            }
          });
        }
      },
      takePicture() {
        CameraPreview.takePicture(base64PictureData => {
          /* code here */
          this.imageBase64 = `data:image/jpeg;base64,${base64PictureData}`;
          CameraPreview.hide();
          CameraPreview.stopCamera();
        });
      },
      flipCamera() {
        CameraPreview.switchCamera();
      },
      goToHome() {
        this.imageBase64 = undefined;
        if (Meteor.isCordova) {
          CameraPreview.hide();
          CameraPreview.stopCamera();
        }

        this.isCameraActive = false;
        this.responseText = undefined;
        this.imageURL = undefined;
      },
      retakeThePhoto() {
        this.imageBase64 = undefined;
        this.startCamera();
      },
      async uploadImage(event) {
        this.imageBase64 = undefined;
        const base64 = await this.convertBase64(event.target.files[0]);
        this.imageBase64 = base64;
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
        this.showLoader = true;
        Meteor.call('uploadImage', this.imageBase64, (error, response) => {
          if (response && !error) {
            Meteor.call('getAIResponseForScannedImage', response.url, (err, res) => {
              if (res && !err) {
                this.responseText = res;
                this.imageURL = response.url;
              }
              this.showLoader = false;
            });
          }
        });
      },
    },
  };
</script>

<style lang="scss"></style>
