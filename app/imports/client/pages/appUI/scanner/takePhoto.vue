<template>
  <div class="d-flex flex-column justify-center" style="width: 100%">
    <nav-back @goback="$router.go(-1)" />
    <video v-show="!image" ref="video" width="640" height="480" autoplay></video>
    <img v-show="image" ref="photo" :src="image" />
    <canvas ref="canvas" width="640" height="480" style="display: none"></canvas>
    <v-row class="mt-3" align-content="center" justify="center">
      <v-btn outlined color="deep-purple lighten-1" class="mt-2" @click="takePhoto">
        {{ !image ? _('scanner_page_take_photo_button_text') : _('scanner_page_retake_photo_button_text') }}
      </v-btn>
      <v-btn v-if="image" outlined color="deep-purple lighten-1" class="mt-2 ml-2" @click="submitImage">
        {{ _('scanner_page_submit_image_text') }}
      </v-btn>
    </v-row>
  </div>
</template>

<script>
  export default {
    name: 'CameraComponent',
    data() {
      return {
        image: undefined,
      };
    },
    mounted() {
      this.initCamera();
    },

    methods: {
      async initCamera() {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({video: true});
          this.$refs.video.srcObject = stream;
          this.stream = stream;
        } catch (err) {
          console.error('Error accessing the camera: ', err);
        }
      },
      takePhoto() {
        if (!this.image) {
          const {canvas} = this.$refs;
          const context = canvas.getContext('2d');
          context.drawImage(this.$refs.video, 0, 0, canvas.width, canvas.height);
          const data = canvas.toDataURL('image/png');
          this.image = data;
          this.stopCamera();
        } else {
          this.image = undefined;
          this.initCamera();
        }
      },
      stopCamera() {
        if (this.stream) {
          const tracks = this.stream.getTracks();
          tracks.forEach(track => track.stop());
        }
      },
      submitImage() {
        this.$emit('imageClick', this.image);
      },
    },
    beforeUnmount() {
      this.stopCamera();
    },
  };
</script>

<style></style>
