<template>
  <v-dialog v-model="dialog" fullscreen hide-overlay transition="dialog-bottom-transition" width="600px" height="100%">
    <v-card class="card">
      <v-toolbar dark color="deep-purple lighten-1">
        <v-btn icon dark @click="$emit('closeModal')">
          <v-icon>mdi-close</v-icon>
        </v-btn>
      </v-toolbar>
      <pdfviewer class="tesstt" :source="source" :content="content" :page="pageNumber" />
    </v-card>
  </v-dialog>
</template>

<script>
  import pdfviewer from './pdfViewerV2.vue';

  export default {
    name: 'ModalPdfViewer',
    data() {
      return {
        dialog: false,
      };
    },
    props: {
      showModal: Boolean,
      source: String,
      content: String,
      pageNumber: Number,
    },
    components: {
      pdfviewer,
    },
    watch: {
      showModal() {
        this.dialog = this.showModal;
      },
    },
  };
</script>

<style lang="scss" scoped>
  .card {
    width: 100%;
    height: 100dvh;
    // overflow: hidden;
    .tesstt {
      height: calc(100dvh - 56px);
    }
  }
  .close-btn {
    padding: 16px;
    position: absolute;
    top: 0;
    right: 0;
  }
</style>
