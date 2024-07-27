<template>
  <iframe id="pdf-js-viewer" :src="getSrc" title="webviewer" frameborder="0" ref="viewer"></iframe>
</template>

<script>
  export default {
    props: {
      source: String,
      content: String,
      page: Number,
    },
    computed: {
      getSrc() {
        console.log('this.page', this.page);
        return `/pdfjs/web/viewer.html?file=${encodeURIComponent(this.mask)}`;
      },
      mask() {
        return `${window.location.origin}/${this.source}#page=${this.page}`;
      },
    },
    methods: {
      find() {
        const iframeDocument = this.$refs.viewer.contentWindow;

        iframeDocument.PDFViewerApplication.eventBus.dispatch('find', {
          caseSensitive: false,
          findPrevious: undefined,
          highlightAll: true,
          phraseSearch: false,
          query: this.content,
        });
      },
    },
    mounted() {
      this.$refs.viewer.addEventListener('load', () => {
        this.find();
      });
    },
  };
</script>

<style lang="scss" scoped>
  #pdf-js-viewer {
    width: 100%;
    @media only screen and (min-width: 600px) {
      height: 100%;
    }
  }
</style>
