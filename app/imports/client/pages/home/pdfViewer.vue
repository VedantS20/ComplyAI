<template>
  <v-container class="pdf-viewer">
    <div class="testss">
      <div class="pdf-container">
        <canvas ref="pdfCanvas" class="pdf-canvas"></canvas>
        <div ref="textLayer" id="text-layer" class="textLayer"></div>
      </div>
      <div class="btns text-center">
        <v-btn color="teal lighten-5" @click="previousPage">previous</v-btn>
        <v-btn color="teal lighten-5" @click="nextPage">next</v-btn>
      </div>
    </div>
  </v-container>
</template>
<script>
  import {getDocument} from 'pdfjs-dist/legacy/build/pdf';
  import 'pdfjs-dist/legacy/build/pdf.worker.entry';

  export default {
    name: 'PdfViewer',
    date() {
      return {
        pdfDocument: null,
        pdfPage: null,
        totalPage: 0,
        currentPage: 0,
        pageRendering: false,
      };
    },
    props: {
      source: String,
      pageNumber: Number,
      sourceIdx: Number,
    },
    watch: {
      sourceIdx() {
        this.loadPdf();
      },
    },
    computed: {
      nextDisable() {
        return this.currentPage >= this.totalPage;
      },
    },
    methods: {
      async loadPdf() {
        this.pageRendering = true;
        try {
          const loadingTask = getDocument(this.source);
          this.pdfDocument = await loadingTask.promise;
          this.totalPage = this.pdfDocument._pdfInfo.numPages;
          this.currentPage = this.pageNumber;
          await this.loadPage(this.pageNumber);
        } catch (e) {
          console.log(e);
        } finally {
          this.pageRendering = false;
        }
      },
      async loadPage(pageNumber) {
        this.pdfPage = await this.pdfDocument.getPage(pageNumber);
        this.renderPage();
      },
      nextPage() {
        if (this.currentPage >= this.totalPage) return;
        this.currentPage += 1;
        if (!this.pageRendering) this.loadPage(this.currentPage);
      },
      previousPage() {
        if (this.currentPage <= 1) return;
        this.currentPage -= 1;
        if (!this.pageRendering) this.loadPage(this.currentPage);
      },
      async renderPage() {
        const canvas = this.$refs.pdfCanvas;
        const context = canvas.getContext('2d');
        const viewport = this.pdfPage.getViewport({scale: 1});
        const outputScale = window.devicePixelRatio || 1;
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        const transform = outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null;
        const renderContext = {
          canvasContext: context,
          transform,
          viewport,
        };
        this.pdfPage.render(renderContext);
      },
    },
    mounted() {
      this.loadPdf();
    },
  };
</script>
<style lang="scss" scoped>
  .pdf-viewer {
    overflow: auto;
    max-height: calc(100dvh - 111px);
    position: relative;
    .testss {
      margin: auto;
      .pdf-container {
        margin-bottom: 36px;
        overflow: auto;
        .pdf-canvas {
          width: 100%;
        }
      }
      .btns {
        position: sticky;
        bottom: 0;
        height: fit-content;
      }
    }
  }
</style>
