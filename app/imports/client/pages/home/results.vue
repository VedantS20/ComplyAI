<template>
  <div class="mt-4 main-container">
    <v-tabs color="deep-purple accent-4" left v-model="active_tab">
      <v-tab>{{ _('result_page_ai_response') }}</v-tab>
      <v-tab>{{ _('result_page_knowledge_bank') }}</v-tab>
      <v-tab-item>
        <v-col sm="4" md-cols="6" class="mb-0">
          <v-select
            class="mt-3"
            :items="lang"
            v-model="selectedlang"
            item-text="name"
            item-value="value"
            label="Language"
            outlined
            :loading="loading"
            @change="$emit('changeLang', selectedlang)"
          ></v-select>
        </v-col>
        <div v-if="aiResponse.length" class="response">
          <template>
            <p v-for="(text, idx) in aiResponse" :key="idx" class="my-0">
              {{ text.target ? text.target : text.source }}
            </p>
          </template>

          <p class="my-4">
            {{ _('result_page_check_out') }}
            <v-btn outlined color="deep-purple accent-4" @click="testfn">{{ _('result_page_knowledge_bank') }}</v-btn>
          </p>
        </div>
      </v-tab-item>
      <v-tab-item>
        <div ref="test" class="results-list d-flex flex-row">
          <v-container class="card">
            <div class="d-flex flex-row flex-wrap">
              <result-card-vue
                v-for="(result, idx) in semanticSearchResponse"
                :key="idx"
                :idx="idx"
                left
                class="mb-4"
                :result="result.document"
                :sourceIdx="sourceIdx"
                @openPdfViewer="openPdfViewer"
              />
            </div>
          </v-container>
          <v-container v-if="source" class="pdf-container d-none d-sm-none d-md-block flex-grow" fluid>
            <pdfviewer :source="source" :content="content" :page="pageNumber" class="child" />
          </v-container>
        </div>
      </v-tab-item>
    </v-tabs>
  </div>
</template>

<script>
  import resultCardVue from './resultCard.vue';
  import pdfviewer from './pdfViewerV2.vue';

  export default {
    name: 'Results',
    props: {
      aiResponse: Array,
      semanticSearchResponse: Array,
      defaultSelected: String,
      loading: Boolean,
    },
    data() {
      return {
        source: null,
        pageNumber: null,
        sourceIdx: null,
        content: '',
        selectedlang: this.defaultSelected,
        lang: [
          {name: 'English', value: 'en'},
          {name: 'हिंदी', value: 'hi'},
          {name: 'ગુજરાતી', value: 'gu'},
          {name: 'मराठी', value: 'mr'},
          {name: 'ಕನ್ನಡ', value: 'kn'},
          {name: 'తెలుగు', value: 'te'},
          {name: 'தமிழ்', value: 'ta'},
        ],
        active_tab: 0,
      };
    },
    components: {resultCardVue, pdfviewer},
    methods: {
      openPdfViewer({source, pageNumber, sourceIdx, content}) {
        this.pageNumber = pageNumber;
        this.sourceIdx = sourceIdx;
        this.source = source;
        this.content = content;
      },
      testfn() {
        this.active_tab = 1;
        window.scrollTo(0, 0);
      },
    },
  };
</script>

<style lang="scss" scoped>
  .response {
    p {
      white-space: pre-line;
      text-wrap: wrap;
      text-align: justify;
    }
  }
  .results-list {
    height: calc(100dvh - 87px);
    overflow-y: auto;
    .card {
      max-width: 50%;

      @media only screen and (max-width: 600px) {
        max-width: calc(100dvw - 40px);
      }
    }
    .pdf-container {
      max-height: fit-content;
      position: relative;
      position: sticky;
      top: 0px;
      .child {
        height: fit-content;
      }
    }
  }
</style>
