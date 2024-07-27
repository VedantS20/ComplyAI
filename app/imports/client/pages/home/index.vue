<template>
  <div class="page-container">
    <v-card>
      <v-container>
        <search-input-vue :loading="loading" @submit-form="searchQuery" :timeTaken="timeTaken" />
        <results v-if="response" :aiResponse="aiResponse" :semanticSearchResponse="semanticSearchResponse"></results>
      </v-container>
    </v-card>
    <v-card class="mb-16">
      <v-container class="mt-4">
        <div class="text-center mt-8">
          <h1 class="mb-4">Scan Product & Packaging</h1>
          <p class="mb-4">
            Be compliance ready with your products by quickly scanning your product and packages with our state of the
            art AI driven scanner.
          </p>
        </div>
        <div class="text-center">
          <v-btn to="/imageScanner" color="primary" x-large>
            <v-icon left>mdi-creation</v-icon>
            Scan Product
          </v-btn>
        </div>
      </v-container>
    </v-card>
    <v-footer absolute padless>
      <v-card flat tile width="100%" class="blue-grey lighten-5 text-center">
        <v-card-text class="text--blue-grey darken-4">
          Built with
          <v-icon color="red">mdi-heart</v-icon>
          by Team Fasalians
        </v-card-text>
      </v-card>
    </v-footer>
  </div>
</template>

<script>
  import results from './results.vue';
  import searchInputVue from './searchInput.vue';

  export default {
    components: {results, searchInputVue},
    data() {
      return {loading: false, response: false, semanticSearchResponse: '', aiResponse: '', timeTaken: undefined};
    },
    methods: {
      async searchQuery(queryText) {
        this.loading = true;
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

        this.aiResponse = await response.json();
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
        this.loading = false;
        this.response = true;
        this.timeTaken = (+new Date() - startTime) / 1000;
      },
    },
  };
</script>

<style lang="scss" scoped>
  .page-container {
    margin: 25px;
  }
</style>
