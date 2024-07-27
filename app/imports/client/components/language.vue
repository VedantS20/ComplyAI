<template>
  <div class="settings">
    <v-overlay v-if="show">
      <div class="settings-form-container">
        <v-container fluid>
          <v-select
            value="English"
            :items="lang"
            v-model="defaultSelected"
            item-text="name"
            item-value="value"
            :label="_('query_page_language_text')"
          ></v-select>

          <v-btn block class="mb-3" color="#4b06ba" @click="changeLanguage">Save</v-btn>
          <v-btn block outlined @click="toggleLanguageSelector">Close</v-btn>
        </v-container>
      </div>
    </v-overlay>

    <v-btn @click="toggleLanguageSelector" icon>
      <v-icon>mdi-cog</v-icon>
    </v-btn>
  </div>
</template>

<script>
  export default {
    data() {
      return {
        show: false,
        lang: [
          {name: 'English', value: 'en'},
          {name: 'हिंदी', value: 'hi'},
          {name: 'ગુજરાતી', value: 'gu'},
          {name: 'मराठी', value: 'mr'},
          {name: 'ಕನ್ನಡ', value: 'kn'},
          {name: 'తెలుగు', value: 'te'},
          {name: 'தமிழ்', value: 'ta'},
        ],
        defaultSelected: localStorage.getItem('language') || 'en',
      };
    },
    methods: {
      toggleLanguageSelector() {
        this.show = !this.show;
      },
      changeLanguage() {
        this.setLocale(this.defaultSelected);
        this.show = false;
      },
    },
  };
</script>

<style scoped lang="scss">
  .settings {
    position: absolute;
    top: 20px;
    right: 20px;
    .settings-form-container {
      max-width: 400px;

      @media only screen and (max-width: 600px) {
        max-width: 300px;
      }
      .select .v-input__control {
        overflow: hidden;
      }
      background: #444;
      padding: 20px;
      box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;
      border-radius: 10px;
    }
  }
</style>
