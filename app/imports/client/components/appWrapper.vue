<template>
  <div class="main-app-wrapper">
    <main>
      <language-selector />
      <slot></slot>
      <p class="about">
        Built with
        <v-icon color="red">mdi-heart</v-icon>
        by Team Fasalians for
        <a href="https://github.com/iamparthaonline/BFB-Hackathon-ONDC-2024">Build For Bharat Hackathon</a>
      </p>
    </main>
  </div>
</template>

<script>
  import languageSelector from './language.vue';

  export default {
    components: {languageSelector},
  };
</script>

<style lang="scss" scoped>
  .main-app-wrapper {
    margin: auto;
    padding-bottom: 80px;
    background-color: #f7f7fc;
    @media only screen and (min-width: 600px) {
      display: flex;
      // height: 100vh;
      align-items: center;
      justify-content: space-around;
      padding: 80px 20px;
    }
    main {
      font-family: 'Poppins', sans-serif;
      padding: 20px;
      display: inline-block;
      text-align: left;
      max-width: 500px;
      @media only screen and (min-width: 600px) {
        max-width: 720px;
      }
    }
    h1 {
      color: #7930f0;
      font-size: 26px;
      line-height: 38px;
      // min-height: 78px;
    }
    a {
      color: #7930f0;
      text-decoration: none;
      font-weight: 500;
    }
    h2 {
      font-size: 18px;
      line-height: 22px;
      margin-bottom: 8px;
      color: #7930f0;
    }
    p {
      font-size: 12px;
      margin: 0;
    }
    .about {
      margin-top: 30px;
    }
  }
</style>
