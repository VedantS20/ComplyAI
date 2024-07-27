<template>
  <v-app>
    <v-main>
      <v-responsive height="100%">
        <router-view></router-view>
      </v-responsive>
    </v-main>
  </v-app>
</template>

<script>
  export default {
    name: 'App',
  };
</script>

<style lang="scss" scoped>
  .v-responsive {
    overflow: auto;
    background-color: rgb(227, 242, 253);
  }
</style>
