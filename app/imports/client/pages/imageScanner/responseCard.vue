<template>
  <v-card width="85%" height="70%" color="white" class="mt-10 overflow-auto">
    <v-img contain :src="imageSourceURL" height="50%" width="100%"></v-img>
    <p v-if="responseText" class="text-caption font-weight-medium text-justify mx-2 mt-4">{{ responseText }}</p>
  </v-card>
</template>

<script>
  export default {
    name: 'ResponseCard',

    props: {
      imageSourceURL: String,
      responseText: String,
    },
  };
</script>

<style></style>
