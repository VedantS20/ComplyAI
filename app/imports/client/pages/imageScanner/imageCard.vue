<template>
  <v-card width="85%" height="70%" color="white" class="mt-12">
    <ScanningOverlay v-if="showLoader" />
    <InstructionSection v-if="!imageBase64" />
    <v-img contain v-else :src="imageBase64" height="100%" width="100%"></v-img>
  </v-card>
</template>

<script>
  import InstructionSection from './instructionSection.vue';
  import ScanningOverlay from './scanningOverlay.vue';

  export default {
    name: 'ImageCard',
    components: {InstructionSection, ScanningOverlay},
    props: {
      showLoader: Boolean,
      imageBase64: String,
    },
  };
</script>

<style></style>
