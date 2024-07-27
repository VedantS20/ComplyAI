<template>
  <v-card elevation="0" height="100%" class="d-flex flex-column justify-center align-center">
    <v-card height="60%" elevation="0" class="d-flex flex-column justify-center align-center">
      <v-card elevation="0" class="d-flex flex-column justify-center align-center ma-2">
        <v-img max-height="80" max-width="80" src="/imageScannerlogos/takephoto.png"></v-img>
        <p class="text-overline font-weight-bold">Take a Photo</p>
      </v-card>
      <p class="text-overline font-weight-black ma-0">OR</p>
      <v-card elevation="0" class="d-flex flex-column justify-center align-center mt-6">
        <v-img max-height="80" max-width="80" src="/imageScannerlogos/uploadimage.png"></v-img>
        <p class="text-overline font-weight-bold">Upload an Image</p>
      </v-card>
    </v-card>
    <v-card elevation="0">
      <ul class="text-caption mr-1 pr-2 text-justify">
        <li>
          <strong>Ensure Good Lighting</strong>
          : Make sure the image is taken in well-lit conditions to enhance the readability and accuracy of text and
          details on the product's backside.
        </li>
        <li>
          <strong>High Resolution</strong>
          : Use a high-resolution camera setting to capture the image.
        </li>
        <li>
          <strong>No Obstructions</strong>
          : Ensure that nothing is covering any part of the product's backside. Hands, fingers, or other objects should
          not obstruct the view of the label.
        </li>
      </ul>
    </v-card>
  </v-card>
</template>

<script>
  export default {
    name: 'InstructionSection',
  };
</script>

<style></style>
