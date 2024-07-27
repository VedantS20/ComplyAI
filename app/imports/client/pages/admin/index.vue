<template>
  <div>
    <input v-model="temperature" type="range" id="volume" name="volume" min="0" max="1" step="0.1" />
    <label for="volume">{{ temperature }}</label>
    <v-btn @click="saveToDB">Save</v-btn>
  </div>
</template>

<script>
  export default {
    data() {
      return {
        url: undefined,
        temperature: 0,
      };
    },
    mounted() {
      this.getTemp();
    },
    methods: {
      saveToDB() {
        console.log(this.temperature);
        Meteor.call('saveTemp', this.temperature, (err, res) => {
          console.log(err, res);
        });
      },
      getTemp() {
        Meteor.call('getTemp', (err, res) => {
          if (res && !err) this.temperature = res.toString();
        });
      },
    },
  };
</script>

<style></style>
