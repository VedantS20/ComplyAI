<template>
  <app-wrapper>
    <div class="dashboard-container">
      <h1 id="heading"></h1>
      <ul class="meta-list">
        <li>✓ {{ _('dashboard_page_feature_list_point1') }}</li>
        <li>✓ {{ _('dashboard_page_feature_list_point2') }}</li>
        <li>✓ {{ _('dashboard_page_feature_list_point3') }}</li>
        <li>✓ {{ _('dashboard_page_feature_list_point4') }}</li>
        <li>✓ {{ _('dashboard_page_feature_list_point5') }}</li>
      </ul>
      <div class="feature-item" v-ripple @click="goToPage('query')">
        <img src="/animated/question.gif" alt="" />
        <div>
          <h2>{{ _('dashboard_page_ask_any_question_title') }}</h2>
          <p>{{ _('dashboard_page_ask_any_question_description') }}</p>
        </div>
      </div>

      <div class="feature-item" v-ripple @click="goToPage('product-packaging-scanner')">
        <img src="/animated/scanning.gif" alt="" />
        <div>
          <h2>{{ _('dashboard_page_ask_scan_product_title') }}</h2>
          <p>{{ _('dashboard_page_ask_scan_product_description') }}</p>
        </div>
      </div>
      <div v-if="isWeb" v-ripple class="download-app">{{ _('dashboard_page_download_app') }}</div>
    </div>
  </app-wrapper>
</template>

<script>
  import appWrapper from '../../../components/appWrapper.vue';

  export default {
    components: {appWrapper},
    data() {
      return {
        heading: 'da',
        isWeb: !Meteor.isCordova,
        typeWriterTimeout: null,
      };
    },
    mounted() {
      const actualHeading = this._('dashboard_page_title');
      let i = 0;
      const speed = 50; /* The speed/duration of the effect in milliseconds */

      const typeWriter = () => {
        if (i < actualHeading.length) {
          const headingEle = document.getElementById('heading');
          if (headingEle) {
            headingEle.innerHTML += actualHeading.charAt(i);
            i += 1;
            this.typeWriterTimeout = setTimeout(typeWriter, speed);
          }
        }
      };
      typeWriter();
    },
    methods: {
      goToPage(path) {
        clearTimeout(this.typeWriterTimeout);
        setTimeout(() => {
          this.$router.push({path});
        }, 300);
      },
    },
  };
</script>

<style lang="scss" scoped>
  .dashboard-container {
    margin-top: 80px;
    .download-app {
      background-color: #7930f0;
      color: #fff;
      cursor: pointer;
      padding: 12px;
      border-radius: 10px;
      text-transform: uppercase;
      text-align: center;
      font-weight: 500;
      font-size: 14px;
      margin-bottom: 14px;
      &:hover {
        background-color: #4b06ba;
      }
    }
    @media only screen and (max-width: 600px) {
      h1 {
        min-height: 78px;
      }
    }
  }
  .meta-list {
    list-style: none;
    margin-left: 0;
    padding-left: 1em;
    text-indent: -1em;
    margin: 20px 0;
  }

  .feature-item {
    background: #fff;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 20px;
    cursor: pointer;
    box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
    &:hover {
      box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;
    }

    img {
      width: 50px;
      margin-right: 16px;
    }
  }
</style>
