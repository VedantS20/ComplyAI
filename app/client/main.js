import Vue from 'vue';
import {DateTime as luxon} from 'luxon';
import VueRouter from 'vue-router';
import '../imports/client/plugins/index';
import '../imports/client/mixins/index';
import App from '../imports/client/App.vue';
import router from '../imports/client/router';
import vuetify from '../imports/client/plugins/vuetify';
import localizationMixin from '../imports/client/mixins/localization.mixin';

Vue.mixin(localizationMixin);

DateTime = luxon;
Meteor.startup(() => {
  // eslint-disable-next-line no-new
  new Vue({
    el: '#app',
    ...App,
    router,
    vuetify,
  });
  if (Meteor.isCordova) {
    window.open = cordova.InAppBrowser.open;
    universalLinks.subscribe('share', function (eventData) {
      const shareParameters = eventData.path.split('/post')[1];
      if (shareParameters) {
        const linkToOpen = `/share${shareParameters}`;
        VueRouter.push({path: linkToOpen});
      }
    });
  }
});
