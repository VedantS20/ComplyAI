import Vue from 'vue';
import {DateTime} from 'luxon';

Vue.mixin({
  methods: {
    getFormattedDate(date) {
      return DateTime.fromJSDate(date).setLocale('en-IN').toFormat('dd LLLL yyyy');
    },
    getAlertTime(date) {
      const isTodaysDate =
        DateTime.now().startOf('day').toMillis() === DateTime.fromJSDate(date).startOf('day').toMillis();
      if (isTodaysDate) {
        const {hours, minutes} = DateTime.now().diff(DateTime.fromJSDate(date), ['hours', 'minutes']).toObject();
        if (hours === 0) {
          if (minutes <= 5) {
            return 'Now';
          }
          if (minutes <= 10 && minutes > 5) {
            return '5 minutes ago';
          }
          if (minutes > 10 && minutes <= 30) {
            return '30 minutes ago';
          }
          if (minutes > 30 && minutes <= 60) {
            return '1 hour ago';
          }
        } else if (hours > 0) {
          if (hours >= 1 && hours <= 3) {
            return '3 hour ago';
          }
          if (hours > 3 && hours <= 6) {
            return '6 hour ago';
          }
          if (hours > 6 && hours <= 12) {
            return '12 hour ago';
          }
        }
      }
      return DateTime.fromJSDate(date).setLocale('en-IN').toFormat('dd LLL yyyy | hh:mm a ');
    },
  },
});
