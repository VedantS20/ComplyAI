import VueRouter from 'vue-router';

const router = new VueRouter({
  base: '/',
  routes: [
    {
      path: '/',
      component: () => import('./pages/appUI/dashboard/index.vue'),
    },
    {
      path: '/product-packaging-scanner',
      component: () => import('./pages/appUI/scanner/index.vue'),
    },
    {
      path: '/query',
      component: () => import('./pages/appUI/query/index.vue'),
    },
    {
      path: '/home',
      component: () => import('./pages/home/index.vue'),
    },
    {
      path: '/imageScanner',
      component: () => import('./pages/imageScanner/index.vue'),
    },
    {
      path: '/test',
      component: () => import('./pages/admin/index.vue'),
    },
  ],
  mode: 'history',
});

export default router;
