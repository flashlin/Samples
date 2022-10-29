import { createRouter, createWebHistory } from 'vue-router';
import DataConverterView from '@/views/DataConverter.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: DataConverterView
    },
    {
      path: '/dataChart',
      name: 'dataChart',
      component: () => import('@/views/DataChart.vue')
    }
  ]
});

export default router;
