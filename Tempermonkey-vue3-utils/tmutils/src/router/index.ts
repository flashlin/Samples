import { createRouter, createWebHistory } from 'vue-router';
import CsvToClassView from '@/components/CsvToClass.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: CsvToClassView
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import('@/components/CsvToClass.vue')
    }
  ]
})

export default router
