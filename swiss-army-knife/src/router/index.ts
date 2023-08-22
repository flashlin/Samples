import { createRouter, createWebHistory } from 'vue-router';
import HomeView from '../views/HomeView.vue';
import MergeTableView from '../views/MergeTable.vue';

const prefixRoutePath = '';
const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: `${prefixRoutePath}/`,
      name: 'home',
      component: HomeView,
    },
    {
      path: `${prefixRoutePath}/mergeTable`,
      name: 'mergeTable',
      component: MergeTableView,
    },
    {
      path: `${prefixRoutePath}/about`,
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import('../views/AboutView.vue'),
    },
  ],
});

export default router;
