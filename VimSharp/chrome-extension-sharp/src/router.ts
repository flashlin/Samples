import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: () => import('./components/HelloWorld.vue'),
  },
  {
    path: '/excel',
    name: 'Excel',
    component: () => import('./components/Excel.vue'),
  },
  {
    path: '/vim',
    name: 'VimDemo',
    component: () => import('./views/demo.vue'),
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router 