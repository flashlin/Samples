import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: () => import('./views/main.vue'),
  },
  {
    path: '/excel',
    name: 'Excel',
    component: () => import('./components/Excel.vue'),
  },
  {
    path: '/fileupload',
    name: 'FileUpload',
    component: () => import('./views/demoFileUpload.vue'),
  },
  {
    path: '/excelQuery',
    name: 'excel',
    component: () => import('./views/excelFileQuery.vue'),
  },
  {
    path: '/mgmtScorePrediction',
    name: 'mgmtScorePrediction',
    component: () => import('./views/mgmtScorePrediction.vue'),
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router 