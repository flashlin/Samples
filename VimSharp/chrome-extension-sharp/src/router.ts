import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: () => import('./views/main.vue'),
  },
  {
    path: '/excelQuery',
    name: 'excel',
    component: () => import('./views/excelFileQuery.vue'),
  },
  {
    path: '/convert',
    name: 'convert',
    component: () => import('./views/ConvertTool.vue'),
  },
  {
    path: '/csv',
    name: 'csv',
    component: () => import('./views/CsvTool.vue'),
  },
  {
    path: '/tableDesigner',
    name: 'tableDesigner',
    component: () => import('./views/TableDesigner.vue'),
  },
  {
    path: '/artemis',
    name: 'artemis',
    component: () => import('./views/ArtemisTool.vue'),
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
    path: '/mgmtScorePrediction',
    name: 'mgmtScorePrediction',
    component: () => import('./views/mgmtScorePrediction.vue'),
  },
  {
    path: '/test',
    name: 'test',
    component: () => import('./views/TestView.vue'),
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router 