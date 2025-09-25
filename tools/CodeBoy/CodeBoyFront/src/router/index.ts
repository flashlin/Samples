import { createRouter, createWebHistory } from 'vue-router'
import CodeGenerator from '@/views/CodeGenerator.vue'
import BuildSwaggerCSharpSdk from '@/views/BuildSwaggerCSharpSdk.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: CodeGenerator,
      meta: {
        title: 'Code Generator'
      }
    },
    {
      path: '/buildSwaggerCSharpSdk',
      name: 'buildSwaggerCSharpSdk',
      component: BuildSwaggerCSharpSdk,
      meta: {
        title: 'Build Swagger C# SDK'
      }
    }
  ]
})

// Update document title based on route meta
router.afterEach((to) => {
  if (to.meta?.title) {
    document.title = `${to.meta.title} - CodeBoy`
  }
})

export default router
