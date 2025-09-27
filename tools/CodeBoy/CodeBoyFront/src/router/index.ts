import { createRouter, createWebHistory } from 'vue-router'
import CodeGenerator from '@/views/CodeGenerator.vue'
import BuildSwaggerCSharpSdk from '@/views/BuildSwaggerCSharpSdk.vue'
import BuildDatabaseModelNupkg from '@/views/BuildDatabaseModelNupkg.vue'
import UseGrpcProtoGeneratorDoc from '@/views/UseGrpcProtoGeneratorDoc.vue'
import GenTypescriptCodeFromSwagger from '@/views/GenTypescriptCodeFromSwagger.vue'

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
    },
    {
      path: '/buildDatabaseModelNupkg',
      name: 'buildDatabaseModelNupkg',
      component: BuildDatabaseModelNupkg,
      meta: {
        title: 'Build Database Model NuGet Package'
      }
    },
    {
      path: '/useGrpcProtoGeneratorDoc',
      name: 'useGrpcProtoGeneratorDoc',
      component: UseGrpcProtoGeneratorDoc,
      meta: {
        title: 'gRPC Proto Generator Documentation'
      }
    },
    {
      path: '/genTypescriptCodeFromSwagger',
      name: 'genTypescriptCodeFromSwagger',
      component: GenTypescriptCodeFromSwagger,
      meta: {
        title: 'Generate TypeScript Code from Swagger'
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
