import { defineComponent, ref, shallowRef, onMounted, h } from 'vue'

// 路由表：path => 組件
const routes: Record<string, any> = {
  '': () => import('@/views/main.vue'),
  'excel': () => import('@/components/Excel.vue'),
  'fileupload': () => import('@/views/demoFileUpload.vue'),
  'excelQuery': () => import('@/views/excelFileQuery.vue'),
}

function getCurrentPath() {
  // 取得 hash（去掉 #/ 或 #）
  return window.location.hash.replace(/^#\/?/, '')
}

export const VisualRouterView = defineComponent({
  name: 'VisualRouterView',
  setup() {
    const currentView = shallowRef(null)
    const currentPath = ref(getCurrentPath())

    async function loadView(path: string) {
      const loader = routes[path]
      if (loader) {
        const mod = await loader()
        currentView.value = mod.default
      } else {
        currentView.value = null
      }
    }

    function onHashChange() {
      currentPath.value = getCurrentPath()
      loadView(currentPath.value)
    }

    onMounted(() => {
      window.addEventListener('hashchange', onHashChange)
      loadView(currentPath.value)
    })

    return () => currentView.value ? h(currentView.value) : null
  }
})

export function goTo(path: string) {
  window.location.hash = '#' + path
} 