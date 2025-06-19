// https://github.com/Akryum/vue-virtual-scroller
declare module 'vue-virtual-scroller' {
  import { Component } from 'vue'
  
  export const RecycleScroller: Component
  export const DynamicScroller: Component
  export const DynamicScrollerItem: Component
  
  export interface RecycleScrollerProps {
    items: any[]
    direction?: 'vertical' | 'horizontal'
    itemSize?: number
    minItemSize?: number
    sizeField?: string
    typeField?: string
    keyField?: string
    pageMode?: boolean
    prerender?: number
    buffer?: number
    emitUpdate?: boolean
  }
}