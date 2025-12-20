import type { App } from 'vue'
import AutoComplete from './components/AutoComplete.vue'
import DropDownList from './components/DropDownList.vue'
import { highlightText, normalizeText, splitIntoWords } from './components/autoCompleteUtils'

// 個別匯出元件
export { AutoComplete, DropDownList }
// 匯出工具函式 (選用)
export { highlightText, normalizeText, splitIntoWords }

// 定義 TypeScript 型別 (選用)
export type { AutoCompleteOption } from './components/AutoComplete.vue'
export type { DropDownOption } from './components/DropDownList.vue'

// 預設匯出外掛程式供整體安裝
export default {
    install: (app: App) => {
        app.component('AutoComplete', AutoComplete)
        app.component('DropDownList', DropDownList)
    }
}
