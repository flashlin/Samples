/// <reference types="../../node_modules/.vue-global-types/vue_3.5_0.d.ts" />
import { ref } from 'vue';
import { useAppStore } from '../store';
const store = useAppStore();
const selectedTech = ref('');
const techOptions = [
    { label: 'Vue.js', value: 'vue' },
    { label: 'TypeScript', value: 'ts' },
    { label: 'TailwindCSS', value: 'tailwind' },
    { label: 'ViteBuildTool', value: 'vite' },
    { label: 'PnpmManager', value: 'pnpm' }
];
const searchQuery = ref('');
const searchOptions = [
    'ViteBuildTool',
    'VueComponents',
    'TailwindLayouts',
    'TypeScriptIntegration',
    'PiniaState'
];
const __VLS_ctx = {
    ...{},
    ...{},
};
let ___VLS_components;
let ___VLS_directives;
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "min-h-screen p-12 bg-gray-900 text-white" },
});
/** @type {__VLS_StyleScopedClasses['min-h-screen']} */ ;
/** @type {__VLS_StyleScopedClasses['p-12']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-gray-900']} */ ;
/** @type {__VLS_StyleScopedClasses['text-white']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "max-w-4xl mx-auto space-y-12" },
});
/** @type {__VLS_StyleScopedClasses['max-w-4xl']} */ ;
/** @type {__VLS_StyleScopedClasses['mx-auto']} */ ;
/** @type {__VLS_StyleScopedClasses['space-y-12']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.h1, __VLS_intrinsics.h1)({
    ...{ class: "text-4xl font-bold text-blue-400 border-b border-gray-800 pb-4" },
});
/** @type {__VLS_StyleScopedClasses['text-4xl']} */ ;
/** @type {__VLS_StyleScopedClasses['font-bold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-blue-400']} */ ;
/** @type {__VLS_StyleScopedClasses['border-b']} */ ;
/** @type {__VLS_StyleScopedClasses['border-gray-800']} */ ;
/** @type {__VLS_StyleScopedClasses['pb-4']} */ ;
(__VLS_ctx.store.title);
__VLS_asFunctionalElement(__VLS_intrinsics.section, __VLS_intrinsics.section)({
    ...{ class: "grid grid-cols-1 md:grid-cols-2 gap-8" },
});
/** @type {__VLS_StyleScopedClasses['grid']} */ ;
/** @type {__VLS_StyleScopedClasses['grid-cols-1']} */ ;
/** @type {__VLS_StyleScopedClasses['md:grid-cols-2']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-8']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700" },
});
/** @type {__VLS_StyleScopedClasses['p-6']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-gray-800']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['shadow-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-gray-700']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.h2, __VLS_intrinsics.h2)({
    ...{ class: "text-xl font-semibold text-blue-300 mb-4" },
});
/** @type {__VLS_StyleScopedClasses['text-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-blue-300']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-4']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.p, __VLS_intrinsics.p)({
    ...{ class: "text-sm text-gray-400 mb-6" },
});
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-gray-400']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-6']} */ ;
let __VLS_0;
/** @ts-ignore @type {typeof ___VLS_components.DropDownList} */
DropDownList;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent(__VLS_0, new __VLS_0({
    modelValue: (__VLS_ctx.selectedTech),
    options: (__VLS_ctx.techOptions),
    placeholder: "Select a technology...",
}));
const __VLS_2 = __VLS_1({
    modelValue: (__VLS_ctx.selectedTech),
    options: (__VLS_ctx.techOptions),
    placeholder: "Select a technology...",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "mt-4 p-2 bg-gray-900 rounded border border-gray-700 text-xs font-mono" },
});
/** @type {__VLS_StyleScopedClasses['mt-4']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-gray-900']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-gray-700']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "text-blue-400" },
});
/** @type {__VLS_StyleScopedClasses['text-blue-400']} */ ;
(__VLS_ctx.selectedTech || 'unset');
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700" },
});
/** @type {__VLS_StyleScopedClasses['p-6']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-gray-800']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['shadow-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-gray-700']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.h2, __VLS_intrinsics.h2)({
    ...{ class: "text-xl font-semibold text-emerald-300 mb-4" },
});
/** @type {__VLS_StyleScopedClasses['text-xl']} */ ;
/** @type {__VLS_StyleScopedClasses['font-semibold']} */ ;
/** @type {__VLS_StyleScopedClasses['text-emerald-300']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-4']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.p, __VLS_intrinsics.p)({
    ...{ class: "text-sm text-gray-400 mb-6" },
});
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['text-gray-400']} */ ;
/** @type {__VLS_StyleScopedClasses['mb-6']} */ ;
let __VLS_5;
/** @ts-ignore @type {typeof ___VLS_components.AutoComplete} */
AutoComplete;
// @ts-ignore
const __VLS_6 = __VLS_asFunctionalComponent(__VLS_5, new __VLS_5({
    modelValue: (__VLS_ctx.searchQuery),
    options: (__VLS_ctx.searchOptions),
    placeholder: "Search or type freely...",
}));
const __VLS_7 = __VLS_6({
    modelValue: (__VLS_ctx.searchQuery),
    options: (__VLS_ctx.searchOptions),
    placeholder: "Search or type freely...",
}, ...__VLS_functionalComponentArgsRest(__VLS_6));
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "mt-4 p-2 bg-gray-900 rounded border border-gray-700 text-xs font-mono" },
});
/** @type {__VLS_StyleScopedClasses['mt-4']} */ ;
/** @type {__VLS_StyleScopedClasses['p-2']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-gray-900']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-gray-700']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "text-emerald-400" },
});
/** @type {__VLS_StyleScopedClasses['text-emerald-400']} */ ;
(__VLS_ctx.searchQuery || '""');
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "p-6 bg-gray-800/50 rounded-lg italic text-gray-400 text-center" },
});
/** @type {__VLS_StyleScopedClasses['p-6']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-gray-800/50']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['italic']} */ ;
/** @type {__VLS_StyleScopedClasses['text-gray-400']} */ ;
/** @type {__VLS_StyleScopedClasses['text-center']} */ ;
// @ts-ignore
[store, selectedTech, selectedTech, techOptions, searchQuery, searchQuery, searchOptions,];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
//# sourceMappingURL=HomeView.vue.js.map