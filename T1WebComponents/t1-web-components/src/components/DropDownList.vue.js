/// <reference types="../../node_modules/.vue-global-types/vue_3.5_0.d.ts" />
import { ref, computed, watch } from 'vue';
import { highlightText, normalizeText, splitIntoWords } from './autoCompleteUtils';
const props = withDefaults(defineProps(), {
    modelValue: '',
    options: () => [],
    inputClass: '',
    placeholder: '請選擇...'
});
const emit = defineEmits();
const showDropdown = ref(false);
const selectedIndex = ref(-1);
const inputText = ref('');
const originalLabel = ref('');
let blurTimeout = null;
const findOptionByValue = (value) => {
    return props.options.find(opt => opt.value === value);
};
const highlight = (text) => {
    return highlightText(inputText.value.trim(), text);
};
watch(() => props.modelValue, (newValue) => {
    const option = findOptionByValue(newValue);
    if (option) {
        inputText.value = option.label;
        originalLabel.value = option.label;
    }
    else {
        inputText.value = '';
        originalLabel.value = '';
    }
}, { immediate: true });
const filteredOptions = computed(() => {
    const text = inputText.value.trim();
    if (!text || text === originalLabel.value) {
        return props.options;
    }
    const searchNormalized = normalizeText(text);
    const searchWords = splitIntoWords(text);
    return props.options.filter(option => {
        const textNormalized = normalizeText(option.label);
        if (textNormalized.includes(searchNormalized) ||
            option.label.toLowerCase().includes(text.toLowerCase())) {
            return true;
        }
        if (searchWords.length > 1) {
            return searchWords.every(word => textNormalized.includes(normalizeText(word)));
        }
        return false;
    });
});
const handleInput = (event) => {
    const target = event.target;
    inputText.value = target.value;
    selectedIndex.value = -1;
    showDropdown.value = true;
};
const handleFocus = () => {
    if (blurTimeout) {
        clearTimeout(blurTimeout);
        blurTimeout = null;
    }
    showDropdown.value = true;
};
const handleBlur = () => {
    blurTimeout = window.setTimeout(() => {
        showDropdown.value = false;
        selectedIndex.value = -1;
        // 如果沒有選中任何東西，且輸入內容不符合任何選項，則還原
        const currentOption = findOptionByValue(props.modelValue);
        if (currentOption) {
            inputText.value = currentOption.label;
        }
        else {
            inputText.value = '';
        }
    }, 200);
};
const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
        handleEnterKey();
    }
    else if (event.key === 'ArrowDown') {
        handleArrowDown();
    }
    else if (event.key === 'ArrowUp') {
        handleArrowUp();
    }
    else if (event.key === 'Escape') {
        handleEscapeKey();
    }
};
const handleEnterKey = () => {
    if (showDropdown.value && filteredOptions.value.length > 0) {
        if (selectedIndex.value >= 0 && selectedIndex.value < filteredOptions.value.length) {
            selectOption(filteredOptions.value[selectedIndex.value]);
        }
        else if (filteredOptions.value.length === 1) {
            selectOption(filteredOptions.value[0]);
        }
    }
};
const handleArrowDown = () => {
    if (!showDropdown.value) {
        showDropdown.value = true;
        return;
    }
    selectedIndex.value = (selectedIndex.value + 1) % filteredOptions.value.length;
};
const handleArrowUp = () => {
    if (!showDropdown.value)
        return;
    selectedIndex.value = selectedIndex.value <= 0
        ? filteredOptions.value.length - 1
        : selectedIndex.value - 1;
};
const handleEscapeKey = () => {
    showDropdown.value = false;
    selectedIndex.value = -1;
    const currentOption = findOptionByValue(props.modelValue);
    if (currentOption) {
        inputText.value = currentOption.label;
    }
};
const selectOption = (option) => {
    inputText.value = option.label;
    originalLabel.value = option.label;
    emit('update:modelValue', option.value);
    showDropdown.value = false;
    selectedIndex.value = -1;
};
const __VLS_defaults = {
    modelValue: '',
    options: () => [],
    inputClass: '',
    placeholder: '請選擇...'
};
const __VLS_ctx = {
    ...{},
    ...{},
    ...{},
    ...{},
    ...{},
};
let ___VLS_components;
let ___VLS_directives;
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: (['relative w-full', __VLS_ctx.$attrs.class]) },
});
/** @type {__VLS_StyleScopedClasses['relative']} */ ;
/** @type {__VLS_StyleScopedClasses['w-full']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "relative" },
});
/** @type {__VLS_StyleScopedClasses['relative']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.input)({
    ...{ onInput: (__VLS_ctx.handleInput) },
    ...{ onKeydown: (__VLS_ctx.handleKeyDown) },
    ...{ onFocus: (__VLS_ctx.handleFocus) },
    ...{ onBlur: (__VLS_ctx.handleBlur) },
    value: (__VLS_ctx.inputText),
    type: "text",
    placeholder: (__VLS_ctx.placeholder),
    ...{ class: ([
            'w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 pr-10',
            __VLS_ctx.inputClass
        ]) },
});
/** @type {__VLS_StyleScopedClasses['w-full']} */ ;
/** @type {__VLS_StyleScopedClasses['px-4']} */ ;
/** @type {__VLS_StyleScopedClasses['py-2']} */ ;
/** @type {__VLS_StyleScopedClasses['bg-gray-800']} */ ;
/** @type {__VLS_StyleScopedClasses['border']} */ ;
/** @type {__VLS_StyleScopedClasses['border-gray-700']} */ ;
/** @type {__VLS_StyleScopedClasses['rounded-md']} */ ;
/** @type {__VLS_StyleScopedClasses['text-white']} */ ;
/** @type {__VLS_StyleScopedClasses['placeholder-gray-500']} */ ;
/** @type {__VLS_StyleScopedClasses['focus:outline-none']} */ ;
/** @type {__VLS_StyleScopedClasses['focus:ring-2']} */ ;
/** @type {__VLS_StyleScopedClasses['focus:ring-blue-500']} */ ;
/** @type {__VLS_StyleScopedClasses['transition-all']} */ ;
/** @type {__VLS_StyleScopedClasses['duration-200']} */ ;
/** @type {__VLS_StyleScopedClasses['pr-10']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none" },
});
/** @type {__VLS_StyleScopedClasses['absolute']} */ ;
/** @type {__VLS_StyleScopedClasses['inset-y-0']} */ ;
/** @type {__VLS_StyleScopedClasses['right-0']} */ ;
/** @type {__VLS_StyleScopedClasses['flex']} */ ;
/** @type {__VLS_StyleScopedClasses['items-center']} */ ;
/** @type {__VLS_StyleScopedClasses['pr-3']} */ ;
/** @type {__VLS_StyleScopedClasses['pointer-events-none']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.svg, __VLS_intrinsics.svg)({
    ...{ class: "w-5 h-5 text-gray-500" },
    fill: "none",
    stroke: "currentColor",
    viewBox: "0 0 24 24",
});
/** @type {__VLS_StyleScopedClasses['w-5']} */ ;
/** @type {__VLS_StyleScopedClasses['h-5']} */ ;
/** @type {__VLS_StyleScopedClasses['text-gray-500']} */ ;
__VLS_asFunctionalElement(__VLS_intrinsics.path)({
    'stroke-linecap': "round",
    'stroke-linejoin': "round",
    'stroke-width': "2",
    d: "M19 9l-7 7-7-7",
});
if (__VLS_ctx.showDropdown && __VLS_ctx.filteredOptions.length > 0) {
    __VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
        ...{ class: "absolute z-20 w-full mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-xl max-h-60 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600" },
    });
    /** @type {__VLS_StyleScopedClasses['absolute']} */ ;
    /** @type {__VLS_StyleScopedClasses['z-20']} */ ;
    /** @type {__VLS_StyleScopedClasses['w-full']} */ ;
    /** @type {__VLS_StyleScopedClasses['mt-1']} */ ;
    /** @type {__VLS_StyleScopedClasses['bg-gray-800']} */ ;
    /** @type {__VLS_StyleScopedClasses['border']} */ ;
    /** @type {__VLS_StyleScopedClasses['border-gray-700']} */ ;
    /** @type {__VLS_StyleScopedClasses['rounded-md']} */ ;
    /** @type {__VLS_StyleScopedClasses['shadow-xl']} */ ;
    /** @type {__VLS_StyleScopedClasses['max-h-60']} */ ;
    /** @type {__VLS_StyleScopedClasses['overflow-y-auto']} */ ;
    /** @type {__VLS_StyleScopedClasses['scrollbar-thin']} */ ;
    /** @type {__VLS_StyleScopedClasses['scrollbar-thumb-gray-600']} */ ;
    for (const [option, index] of __VLS_getVForSourceType((__VLS_ctx.filteredOptions))) {
        __VLS_asFunctionalElement(__VLS_intrinsics.div, __VLS_intrinsics.div)({
            ...{ onMousedown: (...[$event]) => {
                    if (!(__VLS_ctx.showDropdown && __VLS_ctx.filteredOptions.length > 0))
                        return;
                    __VLS_ctx.selectOption(option);
                    // @ts-ignore
                    [$attrs, handleInput, handleKeyDown, handleFocus, handleBlur, inputText, placeholder, inputClass, showDropdown, filteredOptions, filteredOptions, selectOption,];
                } },
            key: (index),
            ...{ class: ([
                    'px-4 py-2 cursor-pointer text-sm text-gray-200 hover:bg-gray-700 transition-colors duration-150 border-b border-gray-700 last:border-0',
                    index === __VLS_ctx.selectedIndex ? 'bg-gray-700 text-white' : '',
                    props.modelValue === option.value ? 'text-blue-400 font-semibold' : ''
                ]) },
        });
        __VLS_asFunctionalDirective(___VLS_directives.vHtml)(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.highlight(option.label)) }, null, null);
        /** @type {__VLS_StyleScopedClasses['px-4']} */ ;
        /** @type {__VLS_StyleScopedClasses['py-2']} */ ;
        /** @type {__VLS_StyleScopedClasses['cursor-pointer']} */ ;
        /** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
        /** @type {__VLS_StyleScopedClasses['text-gray-200']} */ ;
        /** @type {__VLS_StyleScopedClasses['hover:bg-gray-700']} */ ;
        /** @type {__VLS_StyleScopedClasses['transition-colors']} */ ;
        /** @type {__VLS_StyleScopedClasses['duration-150']} */ ;
        /** @type {__VLS_StyleScopedClasses['border-b']} */ ;
        /** @type {__VLS_StyleScopedClasses['border-gray-700']} */ ;
        /** @type {__VLS_StyleScopedClasses['last:border-0']} */ ;
        // @ts-ignore
        [selectedIndex, highlight,];
    }
}
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    __typeEmits: {},
    __typeProps: {},
    props: {},
});
export default {};
//# sourceMappingURL=DropDownList.vue.js.map