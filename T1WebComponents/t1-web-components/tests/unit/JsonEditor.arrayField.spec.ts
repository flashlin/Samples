import { describe, it, expect } from 'vitest'
import { nextTick } from 'vue'
import { mount } from '@vue/test-utils'
import JsonEditor from '@/components/JsonEditor.vue'
import type { JsonSchemaField } from '@/components/JsonEditor.vue'
import {
  mountJsonEditor,
  clickButton,
  clickRowActionButton,
  clickModalSaveButton,
  clickMainSaveButton,
  parseEmittedJson
} from '../helpers/jsonEditorTestUtils'
import {
  ARRAY_FIELD_SCHEMA,
  SAMPLE_DOMAIN_DATA,
  createArrayJson
} from '../helpers/mockData'

describe('JsonEditor - Array Field Type', () => {
  describe('extractSchema', () => {
    it('should detect array type for array-valued fields', () => {
      const wrapper = mount(JsonEditor, {
        props: { modelValue: null, schema: [] }
      })

      const jsonStr = createArrayJson(SAMPLE_DOMAIN_DATA)
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toContainEqual({ key: 'Country', type: 'string' })
      expect(result).toContainEqual({ key: 'SbobetDomains', type: 'array' })
      expect(result).toContainEqual({ key: 'SbotopDomains', type: 'array' })
    })

    it('should detect array type in object mode', () => {
      const wrapper = mount(JsonEditor, {
        props: { modelValue: null, schema: [] }
      })

      const jsonStr = JSON.stringify({
        name: 'Test',
        tags: ['a', 'b', 'c']
      })
      const result = wrapper.vm.extractSchema(jsonStr) as JsonSchemaField[]

      expect(result).toContainEqual({ key: 'name', type: 'string' })
      expect(result).toContainEqual({ key: 'tags', type: 'array' })
    })
  })

  describe('Table Rendering', () => {
    it('should render array values as tag elements', () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_DOMAIN_DATA),
        schema: ARRAY_FIELD_SCHEMA
      })

      const tags = wrapper.findAll('span.bg-gray-700')
      expect(tags.length).toBeGreaterThanOrEqual(4)

      const tagTexts = tags.map(t => t.text())
      expect(tagTexts).toContain('beer789.com')
      expect(tagTexts).toContain('elangjawa.com')
    })

    it('should render empty array as no tags', () => {
      const data = [{ Country: 'XX', SbobetDomains: [], SbotopDomains: [] }]
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(data),
        schema: ARRAY_FIELD_SCHEMA
      })

      const row = wrapper.find('tbody tr')
      expect(row.exists()).toBe(true)
      expect(row.findAll('span.bg-gray-700').length).toBe(0)
    })
  })

  describe('Modal Edit', () => {
    it('should display array as newline-separated text in textarea', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_DOMAIN_DATA),
        schema: ARRAY_FIELD_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'edit')
      await nextTick()

      const textarea = wrapper.find('textarea#SbobetDomains')
      expect(textarea.exists()).toBe(true)
      expect((textarea.element as HTMLTextAreaElement).value).toBe('beer789.com\nelangjawa.com')
    })

    it('should preserve array structure after edit and save', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_DOMAIN_DATA),
        schema: ARRAY_FIELD_SCHEMA
      })

      await clickRowActionButton(wrapper, 0, 'edit')
      await nextTick()

      const textarea = wrapper.find('textarea#SbobetDomains')
      await textarea.setValue('new-domain.com\nupdated.com')
      await textarea.trigger('input')

      await clickModalSaveButton(wrapper)
      await nextTick()

      await clickMainSaveButton(wrapper)
      await nextTick()

      const result = parseEmittedJson<any[]>(wrapper)
      expect(Array.isArray(result[0].SbobetDomains)).toBe(true)
      expect(result[0].SbobetDomains).toEqual(['new-domain.com', 'updated.com'])
    })
  })

  describe('Add New Item', () => {
    it('should initialize array fields as empty array', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_DOMAIN_DATA),
        schema: ARRAY_FIELD_SCHEMA
      })

      await clickButton(wrapper, 'Add Item')
      await nextTick()

      const textarea = wrapper.find('textarea#SbobetDomains')
      expect(textarea.exists()).toBe(true)
      expect((textarea.element as HTMLTextAreaElement).value).toBe('')

      await clickModalSaveButton(wrapper)
      await nextTick()

      await clickMainSaveButton(wrapper)
      await nextTick()

      const result = parseEmittedJson<any[]>(wrapper)
      const lastItem = result[result.length - 1]
      expect(Array.isArray(lastItem.SbobetDomains)).toBe(true)
      expect(lastItem.SbobetDomains).toEqual([])
    })
  })

  describe('Search', () => {
    it('should find rows by searching array element values', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson(SAMPLE_DOMAIN_DATA),
        schema: ARRAY_FIELD_SCHEMA
      })

      const searchInput = wrapper.find('input[placeholder="Search items..."]')
      await searchInput.setValue('wiskeybear')
      await nextTick()

      const rows = wrapper.findAll('tbody tr')
      expect(rows.length).toBe(1)
      expect(rows[0].text()).toContain('ID')
    })
  })
})
