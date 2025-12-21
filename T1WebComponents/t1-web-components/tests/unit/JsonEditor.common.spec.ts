import { describe, it, expect } from 'vitest'
import { nextTick } from 'vue'
import { mountJsonEditor } from '../helpers/jsonEditorTestUtils'
import {
  BASIC_SCHEMA,
  createArrayJson,
  createObjectJson
} from '../helpers/mockData'

describe('JsonEditor - Common Functionality', () => {
  describe('Props Validation', () => {
    it('should accept null as modelValue', () => {
      const wrapper = mountJsonEditor({
        modelValue: null,
        schema: BASIC_SCHEMA
      })

      expect(wrapper.exists()).toBe(true)
    })

    it('should accept empty string as modelValue', () => {
      const wrapper = mountJsonEditor({
        modelValue: '',
        schema: BASIC_SCHEMA
      })

      expect(wrapper.exists()).toBe(true)
      expect(wrapper.find('table').exists()).toBe(false)
    })

    it('should render with minimal schema', () => {
      const minimalSchema = [
        { key: 'value', label: 'Value', type: 'string' as const }
      ]

      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ value: 'test' }),
        schema: minimalSchema
      })

      expect(wrapper.exists()).toBe(true)
    })

    it('should handle schema without labels', () => {
      const schemaNoLabels = [
        { key: 'id', type: 'number' as const },
        { key: 'name', type: 'string' as const }
      ] as any

      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: schemaNoLabels
      })

      expect(wrapper.exists()).toBe(true)
    })
  })

  describe('Event Emissions', () => {
    it('should emit update:modelValue with correct payload', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      await wrapper.find('input#form-name').setValue('Updated')
      await wrapper.find('input#form-name').trigger('input')
      await wrapper.findAll('button').find(btn =>
        btn.text().includes('Save')
      )?.trigger('click')
      await nextTick()

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeTruthy()
      expect(emitted![0][0]).toBeTypeOf('string')
    })

    it('should emit change event on data modification', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      await wrapper.find('input#form-name').setValue('Updated')
      await wrapper.find('input#form-name').trigger('input')
      await wrapper.findAll('button').find(btn =>
        btn.text().includes('Save')
      )?.trigger('click')
      await nextTick()

      expect(wrapper.emitted('change')).toBeTruthy()
    })

    it('should emit error event on invalid JSON', async () => {
      const wrapper = mountJsonEditor({
        modelValue: '{invalid json}',
        schema: BASIC_SCHEMA
      })

      await nextTick()

      expect(wrapper.emitted('error')).toBeTruthy()
    })
  })

  describe('Mode Detection', () => {
    it('should detect array mode from JSON string', () => {
      const wrapper = mountJsonEditor({
        modelValue: createArrayJson([{ id: 1, name: 'Test' }]),
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('table').exists()).toBe(true)
      expect(wrapper.find('form').exists()).toBe(false)
    })

    it('should detect object mode from JSON string', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('table').exists()).toBe(false)
      expect(wrapper.text()).toContain('Save')
      expect(wrapper.text()).toContain('Cancel')
    })

    it('should default to object mode for empty string', () => {
      const wrapper = mountJsonEditor({
        modelValue: '',
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('table').exists()).toBe(false)
      expect(wrapper.text()).toContain('Save')
    })

    it('should default to object mode for null', () => {
      const wrapper = mountJsonEditor({
        modelValue: null,
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('table').exists()).toBe(false)
      expect(wrapper.text()).toContain('Save')
    })
  })

  describe('Reactivity', () => {
    it('should update when modelValue prop changes', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'First' }),
        schema: BASIC_SCHEMA
      })

      await wrapper.setProps({
        modelValue: createObjectJson({ id: 2, name: 'Second' })
      })
      await nextTick()

      const nameInput = wrapper.find('input#form-name') as any
      expect(nameInput.element.value).toBe('Second')
    })

    it('should switch modes when JSON type changes', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('table').exists()).toBe(false)
      expect(wrapper.text()).toContain('Save')

      await wrapper.setProps({
        modelValue: createArrayJson([{ id: 1, name: 'Test' }])
      })
      await nextTick()

      expect(wrapper.find('table').exists()).toBe(true)
      // Array Mode now also has Save/Cancel buttons
    })

    it('should update when compact prop changes', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA,
        compact: false
      })

      await wrapper.setProps({ compact: true })
      await wrapper.find('input#form-name').setValue('Updated')
      await wrapper.find('input#form-name').trigger('input')
      await wrapper.findAll('button').find(btn =>
        btn.text().includes('Save')
      )?.trigger('click')
      await nextTick()

      const emitted = wrapper.emitted('update:modelValue')
      const lastEmitted = emitted![emitted!.length - 1][0] as string
      expect(lastEmitted).not.toContain('\n')
    })
  })

  describe('Schema Handling', () => {
    it('should handle missing fields in data gracefully', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1 }),
        schema: BASIC_SCHEMA
      })

      const nameInput = wrapper.find('input#form-name') as any
      expect(nameInput.element.value).toBe('')
    })

    it('should render all schema fields', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      expect(wrapper.find('input#form-id').exists()).toBe(true)
      expect(wrapper.find('input#form-name').exists()).toBe(true)
    })
  })
})
