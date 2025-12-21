import { describe, it, expect } from 'vitest'
import { nextTick } from 'vue'
import {
  mountJsonEditor,
  clickButton,
  fillInputField,
  parseEmittedJson,
  getEmittedModelValue
} from '../helpers/jsonEditorTestUtils'
import {
  BASIC_SCHEMA,
  FULL_SCHEMA,
  SAMPLE_OBJECT_DATA,
  createObjectJson
} from '../helpers/mockData'

describe('JsonEditor - Object Mode', () => {
  describe('Initial Rendering', () => {
    it('should render form when JSON is object', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson(SAMPLE_OBJECT_DATA),
        schema: FULL_SCHEMA
      })

      expect(wrapper.find('form').exists()).toBe(false)
      const inputs = wrapper.findAll('input')
      expect(inputs.length).toBeGreaterThan(0)
    })

    it('should populate form fields with object values', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 42, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      const idInput = wrapper.find('input#form-id') as any
      const nameInput = wrapper.find('input#form-name') as any

      expect(idInput.element.value).toBe('42')
      expect(nameInput.element.value).toBe('Test')
    })

    it('should render Save and Cancel buttons', () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson(SAMPLE_OBJECT_DATA),
        schema: FULL_SCHEMA
      })

      expect(wrapper.text()).toContain('Save')
      expect(wrapper.text()).toContain('Cancel')
    })

    it('should initialize form from schema when modelValue is empty string', () => {
      const wrapper = mountJsonEditor({
        modelValue: '',
        schema: BASIC_SCHEMA
      })

      const idInput = wrapper.find('input#form-id') as any
      const nameInput = wrapper.find('input#form-name') as any

      expect(idInput.element.value).toBe('0')
      expect(nameInput.element.value).toBe('')
    })
  })

  describe('Form Editing', () => {
    it('should track unsaved changes', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'Modified')
      await nextTick()

      const saveBtn = wrapper.findAll('button').find(btn =>
        btn.text().includes('Save')
      )
      expect(saveBtn?.classes()).not.toContain('opacity-50')
    })

    it('should allow editing multiple fields', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'id', '999')
      await fillInputField(wrapper, 'name', 'New Name')

      const idInput = wrapper.find('input#form-id') as any
      const nameInput = wrapper.find('input#form-name') as any

      expect(idInput.element.value).toBe('999')
      expect(nameInput.element.value).toBe('New Name')
    })
  })

  describe('Save Functionality', () => {
    it('should emit updated object when Save clicked', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'Updated')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const result = parseEmittedJson<any>(wrapper)
      expect(result).toEqual({ id: 1, name: 'Updated' })
    })

    it('should emit change event when saved', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'Updated')
      await clickButton(wrapper, 'Save')
      await nextTick()

      expect(wrapper.emitted('change')).toBeTruthy()
    })

    it('should clear unsaved changes flag after save', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'Updated')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const saveBtn = wrapper.findAll('button').find(btn =>
        btn.text().includes('Save')
      )
      expect(saveBtn?.element.hasAttribute('disabled')).toBe(true)
    })

    it('should convert field types correctly on save', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'id', '999')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const result = parseEmittedJson<any>(wrapper)
      expect(result.id).toBe(999)
      expect(typeof result.id).toBe('number')
    })
  })

  describe('Cancel Functionality', () => {
    it('should revert changes when Cancel clicked', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'Modified')
      await clickButton(wrapper, 'Cancel')
      await nextTick()

      const nameInput = wrapper.find('input#form-name') as any
      expect(nameInput.element.value).toBe('Original')
    })

    it('should not emit update when cancelled', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'Modified')
      await clickButton(wrapper, 'Cancel')
      await nextTick()

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeUndefined()
    })

    it('should clear unsaved changes flag after cancel', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Original' }),
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'name', 'Modified')
      await clickButton(wrapper, 'Cancel')
      await nextTick()

      const saveBtn = wrapper.findAll('button').find(btn =>
        btn.text().includes('Save')
      )
      expect(saveBtn?.element.hasAttribute('disabled')).toBe(true)
    })
  })

  describe('Field Type Handling', () => {
    it('should handle date field correctly', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson(SAMPLE_OBJECT_DATA),
        schema: FULL_SCHEMA
      })

      const dateInput = wrapper.find('input#form-createdAt')
      expect(dateInput.attributes('type')).toBe('date')
      expect((dateInput.element as HTMLInputElement).value).toBe('2025-01-15')
    })

    it('should update date field', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson(SAMPLE_OBJECT_DATA),
        schema: FULL_SCHEMA
      })

      await fillInputField(wrapper, 'createdAt', '2025-12-31')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const result = parseEmittedJson<any>(wrapper)
      expect(result.createdAt).toBe('2025-12-31')
    })

    it('should handle number field type conversion', async () => {
      const wrapper = mountJsonEditor({
        modelValue: '',
        schema: BASIC_SCHEMA
      })

      await fillInputField(wrapper, 'id', '123')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const result = parseEmittedJson<any>(wrapper)
      expect(result.id).toBe(123)
      expect(typeof result.id).toBe('number')
    })
  })

  describe('Compact Output', () => {
    it('should emit compact JSON when compact=true', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA,
        compact: true
      })

      await fillInputField(wrapper, 'name', 'Updated')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emittedValue = getEmittedModelValue(wrapper)
      expect(emittedValue).not.toContain('\n')
    })

    it('should emit formatted JSON when compact=false', async () => {
      const wrapper = mountJsonEditor({
        modelValue: createObjectJson({ id: 1, name: 'Test' }),
        schema: BASIC_SCHEMA,
        compact: false
      })

      await fillInputField(wrapper, 'name', 'Updated')
      await clickButton(wrapper, 'Save')
      await nextTick()

      const emittedValue = getEmittedModelValue(wrapper)
      expect(emittedValue).toContain('\n')
    })
  })
})
