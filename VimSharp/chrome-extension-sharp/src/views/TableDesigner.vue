<script setup lang="ts">
import { ref } from 'vue'

const tableName = ref('')

interface Field {
  id: number
  name: string
  dataType: string
  defaultValue: string
  isIdentify: boolean
  isPkKey: boolean
  description: string
}

interface Key {
  id: number
  name: string
  fieldNames: string[]
  input: string
}

const fields = ref<Field[]>([])
const keys = ref<Key[]>([])
let fieldId = 1
let keyId = 1

function addField() {
  fields.value.push({
    id: fieldId++,
    name: '',
    dataType: '',
    defaultValue: '',
    isIdentify: false,
    isPkKey: false,
    description: ''
  })
}

function deleteField(idx: number) {
  fields.value.splice(idx, 1)
}

function moveField(idx: number, dir: number) {
  const newIdx = idx + dir
  if (newIdx < 0 || newIdx >= fields.value.length) return
  const temp = fields.value[idx]
  fields.value[idx] = fields.value[newIdx]
  fields.value[newIdx] = temp
}

function addKey() {
  keys.value.push({
    id: keyId++,
    name: '',
    fieldNames: [],
    input: ''
  })
}

function deleteKey(idx: number) {
  keys.value.splice(idx, 1)
}

function addTag(keyIdx: number) {
  const key = keys.value[keyIdx]
  const val = key.input.trim()
  if (val && !key.fieldNames.includes(val)) {
    key.fieldNames.push(val)
  }
  key.input = ''
}

function removeTag(keyIdx: number, tagIdx: number) {
  keys.value[keyIdx].fieldNames.splice(tagIdx, 1)
}

function onTagInput(keyIdx: number) {
  // 可加自動補全邏輯
}
</script>

<template>
  <div class="p-6 space-y-8">
    <!-- Table Name + Add -->
    <div class="flex items-center space-x-4">
      <label class="font-bold">TableName:</label>
      <input v-model="tableName" type="text" class="input input-bordered px-2 py-1 border rounded" />
      <button @click="addField" class="btn btn-primary px-3 py-1 bg-blue-500 text-white rounded">Add</button>
    </div>

    <!-- Fields Table -->
    <div class="overflow-x-auto">
      <div class="min-w-max border rounded">
        <div class="flex bg-gray-100 font-bold px-2 py-2">
          <div class="w-10">No</div>
          <div class="w-32">FieldName</div>
          <div class="w-32">DataType</div>
          <div class="w-32">DefaultValue</div>
          <div class="w-24">IsIdentify</div>
          <div class="w-20">IsPkKey</div>
          <div class="w-40">Description</div>
          <div class="w-32">Action</div>
        </div>
        <div v-for="(field, idx) in fields" :key="field.id" class="flex items-center border-t px-2 py-1">
          <div class="w-10">{{ idx + 1 }}</div>
          <div class="w-32"><input v-model="field.name" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-32"><input v-model="field.dataType" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-32"><input v-model="field.defaultValue" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-24 flex justify-center"><input v-model="field.isIdentify" type="checkbox" class="checkbox" /></div>
          <div class="w-20 flex justify-center"><input v-model="field.isPkKey" type="checkbox" class="checkbox" /></div>
          <div class="w-40"><input v-model="field.description" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-32 flex space-x-1">
            <button @click="deleteField(idx)" class="btn px-2 py-0.5 bg-red-500 text-white rounded">Delete</button>
            <button @click="moveField(idx, -1)" :disabled="idx === 0" class="btn px-2 py-0.5 bg-gray-300 rounded">↑</button>
            <button @click="moveField(idx, 1)" :disabled="idx === fields.length - 1" class="btn px-2 py-0.5 bg-gray-300 rounded">↓</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Key + Add -->
    <div class="flex items-center space-x-4 mt-8">
      <label class="font-bold">Key</label>
      <button @click="addKey" class="btn btn-primary px-3 py-1 bg-blue-500 text-white rounded">Add</button>
    </div>

    <!-- Keys Table -->
    <div class="overflow-x-auto">
      <div class="min-w-max border rounded">
        <div class="flex bg-gray-100 font-bold px-2 py-2">
          <div class="w-10">No</div>
          <div class="w-40">IndexName</div>
          <div class="w-64">FieldNames</div>
          <div class="w-24">Action</div>
        </div>
        <div v-for="(key, idx) in keys" :key="key.id" class="flex items-center border-t px-2 py-1">
          <div class="w-10">{{ idx + 1 }}</div>
          <div class="w-40"><input v-model="key.name" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-64 flex flex-wrap items-center gap-1">
            <div v-for="(tag, tIdx) in key.fieldNames" :key="tIdx" class="flex items-center bg-gray-200 rounded px-2 py-0.5 mr-1 mb-1">
              <span>{{ tag }}</span>
              <button @click="removeTag(idx, tIdx)" class="ml-1 text-gray-500 hover:text-red-500 focus:outline-none">✕</button>
            </div>
            <input
              v-model="key.input"
              @keydown.enter.prevent="addTag(idx)"
              @input="onTagInput(idx)"
              type="text"
              class="input px-1 py-0.5 border rounded w-24"
              placeholder="輸入欄位名..."
              list="field-list"
            />
            <datalist id="field-list">
              <option v-for="f in fields" :key="f.id" :value="f.name" />
            </datalist>
          </div>
          <div class="w-24 flex space-x-1">
            <button @click="deleteKey(idx)" class="btn px-2 py-0.5 bg-red-500 text-white rounded">Delete</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.input {
  @apply border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-200;
}
.btn {
  @apply cursor-pointer transition;
}
.checkbox {
  @apply w-4 h-4;
}
</style> 