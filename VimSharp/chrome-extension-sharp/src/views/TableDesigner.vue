<script setup lang="ts">
import CodeEditor from '@/components/codeEditor.vue'
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
  isNull: boolean
}

interface Key {
  id: number
  name: string
  fieldNames: string[]
  input: string
}

const fields = ref<Field[]>([])
const uniqueKeys = ref<Key[]>([])
let fieldId = 1
let keyId = 1
const createTableSqlCode = ref('')
const cSharpCode = ref('')

function addField() {
  fields.value.push({
    id: fieldId++,
    name: '',
    dataType: '',
    defaultValue: '',
    isIdentify: false,
    isPkKey: false,
    description: '',
    isNull: true
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

function addKeyIndex() {
  uniqueKeys.value.push({
    id: keyId++,
    name: '',
    fieldNames: [],
    input: ''
  })
}

function deleteKeyIndex(idx: number) {
  uniqueKeys.value.splice(idx, 1)
}

function addTag(keyIdx: number) {
  const key = uniqueKeys.value[keyIdx]
  const val = key.input.trim()
  if (val && !key.fieldNames.includes(val)) {
    key.fieldNames.push(val)
  }
  key.input = ''
}

function removeTag(keyIdx: number, tagIdx: number) {
  uniqueKeys.value[keyIdx].fieldNames.splice(tagIdx, 1)
}

function onTagInput(keyIdx: number) {
  // 可加自動補全邏輯
}

function generateCreateTableSql() {
  if( fields.value.length === 0 ) {
    return '';
  }
  let createTableSql = `CREATE TABLE [${tableName.value}] (\n`
  const templateBody = fields.value.map(field => {
    const isIdentity = field.isIdentify ? 'IDENTITY(1,1)' : ''
    const defaultValue = field.defaultValue ? `DEFAULT ${field.defaultValue}` : ''
    const isNull = field.isNull ? 'NULL' : 'NOT NULL'
    return `  [${field.name}] ${field.dataType} ${isNull} ${defaultValue} ${isIdentity}`;
  }).join(",\n");
  createTableSql += templateBody
  createTableSql += `\n);\n`
  createTableSql += "\n\n\n"
  return createTableSql
}

function generateCreateDescriptionSql() {
  let createDescriptionSql = ''
  const templateBody = fields.value.map(field => {
    if( field.description === '' ) {
      return '';
    }
    return `EXEC sys.sp_addextendedproperty @name=N'${field.name}', @value=N'${field.description}', @level0type=N'SCHEMA', @level0name=N'dbo', @level1type=N'TABLE', @level1name=N'${tableName.value}'
GO`;
  }).join("\n");
  createDescriptionSql += templateBody
  createDescriptionSql += "\n\n\n"
  return createDescriptionSql
}

function generateCreatePrimaryKeySql() {
  let createPrimaryKeySql = `ALTER TABLE [${tableName.value}] ADD CONSTRAINT [PK_${tableName}] PRIMARY KEY CLUSTERED (`
  const primaryKeys = fields.value.filter(field => field.isPkKey)
  if( primaryKeys.length === 0 ) {
    return '';
  }
  const templateBody = primaryKeys.map(key => {
    return `[${key.name}]`
  }).join(",");
  createPrimaryKeySql += templateBody
  createPrimaryKeySql += `);\n`
  createPrimaryKeySql += "GO\n"
  createPrimaryKeySql += "\n\n\n"
  return createPrimaryKeySql
}

function generateSql() {
  let sql = ''
  sql += generateCreateTableSql()
  sql += generateCreatePrimaryKeySql()
  sql += generateCreateDescriptionSql()
  sql += generateUniqueKeySql()
  createTableSqlCode.value = sql
}

function generateUniqueKeySql() {
  if( uniqueKeys.value.length === 0 ) {
    return '';
  }
  let createUniqueKeySql = ''
  const templateBody = uniqueKeys.value.map(key => {
    const keyDashNames = key.fieldNames.map(name => `${name}`).join('_');
    const keyNames = key.fieldNames.map(name => `[${name}]`).join(',');
    return `CREATE UNIQUE INDEX [UIX_${tableName.value}_${keyDashNames}] ON [${tableName.value}] (${keyNames})
    WITH (PAD_INDEX = OFF, ONLINE = ON, FILLFACTOR = 95) ON [PRIMARY]
GO
`
  }).join("\n");
  createUniqueKeySql += templateBody
  createUniqueKeySql += "\n\n\n"
  return createUniqueKeySql
}

function toClassType(dataType: string) {
  dataType = dataType.toLowerCase();
  if( dataType.includes('integer') ) {
    return 'int';
  }
  if( dataType.includes('decimal') ) {
    return 'decimal';
  }
  if( dataType.includes('datetime') ) {
    return 'DateTime';
  }
  if( dataType.includes('bit') ) {
    return 'bool';
  }
  if( dataType.includes('varchar') ) {
    return 'string';
  }
  return 'string';
}

function generateEntityClass() {
  let entityClass = `public class ${tableName.value}Entity {\n`
  const templateBody = fields.value.map(field => {
    const classType = toClassType(field.dataType);
    return `  public ${classType} ${field.name} { get; set; }`
  }).join("\n");
  entityClass += templateBody
  entityClass += "\n}"
  return entityClass
}

function generateCSharpCode() {
  let csharpCode = generateEntityClass()
  csharpCode += "\n\n\n"
  cSharpCode.value = csharpCode
}

function generateCode(){
  generateSql()
  generateCSharpCode()
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
        <div class="flex font-bold px-2 py-2 bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100">
          <div class="w-10">No</div>
          <div class="w-32">FieldName</div>
          <div class="w-32">DataType</div>
          <div class="w-20">IsNull</div>
          <div class="w-32">DefaultValue</div>
          <div class="w-24">IsIdentity</div>
          <div class="w-20">IsPkKey</div>
          <div class="w-40">Description</div>
          <div class="w-32">Action</div>
        </div>
        <div v-for="(field, idx) in fields" :key="field.id" class="flex items-center border-t px-2 py-1 bg-white text-gray-900 dark:bg-gray-900 dark:text-gray-100">
          <div class="w-10">{{ idx + 1 }}</div>
          <div class="w-32"><input v-model="field.name" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-32"><input v-model="field.dataType" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-20 flex justify-center"><input v-model="field.isNull" type="checkbox" class="checkbox" /></div>
          <div class="w-32"><input v-model="field.defaultValue" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-24 flex justify-center"><input v-model="field.isIdentify" type="checkbox" class="checkbox" /></div>
          <div class="w-20 flex justify-center"><input v-model="field.isPkKey" type="checkbox" class="checkbox" /></div>
          <div class="w-40"><input v-model="field.description" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-32 flex space-x-1">
            <button @click="deleteField(idx)" class="btn px-2 py-0.5 bg-red-500 text-white rounded">Delete</button>
            <button @click="moveField(idx, -1)" :disabled="idx === 0" class="btn px-2 py-0.5 bg-gray-300 dark:bg-gray-700 dark:text-gray-100 rounded">↑</button>
            <button @click="moveField(idx, 1)" :disabled="idx === fields.length - 1" class="btn px-2 py-0.5 bg-gray-300 dark:bg-gray-700 dark:text-gray-100 rounded">↓</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Key + Add -->
    <div class="flex items-center space-x-4 mt-8">
      <label class="font-bold">Key</label>
      <button @click="addKeyIndex" class="btn btn-primary px-3 py-1 bg-blue-500 text-white rounded">Add</button>
    </div>

    <!-- UNIQUE Keys Table -->
    <div class="overflow-x-auto">
      <div class="min-w-max border rounded">
        <div class="flex font-bold px-2 py-2 bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100">
          <div class="w-10">No</div>
          <div class="w-40">IndexName</div>
          <div class="w-64">FieldNames</div>
          <div class="w-24">Action</div>
        </div>
        <div v-for="(key, idx) in uniqueKeys" :key="key.id" class="flex items-center border-t px-2 py-1 bg-white text-gray-900 dark:bg-gray-900 dark:text-gray-100">
          <div class="w-10">{{ idx + 1 }}</div>
          <div class="w-40"><input v-model="key.name" type="text" class="input px-1 py-0.5 border rounded w-full" /></div>
          <div class="w-64 flex flex-wrap items-center gap-1">
            <div v-for="(tag, tIdx) in key.fieldNames" :key="tIdx" class="flex items-center bg-gray-200 dark:bg-gray-700 rounded px-1.5 py-0.5 mr-1 mb-1 min-h-6">
              <span class="text-sm">{{ tag }}</span>
              <button @click="removeTag(idx, tIdx)" class="ml-1 w-5 h-5 flex items-center justify-center rounded-full bg-gray-300 dark:bg-gray-600 text-gray-500 hover:text-red-500 hover:bg-red-200 dark:hover:bg-red-400 focus:outline-none p-0">
                <span class="text-xs">✕</span>
              </button>
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
            <button @click="deleteKeyIndex(idx)" class="btn px-2 py-0.5 bg-red-500 text-white rounded">Delete</button>
          </div>
        </div>
      </div>
    </div>

    <button @click="generateCode" class="btn btn-primary px-3 py-1 bg-blue-500 text-white rounded">Generate</button>
    <CodeEditor v-model="createTableSqlCode" />
    <CodeEditor v-model="cSharpCode" />
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