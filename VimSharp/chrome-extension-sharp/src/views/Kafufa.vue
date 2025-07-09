<script setup lang="ts">
import { decode as msgpackDecode } from '@msgpack/msgpack'
import { decompress } from 'lz4js'
import { ref } from 'vue'

const v1 = "x15j0gAAAFfwEpjOAIrgT6oxMDIzMjA3NzY2rnRiZWFyaWRyMjUwNzAyvA4A8AouaHNpdW5nQHRpdGFuc29mdC5jb20uc2ekHQDwBqJFTqZTQk9UT1CqMDcvMDcvMjAyNQ"

interface RegisterSuccessMessage {
  CustomerId: number
  AccountId: string
  LoginName: string
  Email: string
  FirstName: string
  Language: string
  RegBrand: string
  CreatedDate: string
}

function decodeRegisterSuccessMessage(base64: string): RegisterSuccessMessage {
  // 1. base64 轉 Uint8Array
  const buffer = Uint8Array.from(atob(base64), c => c.charCodeAt(0))
  // 2. LZ4 解壓縮
  const decompressed = decompress(buffer)
  // 3. MessagePack decode
  const arr = msgpackDecode(decompressed) as any[]
  return {
    CustomerId: arr[0],
    AccountId: arr[1],
    LoginName: arr[2],
    Email: arr[3],
    FirstName: arr[4],
    Language: arr[5],
    RegBrand: arr[6],
    CreatedDate: arr[7],
  }
}

const msg = ref<RegisterSuccessMessage | null>(null)
msg.value = decodeRegisterSuccessMessage(v1)
</script>

<template>
  <div>
    <h2>RegisterSuccessMessage 解碼結果</h2>
    <pre>{{ msg }}</pre>
  </div>
</template>
