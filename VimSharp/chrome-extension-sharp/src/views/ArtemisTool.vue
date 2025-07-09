<script lang="ts" setup>
import { inject, ref } from 'vue';
import LargeDataTable from '@/components/LargeDataTable.vue';
import CodeEditor from '@/components/CodeEditor.vue';
import { DataTable } from '@/tools/dataTypes';
import { ArtemisApi } from '@/tools/artemisApi';
import { DropboxItem } from '@/components/ComboDropboxTypes';
import ComboDropbox from '@/components/ComboDropbox.vue';
import TabControl from '@/components/TabControl.vue';
import { LoadingState, ProvideKeys } from '@/tools/ProvideTypes';
import { IntellisenseContext } from '@/components/CodeEditorTypes';
import { tokenizeSql } from '@/t1-sqlts/SqlTokenizer';

const dbFullNameList = ref<DropboxItem[]>([]);
const dbFullNameSelected = ref<string>('AccountDB (maia-z601)');
const dbName = computed(() => {
  return dbFullNameSelected.value.split('(')[0].trim();
})
const envName = ref<string>('staging');
const loginName = ref<string>('flash');
const password = ref<string>('123456');
const linqtsql = ref('')
const sqlText = ref(`
select top 1 PromotionEmail, [CasinoMigrated], [LastLoginProject], [MPDangerLevel], [SecurityMsg],  [email], [remark], [creatdate], [creator], [username], [pwdexpiry], [roleid], [closed], [alive], [LoginName], [LoginNameFlag], [custid], [firstname], [lastname], [address] 
from customer with (nolock) 
where username = 'tflash'
`)
const dt = ref<DataTable | null>(null);
const screenHistory = ref<string[]>([]);
const popupImage = ref<string | null>(null);
const activeTab = ref('query');
const envNameList = ref<DropboxItem[]>([
  { label: 'Staging', value: 'staging' },
  { label: 'Production', value: 'production' }
]);
const loginErrorMessage = ref<string | null>(null);
const loadingState = inject(ProvideKeys.LoadingState) as LoadingState;
const linqtsqlEditorRef = ref()

async function login() {
    loadingState.isLoading = true;
    let artemis = new ArtemisApi();
    try{
      await artemis.login({
          env_name: envName.value,
          username: loginName.value, 
          password: password.value 
      });
      loginErrorMessage.value = "Login Success";
    } catch(e: unknown){
        if (e instanceof Error) {
            loginErrorMessage.value = e.message;
        } else {
            loginErrorMessage.value = null;
        }
    } finally {
      loadingState.isLoading = false;
    }
}

async function test() {
    let artemis = new ArtemisApi();
    let resp = await artemis.query({ sql: sqlText.value, dbFullName: dbFullNameSelected.value });
    dt.value = resp.data;
    console.log(dt.value!.columns) 
}

async function getScreenHistory() {
    let artemis = new ArtemisApi();
    let resp = await artemis.getScreenHistory();
    screenHistory.value = resp.history;
}

async function onShowLinqtsqlIntellisense(context: IntellisenseContext) {
  // 這裡 context.content 是 [before, after]
  // 你可以根據 context 內容回傳 IntellisenseItem[]
  // 例如：
  const prevTokens = tokenizeSql(context.content[0]);
  if (prevTokens.length === 0) {
    return [
      { title: 'FROM', context: 'FROM ' },
    ];
  }
  if (prevTokens[prevTokens.length - 1] === 'FROM') {
    return [
      { title: 'Customer', context: 'Customer ' },
      { title: 'CustomerExtraInfo', context: 'CustomerExtraInfo ' },
    ];
  }
  const question = context.content[0] + " {cursor} " + context.content[1];
  
  return [
    { title: 'SELECT', context: 'SELECT ' },
    { title: 'FROM', context: 'FROM ' },
    { title: 'WHERE', context: 'WHERE ' },
    { title: 'ORDER BY', context: 'ORDER BY ' },
    { title: 'GROUP BY', context: 'GROUP BY ' },
    { title: 'HAVING', context: 'HAVING ' },
    { title: 'UNION', context: 'UNION ' },
    { title: 'INTERSECT', context: 'INTERSECT ' },
    // ...更多提示
  ]
}
</script>

<template>
  <div class="w-full space-y-6">
    <TabControl
      :tabList="[{ id: 'query', name: 'Query' }, { id: 'connect', name: 'Connect' }]"
      v-model:activeTab="activeTab"
    >
      <template #query>
        <div>
          <div>
              <button @click="test">TEST</button>
              <button @click="getScreenHistory">Get Screen History</button>
          </div>
          <div v-if="screenHistory.length > 0" class="flex flex-row space-x-2 mt-2">
            <img v-for="(img, idx) in screenHistory" :key="idx" :src="`data:image/png;base64,${img}`" width="384" height="512" class="object-contain border cursor-pointer" 
              @click="popupImage = img" />
          </div>

          <div>
              <label class="block mb-1 font-bold">Use Database</label>
              <ComboDropbox :list="dbFullNameList" v-model="dbFullNameSelected" placeholder="please input database name..."/>
          </div>

          <div>
            <label class="block mb-1 font-bold">Code Editor</label>
            <div style="height: 200px;">
              <CodeEditor ref="linqtsqlEditorRef" v-model="linqtsql" class="w-full h-full" :onShowIntellisense="onShowLinqtsqlIntellisense" />
            </div>
          </div>

          <div>
            <label class="block mb-1 font-bold">Predict Execute Code</label>
            <textarea 
              class="code-editor w-full" 
              v-model="sqlText" 
              spellcheck="false"
              style="min-height: 100px;"
            ></textarea>
          </div>

          <div>
            <label class="block mb-1 font-bold">Result</label>
            <LargeDataTable :dt="dt" class="w-full" />
          </div>
        </div>
      </template>
      <template #connect>
        <div class="flex flex-col space-y-2 w-64">
          <label class="font-bold">EnvName</label>
          <ComboDropbox :list="envNameList" v-model="envName" />
          <label class="font-bold">LoginName</label>
          <input type="text" v-model="loginName" class="border rounded px-2 py-1 bg-gray-900 text-white" />
          <label class="font-bold">Password</label>
          <input type="password" v-model="password" class="border rounded px-2 py-1 bg-gray-900 text-white" />
          <button @click="login" class="mt-2 px-4 py-1 bg-blue-600 text-white rounded hover:bg-blue-700">Login</button>
          <span v-if="loginErrorMessage" class="text-red-500 mt-1">{{ loginErrorMessage }}</span>
        </div>
      </template>
    </TabControl>
    <div v-if="popupImage" class="fixed inset-0 flex items-center justify-center bg-black bg-opacity-60 z-50">
      <div class="relative">
        <button @click="popupImage = null" class="absolute -top-3 -right-3 bg-white rounded-full shadow p-1 text-black hover:bg-gray-200">✕</button>
        <img :src="`data:image/png;base64,${popupImage}`" class="max-w-full max-h-[90vh] rounded shadow-lg" />
      </div>
    </div>
  </div>
</template>
