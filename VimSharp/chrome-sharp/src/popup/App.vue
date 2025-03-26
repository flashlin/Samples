<template>
  <div class="container">
    <h1>我的 Chrome 擴充功能</h1>
    <p>這是一個使用 Vue 和 TypeScript 構建的 Chrome 擴充功能。</p>
    
    <button @click="handleButtonClick" class="action-button">點擊我</button>
    
    <div v-if="showResult" class="result">
      <p>{{ resultMessage }}</p>
    </div>
    
    <div v-if="installDate" class="info-section">
      <p class="install-date">安裝日期: {{ formattedInstallDate }}</p>
    </div>
    
    <div v-if="counter !== null" class="info-section">
      <p class="counter">總點擊次數: {{ counter }}</p>
    </div>
    
    <div v-if="pageInfo" class="page-info">
      <h3>當前頁面信息</h3>
      <p><strong>標題:</strong> {{ pageInfo.title }}</p>
      <p><strong>鏈接數:</strong> {{ pageInfo.links }}</p>
      <p><strong>圖片數:</strong> {{ pageInfo.images }}</p>
    </div>
    
    <div class="toggle-section">
      <label class="toggle-label">
        <input type="checkbox" v-model="showFloatingButton" @change="toggleFloatingButton">
        顯示網頁浮動按鈕
      </label>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, computed } from 'vue';

interface PageInfo {
  title: string;
  url: string;
  links: number;
  images: number;
  headings: Array<{ type: string; text: string }>;
}

export default defineComponent({
  name: 'PopupApp',
  setup() {
    const showResult = ref(false);
    const resultMessage = ref('');
    const installDate = ref<string | null>(null);
    const counter = ref<number | null>(null);
    const pageInfo = ref<PageInfo | null>(null);
    const showFloatingButton = ref(true);
    
    // 格式化安裝日期
    const formattedInstallDate = computed(() => {
      if (!installDate.value) return '';
      return new Date(installDate.value).toLocaleDateString();
    });
    
    // 獲取安裝日期
    const getInstallDate = () => {
      chrome.storage.local.get('installDate', (data) => {
        if (data.installDate) {
          installDate.value = data.installDate;
        }
      });
    };
    
    // 獲取計數器值
    const getCounter = () => {
      chrome.storage.local.get('counter', (data) => {
        if (data.counter !== undefined) {
          counter.value = data.counter;
        }
      });
    };
    
    // 獲取上次點擊時間
    const getLastClickedTime = () => {
      chrome.storage.local.get('lastClicked', (data) => {
        if (data.lastClicked) {
          showResult.value = true;
          resultMessage.value = `上次點擊時間: ${data.lastClicked}`;
        }
      });
    };
    
    // 獲取浮動按鈕狀態
    const getFloatingButtonState = () => {
      chrome.storage.local.get('showFloatingButton', (data) => {
        showFloatingButton.value = data.showFloatingButton !== false;
      });
    };
    
    // 切換浮動按鈕
    const toggleFloatingButton = () => {
      chrome.storage.local.set({ showFloatingButton: showFloatingButton.value });
    };
    
    // 按鈕點擊處理
    const handleButtonClick = () => {
      showResult.value = true;
      const currentTime = new Date().toLocaleTimeString();
      resultMessage.value = `按鈕被點擊了！當前時間: ${currentTime}`;
      
      // 保存點擊時間
      chrome.storage.local.set({ lastClicked: currentTime });
      
      // 獲取當前頁面信息
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length > 0) {
          const activeTab = tabs[0];
          
          chrome.tabs.sendMessage(
            activeTab.id as number,
            { action: 'getPageInfo' },
            (response: PageInfo) => {
              if (response) {
                pageInfo.value = response;
              }
            }
          );
        }
      });
    };
    
    // 組件掛載時執行
    onMounted(() => {
      getInstallDate();
      getCounter();
      getLastClickedTime();
      getFloatingButtonState();
    });
    
    return {
      showResult,
      resultMessage,
      installDate,
      formattedInstallDate,
      counter,
      pageInfo,
      showFloatingButton,
      handleButtonClick,
      toggleFloatingButton
    };
  }
});
</script>

<style scoped>
.container {
  width: 300px;
  padding: 15px;
  background-color: #f8f9fa;
  font-family: 'Arial', sans-serif;
}

h1 {
  font-size: 18px;
  color: #333;
  margin-top: 0;
}

p {
  font-size: 14px;
  color: #666;
}

.action-button {
  background-color: #4285f4;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  margin-top: 10px;
}

.action-button:hover {
  background-color: #3367d6;
}

.result {
  margin-top: 15px;
  padding: 10px;
  background-color: #e9ecef;
  border-radius: 4px;
  min-height: 20px;
}

.info-section {
  margin-top: 10px;
  font-size: 14px;
}

.install-date {
  font-size: 12px;
  color: #666;
}

.counter {
  font-weight: bold;
}

.page-info {
  margin-top: 15px;
  padding: 10px;
  background-color: #e9ecef;
  border-radius: 4px;
}

.page-info h3 {
  font-size: 16px;
  margin-top: 0;
  margin-bottom: 10px;
}

.toggle-section {
  margin-top: 20px;
  padding-top: 10px;
  border-top: 1px solid #ddd;
}

.toggle-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}
</style> 