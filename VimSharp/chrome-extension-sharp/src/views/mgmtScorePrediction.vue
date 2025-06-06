<script lang="ts" setup>
import { ref } from 'vue';

const matches = ref([
  { id: 1, date: '06/16 10:00 (GMT+8)', home: 'Botafogo', away: 'Seattle Sounders FC', homeScore: 2, awayScore: 0 },
  { id: 2, date: '06/17 03:00 (GMT+8)', home: 'Chelsea FC', away: 'Club León', homeScore: 0, awayScore: 1 },
  { id: 3, date: '06/21 10:00 (GMT+8)', home: 'CA Boca Juniors', away: 'SL Benfica', homeScore: 2, awayScore: 0 },
  { id: 4, date: '06/21 12:00 (GMT+8)', home: 'CR Flamengo', away: 'Espérance Sportive de Tunis', homeScore: 0, awayScore: 1 },
]);
</script>
<template>
  <div class="min-h-screen bg-gradient-to-b from-black via-gray-900 to-black flex flex-col items-center py-8 px-2">
    <!-- Header -->
    <div class="w-full max-w-2xl rounded-t-2xl bg-gradient-to-r from-yellow-400 to-yellow-300 shadow-lg p-6 flex flex-col items-center border-b-4 border-yellow-600">
      <h2 class="text-2xl font-bold text-gray-900 tracking-wide mb-2">Featured Matches</h2>
      <div class="flex flex-col items-center">
        <span class="text-gray-900 text-sm font-medium">Place your qualifying bet before</span>
        <span class="text-2xl font-mono font-bold text-gray-900 mt-1">03 : 09 : 12 : 05</span>
        <span class="text-xs text-gray-800 tracking-widest mt-1">DAYS&nbsp;&nbsp;HRS&nbsp;&nbsp;MINS&nbsp;&nbsp;SECS</span>
      </div>
    </div>

    <!-- Progress Bar & Steps -->
    <div class="w-full max-w-2xl bg-black rounded-b-2xl shadow-lg px-6 pt-6 pb-4 flex flex-col items-center border-t border-yellow-600 relative z-10">
      <div class="w-full flex flex-col items-center">
        <span class="text-white font-semibold text-lg mb-2">Almost there!</span>
        <!-- Progress Bar -->
        <div class="relative w-3/4 h-2 bg-gray-700 rounded-full mb-6 flex items-center">
          <div class="absolute left-1/2 -translate-x-1/2 -top-3 w-8 h-8 bg-white border-4 border-yellow-400 rounded-full flex items-center justify-center shadow"><span class="text-2xl">⚽</span></div>
          <div class="absolute left-0 top-0 h-2 bg-yellow-400 rounded-full" style="width: 50%"></div>
        </div>
        <!-- Steps -->
        <div class="flex w-full justify-between items-center mb-4">
          <div class="flex items-center gap-2">
            <span class="w-5 h-5 flex items-center justify-center bg-green-500 rounded-full text-white text-xs">✔</span>
            <span class="text-white text-sm">Prediction submitted</span>
          </div>
          <div class="flex items-center gap-2 relative group">
            <span class="w-5 h-5 flex items-center justify-center bg-gray-700 rounded-full text-yellow-400 text-xs">●</span>
            <span class="text-white text-sm flex items-center">
              Place qualifying bet
              <span class="ml-1 relative">
                <svg class="w-4 h-4 text-gray-400 cursor-pointer" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><text x="12" y="16" text-anchor="middle" font-size="12" fill="currentColor">i</text></svg>
                <span class="absolute left-1/2 -translate-x-1/2 top-6 w-64 bg-white text-gray-900 text-xs rounded shadow-lg p-2 z-20 hidden group-hover:block">
                  A qualifying bet is a bet placed on a match in the same round.<br>
                  <a href="#" class="text-blue-600 underline">View T&amp;C.</a>
                </span>
              </span>
            </span>
          </div>
        </div>
        <button class="mt-2 bg-yellow-400 hover:bg-yellow-500 text-black font-bold py-2 px-6 rounded shadow transition">Place bet now</button>
      </div>
    </div>

    <!-- Match List -->
    <div class="w-full max-w-2xl bg-black rounded-b-2xl shadow-lg mt-6 divide-y divide-yellow-900 border border-yellow-900">
      <div v-for="match in matches" :key="match.id" class="flex flex-col md:flex-row items-center justify-between py-4 px-6">
        <div class="text-xs text-gray-400 w-40 md:w-48 text-left mb-2 md:mb-0">{{ match.date }}</div>
        <div class="flex items-center gap-4 flex-1 justify-center">
          <span class="text-white font-bold text-lg md:text-xl">{{ match.home }}</span>
          <span class="text-yellow-400 font-bold text-xl">{{ match.homeScore }} : {{ match.awayScore }}</span>
          <span class="text-white font-bold text-lg md:text-xl">{{ match.away }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
