<script lang="ts" setup>
import { ref } from 'vue';

const matches = ref([
  { id: 1, time: '10:00', home: 'Team A', away: 'Team B', homeScore: 0, awayScore: 0 },
  { id: 2, time: '12:00', home: 'Team C', away: 'Team D', homeScore: 0, awayScore: 0 },
  { id: 3, time: '14:00', home: 'Team E', away: 'Team F', homeScore: 0, awayScore: 0 },
]);
</script>
<template>
  <div class="featured-matches">
    <!-- Header -->
    <div class="header">
      <h2>Featured Matches</h2>
      <div class="countdown">
        <span>Place your qualifying bet before</span>
        <span class="timer">03 : 09 : 12 : 05</span>
        <span class="timer-label">DAYS HRS MINS SECS</span>
      </div>
    </div>

    <!-- Progress Bar -->
    <div class="progress-section">
      <div class="progress-label">Almost there!</div>
      <div class="progress-bar">
        <div class="progress-track">
          <div class="progress-ball"></div>
        </div>
      </div>
      <div class="progress-steps">
        <div class="step completed">
          <span class="icon">✔️</span>
          <span>Prediction submitted</span>
        </div>
        <div class="step">
          <span class="icon">⭕</span>
          <span>
            Place qualifying bet
            <span class="tooltip">
              A qualifying bet is a bet placed on a match in the same round.
              <a href="#">View T&amp;C.</a>
            </span>
          </span>
        </div>
      </div>
      <button class="place-bet-btn">Place bet now</button>
    </div>

    <!-- Match List -->
    <div class="match-list">
      <div class="match" v-for="match in matches" :key="match.id">
        <div class="match-time">{{ match.time }}</div>
        <div class="teams">
          <span class="team">{{ match.home }}</span>
          <span class="score">{{ match.homeScore }} : {{ match.awayScore }}</span>
          <span class="team">{{ match.away }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
