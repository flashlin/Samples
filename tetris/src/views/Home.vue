<template>
  <div>
    <div class="tetris">
      <div class="row">
        <div class="c"></div>
        <div class></div>
        <div class="c"></div>
        <div class="c"></div>
        <div class="c"></div>
        <div class="c"></div>
        <div class></div>
        <div class></div>
        <div class></div>
        <div class></div>
      </div>
      <div class="row">
        <div class="c"></div>
        <div class></div>
        <div class="c"></div>
        <div class="c"></div>
        <div class="c"></div>
        <div class="c"></div>
        <div class></div>
        <div class></div>
        <div class></div>
        <div class></div>
      </div>
    </div>
  </div>
</template>


<style lang="scss">
// .mini_cell {
//   float: left;
//   width: 14px;
//   height: 14px;
//   border: 1px solid rgb(122, 122, 107);
//   margin: -1px 0 0 -1px;
// }

// .mini_cell.active {
//   background: #999;
//   border: 1px solid #ccc;
// }

.tetris {
  border: 4px solid #000;
  padding: 6px 2px 2px 6px;
  width: 400px;
  height: 600px;
  background: #9ead86;

  // .row {
  //   width: 440px;
  //   height: 20px;
  // }
  .row div.c {
    border-color: #000;
  }
  .row div {
    display: block;
    width: 24px;
    height: 24px;
    padding: 4px;
    border: 4px solid #879372;
    margin: 0 4px 4px 0;
    float: left;
  }
  .row div.c:after {
    background: #000;
  }
  .row div:after {
    content: "";
    display: block;
    width: 14px;
    height: 14px;
    background: #879372;
    overflow: hidden;
  }
}

* {
  box-sizing: border-box;
}
</style>

<script lang="ts" src="./Home.ts">
</script>