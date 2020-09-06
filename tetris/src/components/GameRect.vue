<template>
   <div>
      <div class="tetris">
         <div class="row" v-for="row in flatRect" :key="row.rowId">
            <div :class="box.klass" v-for="box in row.cols" :key="box.columnId"></div>
         </div>
      </div>
   </div>
</template>

<style lang="scss">
.tetris {
   border: 4px solid #000;
   padding: 6px 2px 2px 6px;
   width: 296px;
   height: 576px;
   background: #9ead86;

   .row div.c {
      border-color: #000;
   }
   .row div {
      display: block;
      width: 24px;
      height: 24px;
      padding: 1px;
      border: 4px solid #879372;
      margin: 0 4px 4px 0;
      float: left;
   }
   .row div.c:after {
      background: #000;
   }
   .row div:after {
      content: '';
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

<script lang="ts" src="./GameRect.ts"></script>
