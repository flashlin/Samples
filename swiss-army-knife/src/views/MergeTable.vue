<script setup lang="ts">
import type { IMergeTableReq } from '../helpers/dataTypes';
import { type IDisplayDataTable } from '../helpers/dataTypes';
import { ref } from 'vue';

interface IData {
  table1: IDisplayDataTable;

}

const data = ref<IData>({
});

const emit = defineEmits<{
  (e: 'confirm', value: IMergeTableReq): void;
}>();

const handleClickMergeTable = () => {
  emit('confirm', {
    name: '',
    table1: {
      name: '',
      columns: [],
      joinOnColumns: [],
    },
    table2: {
      name: '',
      columns: [],
      joinOnColumns: [],
    },
  });
};
</script>

<template>
  <div>
    <el-card width="480px">
      <el-row :gutter="1">
        <el-col :span="24">
          <el-input placeholder="table name" />
        </el-col>
      </el-row>
      <el-row>
        <el-col :span="12">
          <select name="cars" size="5">
            <option value="volvo">Volvo</option>
            <option value="saab">Saab</option>
            <option value="mercedes">Mercedes</option>
            <option value="audi">Audi</option>
          </select>
        </el-col>
        <el-col :span="12">
          <select name="cars" size="5">
            <option value="volvo">Volvo</option>
            <option value="saab">Saab</option>
            <option value="mercedes">Mercedes</option>
            <option value="audi">Audi</option>
          </select>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>
