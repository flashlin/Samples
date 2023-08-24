<script setup lang="ts">
import type { IMergeTableReq } from '../helpers/dataTypes';
import { type IDataTable } from '../helpers/dataTypes';
import { ref } from 'vue';

// interface IData {
//   table1: IDataTable;
// }

// const data = ref<IData>({
//   table1: {
//     columnNames: [],
//     joinOnColumns: [],
//   },
// });

const table1 = {
  columnNames: ['id', 'name', 'birth'],
  rows: [
    {id: 1, name: 'flash', birth: new Date(2023,1,1)},
    {id: 2, name: 'jack', birth: new Date(2023,2,1)},
    {id: 2, name: 'mary', birth: new Date(2023,3,1)},
  ]
}

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
          <ListBox :modelValue="table1" dataKeyField="id" dataValueField="name" />
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
