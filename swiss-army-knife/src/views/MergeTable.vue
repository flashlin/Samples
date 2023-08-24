<script setup lang="ts">
import type { IMergeTableForm } from '../helpers/dataTypes';
import { computed, ref, toRefs } from 'vue';

interface IMergeTableProps {
  modelValue: IMergeTableForm;
}

const props = defineProps<IMergeTableProps>();

const table1Columns = computed(() => {
  return props.modelValue.table1.columns.map(columnName => ({
    label: columnName,
    value: columnName
  }));
});

const table2Columns = computed(() => {
  return props.modelValue.table2.columns.map(columnName => ({
    label: columnName,
    value: columnName
  }));
});

// interface IData {
//   table1: IDataTable;
// }

// const data = ref<IData>({
//   table1: {
//     columnNames: [],
//     joinOnColumns: [],
//   },
// });

// const emit = defineEmits<{
//   (e: 'confirm', value: IMergeTableReq): void;
// }>();

// const handleClickMergeTable = () => {
//   emit('confirm', {
//     name: '',
//     table1: {
//       name: '',
//       columns: [],
//       joinOnColumns: [],
//     },
//     table2: {
//       name: '',
//       columns: [],
//       joinOnColumns: [],
//     },
//   });
// };
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
          <ListBox :modelValue="table1Columns" />
        </el-col>
        <el-col :span="12">
          <ListBox :modelValue="table2Columns" />
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>
