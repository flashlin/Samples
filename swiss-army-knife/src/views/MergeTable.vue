<script setup lang="ts">
import type { IMergeTableForm } from '../helpers/dataTypes';
import { computed, ref, } from 'vue';

interface IMergeTableProps {
  modelValue: IMergeTableForm;
}

const props = defineProps<IMergeTableProps>();

const tableName = ref(props.modelValue.name);

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

const emit = defineEmits<{
  (e: 'confirm', value: IMergeTableForm): void;
}>();

const handleClickConfirm = () => {

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
          <el-input v-model="tableName" placeholder="table name" />
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
      <el-row :span="24" justify="center">
        <el-button @click="handleClickConfirm">Confirm</el-button>
      </el-row>
    </el-card>
  </div>
</template>

<style scoped>
.el-row {
  margin-bottom: 20px;
}
</style>