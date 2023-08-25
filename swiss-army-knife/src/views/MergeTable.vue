<script setup lang="ts">
import { MessageTypes, type IMergeTableForm } from '../helpers/dataTypes';
import { computed, ref, } from 'vue';
import { useFlashKnifeStore } from '../stores/flashKnife';
import ListBox from '../components/ListBox.vue';
const flashKnifeStore = useFlashKnifeStore();
const { notify } = flashKnifeStore;

interface IMergeTableProps {
  modelValue: IMergeTableForm;
}

const props = defineProps<IMergeTableProps>();
const table1Ref = ref<InstanceType<typeof ListBox>>();
const table2Ref = ref<InstanceType<typeof ListBox>>();
const tableName = ref(props.modelValue.name);

const table1Columns = computed(() => {
  return props.modelValue.table1.joinOnColumns.map(columnName => ({
    label: columnName,
    value: columnName
  }));
});

const table2Columns = computed(() => {
  return props.modelValue.table2.joinOnColumns.map(columnName => ({
    label: columnName,
    value: columnName
  }));
});

const emit = defineEmits<{
  (e: 'confirm', value: IMergeTableForm): void;
}>();

const handleClickConfirm = () => {
  const table1ColumnsSelected = table1Ref.value!.getSelectedValues();
  if (table1ColumnsSelected.length == 0) {
    notify(MessageTypes.Error, 'please table1 columns');
    return;
  }
  const table2ColumnsSelected = table2Ref.value!.getSelectedValues();
  if (table2ColumnsSelected.length == 0) {
    notify(MessageTypes.Error, 'please table2 columns');
    return;
  }
  if (tableName.value == '') {
    notify(MessageTypes.Error, 'please input import table name');
    return;
  }
  emit('confirm', {
    name: tableName.value,
    table1: {
      name: props.modelValue.table1.name,
      joinOnColumns: table1ColumnsSelected,
    },
    table2: {
      name: props.modelValue.table2.name,
      joinOnColumns: table2ColumnsSelected,
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
          <ListBox :modelValue="table1Columns" ref="table1Ref" />
        </el-col>
        <el-col :span="12">
          <ListBox :modelValue="table2Columns" ref="table2Ref" />
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