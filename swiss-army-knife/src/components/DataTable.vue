<script setup lang="ts">
import type { IDataTable } from '@/helpers/dataTypes';
import { computed } from 'vue';
interface DataTableProps {
    modelValue: IDataTable
}
const props = withDefaults(defineProps<DataTableProps>(), {
    modelValue: () => {
        return {
            columnNames: [],
            rows: [],
        }
    },
});

const columns = computed(() => {
    return props.modelValue.columnNames.map(name => {
        return {
            key: `${name}`,
            dataKey: `${name}`,
            title: `${name}`,
            width: 150,
        }
    });
});


const data = computed(() => {
    console.log('rows', props.modelValue.rows)
    return props.modelValue.rows;
});

</script>
<template>
    <!-- <el-table :data="props.modelValue.rows" stripe style="width: 100%">
        <template v-for="headerName in props.modelValue.columnNames" :key="headerName">
            <el-table-column :label="headerName" width="180">
                <template #default="scope">
                    <div style="display: flex; align-items: center">
                        <el-input v-model="scope.row[headerName]" placeholder="" />
                    </div>
                </template>
            </el-table-column>
        </template>
    </el-table> -->
    <el-auto-resizer>
        <template #default="{ height, width }">
            <el-table-v2 :columns="columns" :data="data" :width="height" :height="width" fixed />
        </template>
    </el-auto-resizer>
</template>
