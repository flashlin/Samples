<script setup lang="ts">
import type { IDataTable } from '@/helpers/dataTypes';
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

</script>
<template>
    <el-table :data="props.modelValue.rows" stripe style="width: 100%">
        <template v-for="headerName in props.modelValue.columnNames" :key="headerName">
            <el-table-column :label="headerName" width="180">
                <template #default="scope">
                    <div style="display: flex; align-items: center">
                        <el-input v-model="scope.row[headerName]" placeholder="" />
                    </div>
                </template>
            </el-table-column>
        </template>
        <el-table-column fixed="right" label="Operations" width="80">
            <template #default>
                <el-button link type="primary" size="small">Delete</el-button>
            </template>
        </el-table-column>
    </el-table>
</template>
