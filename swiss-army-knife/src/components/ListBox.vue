<script setup lang="ts">
import { reactive } from 'vue';
import { Top, Bottom } from '@element-plus/icons-vue'

interface ListBoxProps {
    modelValue: any[]
}

const props = withDefaults(defineProps<ListBoxProps>(), {
    modelValue: () => {
        return [];
    },
});

const data = reactive(props.modelValue.map((row, idx) => {
    console.log('idx', idx);
    return {
        id: idx,
        isSelected: false,
        field: row,
    };
}));

const handleOnUp = (idx: number) => {
    console.log('up', idx)
}

const handleOnDown = (idx: number) => {
    console.log('down', idx)
}

</script>
<template>
    <el-table :show-header="false" :data="data">
        <el-table-column label="Date" width="20">
            <template #default="scope">
                <div style="display: flex; align-items: center">
                    <el-checkbox v-model="scope.row.isSelected" :label="scope.row.field" />
                </div>
            </template>
        </el-table-column>
        <el-table-column prop="field" width="150" />
        <el-table-column label="Date" width="150">
            <template #default="scope">
                <div style="display: flex; align-items: center">
                    <el-icon :size="15">
                        <Top @click="handleOnUp(scope.row.id)" />
                    </el-icon>
                    <el-icon :size="15">
                        <Bottom @click="handleOnDown(scope.row.id)" />
                    </el-icon>
                </div>
            </template>
        </el-table-column>
    </el-table>
</template>

<style scoped></style>