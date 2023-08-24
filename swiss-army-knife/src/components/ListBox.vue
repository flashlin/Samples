<script setup lang="ts">
import { ref } from 'vue';
import { Top, Bottom } from '@element-plus/icons-vue'
import { type IDataTable } from '../helpers/dataTypes';

interface ListBoxProps {
    modelValue: IDataTable;
    dataKeyField: string;
    dataValueField: string;
}

const props = withDefaults(defineProps<ListBoxProps>(), {
    modelValue: () => {
        return {
            columnNames: [],
            rows: [],
        };
    },
    dataKeyField: '_id_',
    dataValueField: 'name',
});

interface IListBoxExpose {
    getSelectedValues(): any[];
}

defineExpose<IListBoxExpose>({
    getSelectedValues: () => {
        return data.value.filter(row => row._isSelected_)
            .map(row => {
                const newRow: any = {};
                const labelField = getLabelField();
                newRow[props.dataKeyField] = row[props.dataKeyField];
                newRow[labelField] = row[labelField];
                return newRow;
            });
    }
})

const data = ref(props.modelValue.rows.map((row, idx) => {
    return {
        ...row,
        _id_: idx,
        _isSelected_: false,
    };
}));

const getLabelField = () => {
    let valueField = props.dataValueField;
    if (props.dataValueField == '') {
        valueField = props.modelValue.columnNames[0];
    }
    return valueField;
}

const handleOnUp = (idx: number) => {
    const dataValue = data.value;
    const index = dataValue.findIndex(row => row._id_ == idx);
    if (index < 0) {
        return;
    }
    const currItem = dataValue[index];
    const prevItem = dataValue[index - 1];
    const prefix = dataValue.slice(0, index - 1);
    const postfix = dataValue.slice(index + 1);
    const newData = [...prefix, currItem, prevItem, ...postfix];
    data.value = newData;
}

const handleOnDown = (idx: number) => {
    const dataValue = data.value;
    const index = dataValue.findIndex(row => row._id_ == idx);
    if (index >= data.value.length - 1) {
        return;
    }
    const currItem = dataValue[index];
    const nextItem = dataValue[index + 1];

    const prefix = index == 0 ? [] : dataValue.slice(0, index);
    const postfix = dataValue.slice(index + 2);
    const newData = [...prefix, nextItem, currItem, ...postfix];
    data.value = newData;
}

</script>
<template>
    <el-table :show-header="false" :data="data">
        <el-table-column label="Date" width="20">
            <template #default="scope">
                <div style="display: flex; align-items: center">
                    <el-checkbox v-model="scope.row._isSelected_" label="" />
                </div>
            </template>
        </el-table-column>
        <el-table-column :prop="getLabelField()" width="150" />
        <el-table-column label="Date" width="50">
            <template #default="scope">
                <div style="display: flex; align-items: center">
                    <el-icon :size="15" v-if="scope.$index == 0">
                    </el-icon>
                    <el-icon :size="15" v-if="scope.$index != 0">
                        <Top @click="handleOnUp(scope.row._id_)" />
                    </el-icon>
                    <el-icon :size="15" v-if="scope.$index < data.length - 1">
                        <Bottom @click="handleOnDown(scope.row._id_)" />
                    </el-icon>
                </div>
            </template>
        </el-table-column>
    </el-table>
</template>

<style scoped></style>