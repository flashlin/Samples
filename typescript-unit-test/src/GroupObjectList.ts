export interface IColumn {
    index: number;
    key: string;
    value: any;
}

export interface IGroupCounts {
    [columnIndex: number]: number;
}

function GetColumns(obj: any): IColumn[] {
    let columns: IColumn[] = [];
    Object.entries(obj).forEach(([key, value], index) => {
        columns.push({
            index,
            key,
            value
        });
    });
    return columns;
}

function GetGroupKeyBy(obj: any, byGroupName: string, groupColumnNames: string[]): string {
    const byGroupIndex = groupColumnNames.indexOf(byGroupName);
    const groupNames = groupColumnNames.filter((groupName, index) => {
        return index <= byGroupIndex;
    });
    let groupKey = '';
    groupNames.forEach((groupName, index) => {
        groupKey += `${obj[groupName]}`;
        if (index < groupNames.length - 1) {
            groupKey += '-';
        }
    });
    return groupKey;
}

function EnsureGroupCount(groups: IGroupCounts[], rowIndex: number, columnIndex: number) {
    if (groups[rowIndex] === undefined) {
        groups[rowIndex] = [];
    }
    if (groups[rowIndex][columnIndex] === undefined) {
        groups[rowIndex][columnIndex] = 0;
    }
}

function CountSameValue(objList: any[], rowIndex: number, column: IColumn, groupColumnNames: string[]): number {
    const firstRow = objList[rowIndex];
    const firstKey = GetGroupKeyBy(firstRow, column.key, groupColumnNames);
    let count = 0;
    for (let index = rowIndex; index < objList.length; index++) {
        const row = objList[index];
        const currentKey = GetGroupKeyBy(row, column.key, groupColumnNames);
        if (currentKey !== firstKey) {
            break;
        }
        count++;
    }
    return count;
}

function ComputeGroupRow(rowIndex: number, objList: any[], groupColumnNames: string[], groups: IGroupCounts[]) {
    const columns = GetColumns(objList[0]);
    columns.forEach(column => {
        EnsureGroupCount(groups, rowIndex, column.index);
        if( groups[rowIndex][column.index] != 0) {
            return;
        }
        if( groupColumnNames.indexOf(column.key) == -1 ) {
            groups[rowIndex][column.index] = 1;
            return;
        }
        const sameCount = CountSameValue(objList, rowIndex, column, groupColumnNames);
        groups[rowIndex][column.index] = sameCount;
        for (let index = rowIndex + 1; index < rowIndex + sameCount; index++) {
            EnsureGroupCount(groups, index, column.index);
            groups[index][column.index] = 1;
        }
    });
}

function ComputeGroups(objList: any[], groupColumnNames: string[], groups: IGroupCounts[]) {
    objList.forEach((row, rowIndex) => {
        ComputeGroupRow(rowIndex, objList, groupColumnNames, groups);
    });
}

export function GroupObjectList(objList: any[], groupColumnNames: string[]): IGroupCounts[] {
    let groups: IGroupCounts[] = [];
    ComputeGroups(objList, groupColumnNames, groups);
    return groups;
}