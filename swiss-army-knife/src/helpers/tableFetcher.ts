import { type IDataTable } from "./dataTypes";

function fetchHeadersByTh(thead: HTMLTableSectionElement) {
    const headerCells = thead.querySelectorAll('th');
    const headers = Array.from(headerCells).map(cell => cell.textContent || '');
    return headers;
}


function fetchHeadersByTd(thead: HTMLTableSectionElement) {
    const headerCells = thead.querySelectorAll('td');
    const headers = Array.from(headerCells).map(cell => cell.textContent || '');
    return headers;
}

function fetchHeaders(thead: HTMLTableSectionElement) {
    const handlers = [fetchHeadersByTh, fetchHeadersByTd];
    let headers: string[] = [];
    for (const handler of handlers) {
        headers = handler(thead);
        if (headers.length != 0) return headers;
    }
    return [];
}

function removeNonAlphaNumeric(text: string) {
    const removedNonAlphaNumeric = text.replace(/[^a-zA-Z0-9_]+/g, '_');
    const consolidatedUnderscores = removedNonAlphaNumeric.replace(/_+/g, '_');
    if (consolidatedUnderscores.startsWith("_") && consolidatedUnderscores.endsWith("_")) {
        return consolidatedUnderscores.slice(1, -1);
    }
    return consolidatedUnderscores;
}


export function fetchTableData(table: HTMLTableElement): IDataTable {
    const tableData: object[] = [];

    const rows = table.querySelectorAll('tr');
    const thead = table.querySelector('thead');

    let headers: string[] = [];
    if (thead) {
        headers = fetchHeaders(thead);
    } else if (rows.length > 0) {
        headers = fetchHeadersByTableTh(table);
        if (headers.length == 0) {
            headers = fetchHeadersByRow(rows[0]);
        }
    }

    headers = headers.map(x => x.replace(/#/g, '_id').replace(/ /g, '_'));
    headers = headers.map(x => removeNonAlphaNumeric(x));

    Array.from(rows).slice(1).forEach(row => {
        const rowData: { [key: string]: string } = {};
        const cells = row.querySelectorAll('td');
        let allEmpty = true;
        cells.forEach((cell, index) => {
            const header = headers[index] || `column_${index + 1}`;
            //rowData[header] = cell.textContent || '';
            rowData[header] = cell.innerText || '';
            if (cell.innerText != '') {
                allEmpty = false;
            }
        });
        if (!allEmpty) {
            tableData.push(rowData);
        }
    });

    return {
        columnNames: headers,
        rows: tableData
    };
}

function fetchHeadersByTableTh(table: HTMLTableElement) {
    const thCells = table.querySelectorAll('th');
    return Array.from(thCells).map(cell => cell.textContent || '');
}

function fetchHeadersByRow(row: HTMLTableRowElement) {
    const firstRowCells = row.querySelectorAll('td');
    return Array.from(firstRowCells).map(cell => cell.textContent || '');
}

export function fetchAllTable() {
    const tableDataList: IDataTable[] = [];
    const tableElements = document.querySelectorAll('table');
    tableElements.forEach(table => {
        const tableData = fetchTableData(table);
        const hasEmpty = tableData.columnNames.some(x => x === null || x === undefined || x === '');
        if (!hasEmpty && tableData.rows.length > 0) {
            tableDataList.push(tableData);
        }
    });
    return tableDataList;
}