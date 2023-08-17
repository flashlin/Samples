function fetchHeadersByTh(thead: HTMLTableSectionElement){
    const headerCells = thead.querySelectorAll('th');
    const headers = Array.from(headerCells).map(cell => cell.textContent || '');
    return headers;
}

function fetchHeadersByTd(thead: HTMLTableSectionElement){
    const headerCells = thead.querySelectorAll('td');
    const headers = Array.from(headerCells).map(cell => cell.textContent || '');
    return headers;
}

function fetchHeaders(thead: HTMLTableSectionElement){
    const handlers = [fetchHeadersByTh, fetchHeadersByTd];
    let headers: string[] = [];
    for(const handler of handlers){
        headers = handler(thead);
        if(headers.length != 0) return headers;
    }
    return [];
}

export function fetchTableData(table: HTMLTableElement) {
    const tableData: object[] = [];

    const rows = table.querySelectorAll('tr');
    const thead = table.querySelector('thead');
    
    let headers: string[] = [];
    if (thead) {
        headers = fetchHeaders(thead);
    } else if (rows.length > 0) {
        const firstRowCells = rows[0].querySelectorAll('td');
        headers = Array.from(firstRowCells).map(cell => cell.textContent || '');
    }

    headers = headers.map(x=> x.replace(/#/g, '_id').replace(/ /g, '_'));

    Array.from(rows).slice(1).forEach(row => {
        const rowData: { [key: string]: string } = {};
        const cells = row.querySelectorAll('td');
        cells.forEach((cell, index) => {
            const header = headers[index] || `column_${index + 1}`;
            rowData[header] = cell.textContent || '';
        });
        tableData.push(rowData);
    });

    return tableData;
}

export function fetchAllTable() {
    const tableDataList: object[][] = [];
    const tableElements = document.querySelectorAll('table');
    tableElements.forEach(table => {
        const tableData: object[] = fetchTableData(table);
        tableDataList.push(tableData);
    });
    return tableDataList;
}