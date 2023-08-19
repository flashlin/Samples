import { type IDataTable } from './dataTypes'

export function getCurrentTime(): string {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    return `${year}${month}${day}T${hours}${minutes}${seconds}`;
}

export function exportToCsv(name: string, data: any[]) {
    const csvContent = "data:text/csv;charset=utf-8," + data.map(item => Object.values(item).join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${name}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
}

export function readFileContentAsync(file: File): Promise<string> {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = function (event) {
            if (event.target) {
                const content = event.target.result as string;
                resolve(content);
                //const objectArray = parseContentToObjectArray(content);
                //output.textContent = JSON.stringify(objectArray, null, 2);
            }
        };
        reader.readAsText(file);
    })
}

export function parseCsvContentToObjectArray(content: string): IDataTable {
    const lines = content.split('\n');
    const objectArray: any[] = [];
    const names = lines[0].split(',');
    lines.slice(1).forEach(line => {
        let idx = 0;
        const item: any = {};
        line.split(',').forEach(value => {
            item[names[idx]] = value;
            idx++;
        });
        objectArray.push(item);
    });
    return {
        columnNames: names,
        rows: objectArray
    };
}

export function getObjectKeys<T extends object>(obj: T) {
    const keys: string[] = [];
    for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            keys.push(key);
        }
    }
    return keys;
}


export type MapFn = (key: string, value: any) => any;

export function mapObject<T extends object>(obj: T, mapFn: MapFn) {
    const newObj: any = {};
    for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            const value = obj[key];
            newObj[key] = mapFn(key, value);
        }
    }
    return newObj;
}