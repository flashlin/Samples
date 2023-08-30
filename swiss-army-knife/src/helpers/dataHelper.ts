import { type IDataTable } from './dataTypes'

export function normalColumnName(text: string) {
    const text1 = text.replace(/#/g, '_id').replace(/ /g, '_');
    const removedNonAlphaNumeric = text1.replace(/[^a-zA-Z0-9_]+/g, '_');
    const consolidatedUnderscores = removedNonAlphaNumeric.replace(/_+/g, '_');
    if (consolidatedUnderscores.startsWith("_") && consolidatedUnderscores.endsWith("_")) {
        return consolidatedUnderscores.slice(1, -1);
    }
    return consolidatedUnderscores;
}

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
    const names = lines[0].split(',').map(name => normalColumnName(name));
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

export function* filter<T>(items: T[], condition: (item: T) => boolean) {
    for (let idx = 0; idx < items.length; idx++) {
        const item = items[idx];
        if (condition(item)) {
            yield item;
        }
    }
}

export function take<T>(generator: Iterable<T>, count: number): T[] {
    const items: T[] = [];
    for (const item of generator) {
        items.push(item);
        if (items.length === count) break;
    }
    return items;
}

export function zip<T1, T2>(a: T1[], b: T2[]) {
    return a.map((value, index) => [value, b[index]]);
}





export type TKey = string | number | Symbol;
export interface Entry<K extends TKey, V> {
    key: K;
    value: V;
}

export class Iterator<T> implements Iterable<T> {
    private data: T[] = [];
    constructor(data: T[]) {
        this.data = data;
    }

    *[Symbol.iterator](): IterableIterator<T> {
        for (const item of this.data) {
            yield item;
        }
    }
}


export class ObjectIterable<T> implements Iterable<Entry<string, T>> {
    private data: { [key: string]: T };

    constructor(data: { [key: string]: T }) {
        this.data = data;
    }

    *[Symbol.iterator](): IterableIterator<Entry<string, T>> {
        const keys = Object.keys(this.data);
        for (const key of keys) {
            yield {
                key,
                value: this.data[key]
            };
        }
    }
}

export class MapIterable<K extends TKey, V> implements Iterable<Entry<K, V>> {
    private data: Map<K, V>;

    constructor(data: Map<K, V>) {
        this.data = data;
    }

    *[Symbol.iterator](): IterableIterator<Entry<K, V>> {
        for (const [key, value] of this.data) {
            yield {
                key: key as K,
                value: value
            };
        }
    }
}

export class DataHelper<T> {
    private data: Iterable<T>;

    constructor(data: Iterable<T>) {
        this.data = data;
    }

    groupBy<TKey extends keyof T, K extends TKey>(propertyName: TKey): DataHelper<Entry<K, T[]>> {
        const groupedData = new Map<K, T[]>();
        for (const item of this.data) {
            const key = item[propertyName] as K;
            if (!groupedData.has(key)) {
                groupedData.set(key, []);
            }
            groupedData.get(key)!.push(item);
        }
        const data = new MapIterable(groupedData);
        return new DataHelper(data);
    }

    toArray(): T[] {
        const result: T[] = [];
        for (const item of this.data) {
            result.push(item);
        }
        return result;
    }
}