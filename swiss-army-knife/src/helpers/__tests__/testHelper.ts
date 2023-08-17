import * as fs from 'fs';
import path from 'path';
import jsdom from 'jsdom';

const { JSDOM } = jsdom;
const { window } = new JSDOM();
(global as any).document = window.document;

export function readFileContent(relativeFilePath: string): string {
    const filePath = path.join(__dirname, relativeFilePath);
    return fs.readFileSync(filePath, 'utf-8');
}