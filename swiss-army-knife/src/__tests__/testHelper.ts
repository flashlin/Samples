import { expect } from 'vitest';

export const getItName = () => {
    const fullTestName = expect.getState().currentTestName ?? '';
    const firstIndex = fullTestName.indexOf('>');
    const secondIndex = fullTestName.indexOf('>', firstIndex + 1);
    const currentTestName = fullTestName.substring(secondIndex + 1).trimStart();
    return currentTestName;
};