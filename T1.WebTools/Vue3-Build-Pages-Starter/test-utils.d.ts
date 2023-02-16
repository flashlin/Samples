/* eslint-disable */
import type { VueWrapper } from '@vue/test-utils';
import { DOMWrapper } from '@vue/test-utils/dist/domWrapper'
declare module '@vue/test-utils' {
  interface VueWrapper<T> {
    findByTestId<T extends Element = Element>(testId: string): VueWrapper<T>
    getByTestId<T extends Element = Element>(testId: string): Omit<DOMWrapper<T>, 'exists'>
  }
}