import { describe, expect, test } from "@jest/globals";
import { program } from '@/ai'; 

test('ai1', () => {
  const result = program(1);

  expect(result).toBe(1);
});
