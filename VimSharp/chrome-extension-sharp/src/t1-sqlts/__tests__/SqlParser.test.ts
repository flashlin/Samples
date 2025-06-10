import { parseSql } from '../SqlParser';
import { SqlType, SqlExpr } from '../Expressions/SqlType';
import { TextSpan } from '../StringParser';

describe('SqlParser', () => {
  describe('parseSql', () => {
    it('應該解析整數值', () => {
      // 準備
      const sql = "123";
      
      // 執行
      const result = parseSql(sql);
      
      // 驗證
      expect(result).toHaveLength(1);
      expect(result[0]).toBeInstanceOf(SqlExpr);
      expect(result[0].SqlType).toBe(SqlType.IntValue);
      expect(result[0].Span).toBeInstanceOf(TextSpan);
      expect(result[0].Span.Offset).toBe(0);
      expect(result[0].Span.Length).toBe(3);
      expect(result[0].Span.Word).toBe("123");
    });
  });
}); 