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
      
      // 使用簡潔的方式驗證 SqlExpr 物件
      expect(result[0]).toEqual(
        new SqlExpr(
          SqlType.IntValue, 
          new TextSpan("123", 0, 3)
        )
      );
    });
  });
}); 