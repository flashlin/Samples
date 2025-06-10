import { createParseInput, parseIntValue } from '../SqlParser';
import { SqlType, SqlExpr } from '../Expressions/SqlType';
import { TextSpan } from '../StringParser';

describe('SqlParser', () => {
  describe('parseSql', () => {
    it('應該解析整數值', () => {
      // 準備
      const sql = "123";
      
      // 執行
      const input = createParseInput(sql);
      const result = parseIntValue()(input);
      
      // 驗證
      expect(result.value).toEqual(
        new SqlExpr(
          SqlType.IntValue, 
          new TextSpan("123", 0, 3)
        )
      );
    });
  });
}); 