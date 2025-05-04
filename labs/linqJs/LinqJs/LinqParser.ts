import {
  LinqQueryExpr,
  LinqFromExpr,
  LinqSelectExpr,
  LinqJoinExpr,
  LinqIdentifierExpr,
  LinqMemberAccessExpr
} from './LinqExprs';

// LinqParser class for parsing LINQ query string to AST
export class LinqParser {
   // 解析成 LinqMemberAccessExpr
   private _parseMemberAccess(expr: string): LinqMemberAccessExpr {
     const parts = expr.split('.');
     const member = new LinqMemberAccessExpr();
     const target = new LinqIdentifierExpr();
     target.Name = parts[0];
     member.Target = target;
     member.MemberName = parts[1] || '';
     return member;
   }
   // 解析 LINQ 查詢字串，回傳 AST
   public parse(query: string): LinqQueryExpr {
     // 支援 join 語法與 select new
     const joinMatch = query.match(/from\s+(\w+)\s+in\s+(\w+)(?:\s+join\s+(\w+)\s+in\s+(\w+)\s+on\s+([\w\.]+)\s+equals\s+([\w\.]+))?\s+select\s+(new\s+\{[^}]+\}|\w+)/);
     if (!joinMatch) throw new Error('查詢語法錯誤');
     const fromId = joinMatch[1];
     const fromSrc = joinMatch[2];
     const joinId = joinMatch[3];
     const joinSrc = joinMatch[4];
     const joinLeft = joinMatch[5];
     const joinRight = joinMatch[6];
     const selectRaw = joinMatch[7];
     const expr = new LinqQueryExpr();
     expr.From = new LinqFromExpr();
     expr.From.Identifier = fromId;
     expr.From.Source = fromSrc;
     // 處理 join
     if (joinId && joinSrc && joinLeft && joinRight) {
       const joinExpr = new LinqJoinExpr();
       joinExpr.Identifier = joinId;
       joinExpr.Source = joinSrc;
       joinExpr.OuterKey = this._parseMemberAccess(joinLeft);
       joinExpr.InnerKey = this._parseMemberAccess(joinRight);
       expr.Joins.push(joinExpr);
     }
     // 處理 select
     expr.Select = new LinqSelectExpr();
     if (/^new\s+\{/.test(selectRaw)) {
       // select new 結構
       (expr.Select.Expression as any).Raw = selectRaw;
     } else {
       const idExpr = new LinqIdentifierExpr();
       idExpr.Name = selectRaw;
       expr.Select.Expression = idExpr;
     }
     return expr;
   }
} 