import {
  LinqQueryExpr,
  LinqBinaryExpr,
  LinqMemberAccessExpr,
  LinqLiteralExpr
} from './LinqExprs';

// LinqExecutor class for executing LinqQueryExpr
export class LinqExecutor {
   // 儲存資料來源
   public Data: Record<string, any[]> = {};
 
   // 執行查詢
   public query(expr: LinqQueryExpr): any[] {
     // 取得主資料來源
     const fromSource = this.Data[expr.From.Source];
     if (!fromSource) {
       throw new Error(`資料來源 ${expr.From.Source} 不存在`);
     }
     // 過濾 where 條件
     let result = this._applyWhere(fromSource, expr);
     // 處理 join
     if (expr.Joins && expr.Joins.length > 0) {
       result = this._applyJoins(result, expr);
     }
     // 處理 select
     if (expr.Select) {
       result = this._applySelect(result, expr);
     }
     return result;
   }
 
   // 處理 where 條件
   private _applyWhere(list: any[], expr: LinqQueryExpr): any[] {
     if (!expr.Where) return list;
     // 只支援 c.status == "active" 這種簡單條件
     const cond = expr.Where.Condition as LinqBinaryExpr;
     if (!cond || cond.Operator !== '==') return list;
     const left = cond.Left as LinqMemberAccessExpr;
     const right = cond.Right as LinqLiteralExpr;
     return list.filter(item => {
       // 只支援一層屬性
       return item[left.MemberName] == right.Value;
     });
   }
 
   // 處理 join
   private _applyJoins(list: any[], expr: LinqQueryExpr): any[] {
     let result = list;
     for (const join of expr.Joins) {
       const joinSource = this.Data[join.Source];
       if (!joinSource) continue;
       const outerKey = (join.OuterKey as LinqMemberAccessExpr).MemberName;
       const innerKey = (join.InnerKey as LinqMemberAccessExpr).MemberName;
       result = result.flatMap(item => {
         const matches = joinSource.filter(j => item[outerKey] === j[innerKey]);
         return matches.map(j => ({ ...item, [join.Identifier]: j }));
       });
     }
     return result;
   }
 
   // 處理 select
   private _applySelect(list: any[], expr: LinqQueryExpr): any[] {
     // 只支援 select {c, o} 這種簡單結構
     return list.map(item => {
       return { c: item, o: item['o'] };
     });
   }
}
 