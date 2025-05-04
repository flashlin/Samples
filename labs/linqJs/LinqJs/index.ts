// 定義基底類別 LinqExpr
export abstract class LinqExpr {}

// 定義查詢表達式 LinqQueryExpr，這是最常見的查詢結構
export class LinqQueryExpr extends LinqExpr {
  From: LinqFromExpr = new LinqFromExpr();
  Where?: LinqWhereExpr;
  Select?: LinqSelectExpr;
  Joins: LinqJoinExpr[] = [];
  Group?: LinqGroupByExpr;
  Into?: LinqIntoExpr;
  Order?: LinqOrderByExpr;
  Take?: LinqTakeExpr;
  Skip?: LinqSkipExpr;
}

// 定義不同的子查詢表達式（例如：From, Where, Select, etc.）
export class LinqFromExpr extends LinqExpr {
  Identifier: string = "";
  Source: string = "";
}

export class LinqWhereExpr extends LinqExpr {
  Condition: LinqValueExpr = new LinqBinaryExpr();
}

export class LinqSelectExpr extends LinqExpr {
  Expression: LinqValueExpr = new LinqIdentifierExpr();
}

export class LinqJoinExpr extends LinqExpr {
  Identifier: string = "";
  Source: string = "";
  OuterKey: LinqValueExpr = new LinqMemberAccessExpr();
  InnerKey: LinqValueExpr = new LinqMemberAccessExpr();
  IntoIdentifier: string = "";
}

export class LinqGroupByExpr extends LinqExpr {
  Element: LinqValueExpr = new LinqIdentifierExpr();
  Key: LinqValueExpr = new LinqIdentifierExpr();
}

export class LinqOrderByExpr extends LinqExpr {
  Orderings: LinqOrderExpr[] = [];
}

export class LinqOrderExpr {
  Expression: LinqValueExpr = new LinqIdentifierExpr();
  Descending: boolean = false;
}

export class LinqTakeExpr extends LinqExpr {
  Count: LinqValueExpr = new LinqLiteralExpr();
}

export class LinqSkipExpr extends LinqExpr {
  Count: LinqValueExpr = new LinqLiteralExpr();
}

export abstract class LinqValueExpr {}

export class LinqIdentifierExpr extends LinqValueExpr {
  Name: string = "";
}

export class LinqMemberAccessExpr extends LinqValueExpr {
  Target: LinqValueExpr = new LinqIdentifierExpr();
  MemberName: string = "";
}

export class LinqBinaryExpr extends LinqValueExpr {
  Left: LinqValueExpr = new LinqIdentifierExpr();
  Operator: string = "";
  Right: LinqValueExpr = new LinqIdentifierExpr();
}

export class LinqLiteralExpr extends LinqValueExpr {
  Value: string = "";
}

// 預留 IntoExpr 結構
export class LinqIntoExpr extends LinqExpr {}

// LinqExecutor class for executing LinqQueryExpr
export class LinqExecutor {
  // 儲存資料來源
  public Data: Record<string, any[]> = {};

  // 執行查詢
  public execute(expr: LinqQueryExpr): any[] {
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