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

// select new { ... } AST 結構
export class LinqNewExpr extends LinqValueExpr {
  Properties: LinqPropertyExpr[] = [];
}

export class LinqPropertyExpr {
  Name: string = '';
  Value: LinqValueExpr = new LinqIdentifierExpr();
} 