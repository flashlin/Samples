import {
  LinqExpr,
  LinqQueryExpr,
  LinqFromExpr,
  LinqSelectExpr,
  LinqJoinExpr,
  LinqIdentifierExpr,
  LinqMemberAccessExpr,
  LinqNewExpr,
  LinqPropertyExpr,
  LinqBinaryExpr,
  LinqWhereExpr
} from './LinqExprs';

// 將查詢字串轉為 token 陣列
export class LinqTokenizer {
  public tokenize(query: string): string[] {
    // 以空白、運算子、標點符號分割
    const regex = /([{}()\[\].,=><!]+|\w+|==|!=|<=|>=|\S)/g;
    // 避免數字與識別字混淆，保留所有符號
    return query.match(regex)?.filter(t => t.trim().length > 0) ?? [];
  }
}

// Result 型別與 ok/err 工具
export type Result<T> = { ok: true; value: T } | { ok: false; error: string };
export const ok = <T>(value: T): Result<T> => ({ ok: true, value });
export const err = <T = never>(error: string): Result<T> => ({ ok: false, error });


function parseLiteral(tokens: string[]): Result<{ expr: LinqExpr; rest: string[] }> {
  const [first, ...rest] = tokens;
  if (/^\d+$/.test(first)) {
    const lit = new LinqIdentifierExpr();
    lit.Name = first;
    return ok({ expr: lit, rest });
  }
  return err('Expected literal');
}

function parseMemberAccess(tokens: string[]): Result<{ expr: LinqExpr; rest: string[] }> {
  const [first, second, third, ...rest] = tokens;
  if (second === '.') {
    const member = new LinqMemberAccessExpr();
    const target = new LinqIdentifierExpr();
    target.Name = first;
    member.Target = target;
    member.MemberName = third;
    return ok({ expr: member, rest });
  }
  if (first) {
    const id = new LinqIdentifierExpr();
    id.Name = first;
    return ok({ expr: id, rest: [second, third, ...rest].filter(x => x !== undefined) });
  }
  return err('Expected identifier');
}

function parseBinaryExpr(tokens: string[]): Result<{ expr: LinqExpr; rest: string[] }> {
  const leftResult = parseMemberAccess(tokens);
  if (!leftResult.ok) return leftResult;
  const [op, ...rest1] = leftResult.value.rest;
  if (!['==', '!=', '<', '>', '<=', '>='].includes(op)) {
    return err('Expected binary operator after left expression');
  }
  const rightResult = parseLiteral(rest1);
  if (!rightResult.ok) return rightResult;
  // 正確建立 LinqBinaryExpr
  const bin = new LinqBinaryExpr();
  bin.Left = leftResult.value.expr;
  bin.Operator = op;
  bin.Right = rightResult.value.expr;
  return ok({
    expr: bin,
    rest: rightResult.value.rest,
  });
}

function parseWhere(tokens: string[]): Result<{ expr: LinqExpr; rest: string[] }> {
  if (tokens[0] !== 'where') return err('Expected where');
  const condResult = parseBinaryExpr(tokens.slice(1));
  if (!condResult.ok) return condResult;
  const whereExpr = new LinqWhereExpr();
  whereExpr.Condition = condResult.value.expr;
  return ok({ expr: whereExpr, rest: condResult.value.rest });
}

function parseSelect(tokens: string[]): Result<{ expr: LinqExpr; rest: string[] }> {
  if (tokens[0] !== 'select') return err('Expected select');
  return parseMemberAccess(tokens.slice(1));
}

function parseFrom(tokens: string[]): Result<{ from: string; fromAlias: string; rest: string[] }> {
  if (tokens[0] !== 'from') return err('Expected from');
  const fromAlias = tokens[1];
  if (tokens[2] !== 'in') return err('Expected in after from alias');
  const from = tokens[3];
  return ok({ from, fromAlias, rest: tokens.slice(4) });
}

function parseJoin(tokens: string[]): Result<{ expr: LinqExpr; rest: string[] }> {
  // join tb2 in orders on tb2.CustomerId equals tb1.id
  if (tokens[0] !== 'join') return err('Expected join');
  const alias = tokens[1];
  if (tokens[2] !== 'in') return err('Expected in after join alias');
  const source = tokens[3];
  if (tokens[4] !== 'on') return err('Expected on after join source');
  // tb2.CustomerId equals tb1.id
  const leftRes = parseMemberAccess(tokens.slice(5));
  if (!leftRes.ok) return leftRes;
  if (leftRes.value.rest[0] !== 'equals') return err('Expected equals in join');
  const rightRes = parseMemberAccess(leftRes.value.rest.slice(1));
  if (!rightRes.ok) return rightRes;
  // 這裡僅示意，實際應建立 LinqJoinExpr
  const joinExpr = new LinqJoinExpr();
  joinExpr.Identifier = alias;
  joinExpr.Source = source;
  joinExpr.OuterKey = leftRes.value.expr;
  joinExpr.InnerKey = rightRes.value.expr;
  return ok({ expr: joinExpr, rest: rightRes.value.rest });
}

export function parseExpr(tokens: string[]): Result<{ expr: LinqExpr; rest: string[] }> {
  // from tb1 in customer [join ...] [where ...] select ...
  const fromResult = parseFrom(tokens);
  if (!fromResult.ok) return fromResult;
  let rest = fromResult.value.rest;
  let join: LinqExpr | undefined;
  let where: LinqExpr | undefined;
  let groupBy: LinqExpr | undefined;
  // join
  if (rest[0] === 'join') {
    const joinRes = parseJoin(rest);
    if (!joinRes.ok) return joinRes;
    join = joinRes.value.expr;
    rest = joinRes.value.rest;
  }
  // where
  if (rest[0] === 'where') {
    const whereRes = parseWhere(rest);
    if (!whereRes.ok) return whereRes;
    where = whereRes.value.expr;
    rest = whereRes.value.rest;
  }
  // group by (暫略)
  if (rest[0] === 'group') {
    rest = rest.slice(3); // 跳過 group by ...
  }
  // select
  const selectRes = parseSelect(rest);
  if (!selectRes.ok) return selectRes;
  return ok({
    expr: selectRes.value.expr,
    rest: selectRes.value.rest,
  });
}

// LinqParser class for parsing LINQ query string to AST
export class LinqParser {
  private _tokens: string[] = [];
  private _i: number = 0;
  private nextToken(): string {
    return this._tokens[this._i++];
  }
  private peekToken(): string {
    return this._tokens[this._i];
  }
  // 解析成 LinqMemberAccessExpr
  private _parseMemberAccess(tokens: string[], start: number): [LinqMemberAccessExpr, number] {
    // 例如 tb1.id
    const id = tokens[start];
    if (tokens[start + 1] === '.') {
      const member = new LinqMemberAccessExpr();
      const target = new LinqIdentifierExpr();
      target.Name = id;
      member.Target = target;
      member.MemberName = tokens[start + 2];
      return [member, start + 3];
    } else {
      // 單一識別字
      const member = new LinqMemberAccessExpr();
      const target = new LinqIdentifierExpr();
      target.Name = id;
      member.Target = target;
      member.MemberName = '';
      return [member, start + 1];
    }
  }
  // 解析 select new { ... } 區塊
  private _parseSelectNew(): LinqNewExpr {
    this.nextToken(); // new
    this.nextToken(); // {
    const newExpr = new LinqNewExpr();
    while (this._i < this._tokens.length && this.peekToken() !== '}') {
      if (this.peekToken() === ',') {
        this.nextToken();
        continue;
      }
      let name = this.nextToken();
      let valueExpr: any;
      if (this.peekToken() === '=') {
        this.nextToken(); // =
        if (this._tokens[this._i + 1] === '.') {
          const [member, nextIdx] = this._parseMemberAccess(this._tokens, this._i);
          valueExpr = member;
          this._i = nextIdx;
        } else {
          valueExpr = new LinqIdentifierExpr();
          valueExpr.Name = this.nextToken();
        }
      } else if (this.peekToken() === '.') {
        const [member, nextIdx] = this._parseMemberAccess(this._tokens, this._i - 1);
        valueExpr = member;
        name = this._tokens[this._i - 1];
        this._i = nextIdx;
      } else {
        valueExpr = new LinqIdentifierExpr();
        valueExpr.Name = name;
      }
      const prop = new LinqPropertyExpr();
      prop.Name = name;
      prop.Value = valueExpr;
      newExpr.Properties.push(prop);
    }
    this.nextToken(); // }
    return newExpr;
  }
  // 解析 join 區塊
  private _parseJoin(expr: LinqQueryExpr) {
    this.nextToken(); // join
    const join = new LinqJoinExpr();
    join.Identifier = this.nextToken();
    if (this.nextToken() !== 'in') throw new Error('join 後需 in');
    join.Source = this.nextToken();
    if (this.nextToken() !== 'on') throw new Error('join 後需 on');
    const [outerKey, nextIdx] = this._parseMemberAccess(this._tokens, this._i);
    join.OuterKey = outerKey;
    this._i = nextIdx;
    if (this.nextToken() !== 'equals') throw new Error('join on 後需 equals');
    const [innerKey, nextIdx2] = this._parseMemberAccess(this._tokens, this._i);
    join.InnerKey = innerKey;
    this._i = nextIdx2;
    expr.Joins.push(join);
  }
  // 解析 where 區塊
  private _parseWhere(expr: LinqQueryExpr) {
    this.nextToken(); // where
    // 解析左側（如 tb1.status）
    const left = this._parseMemberAccess(this._tokens, this._i);
    const leftExpr = left[0];
    this._i = left[1];
    const op = this.nextToken(); // 例如 ==
    // 解析右側（如 1）
    let rightExpr;
    const rightToken = this._tokens[this._i];
    if (/^\d+$/.test(rightToken)) {
      rightExpr = new LinqIdentifierExpr();
      rightExpr.Name = rightToken;
      this._i++;
    } else {
      rightExpr = new LinqIdentifierExpr();
      rightExpr.Name = this.nextToken();
    }
    // 組成 LinqBinaryExpr
    const cond = new LinqBinaryExpr();
    cond.Left = leftExpr;
    cond.Operator = op;
    cond.Right = rightExpr;
    expr.Where = new LinqWhereExpr();
    expr.Where.Condition = cond;
  }
  // 初始化 tokens 與索引
  private _initTokens(query: string) {
    const tokenizer = new LinqTokenizer();
    this._tokens = tokenizer.tokenize(query);
    this._i = 0;
  }
  // 解析 LINQ 查詢字串，回傳 AST
  public parse(query: string): LinqQueryExpr {
    this._initTokens(query);
    const expr = new LinqQueryExpr();
    // from tb1 in customer
    if (this.nextToken() !== 'from') throw new Error('必須以 from 開頭');
    expr.From = new LinqFromExpr();
    expr.From.Identifier = this.nextToken();
    if (this.nextToken() !== 'in') throw new Error('from 後需 in');
    expr.From.Source = this.nextToken();
    // join ...
    if (this.peekToken() === 'join') {
      this._parseJoin(expr);
    }
    // where ...
    if (this.peekToken() === 'where') {
      this._parseWhere(expr);
    }
    // select ...
    if (this.nextToken() !== 'select') throw new Error('缺少 select');
    expr.Select = new LinqSelectExpr();
    if (this.peekToken() === 'new' && this._tokens[this._i + 1] === '{') {
      expr.Select.Expression = this._parseSelectNew();
    } else {
      const idExpr = new LinqIdentifierExpr();
      idExpr.Name = this.nextToken();
      expr.Select.Expression = idExpr;
    }
    return expr;
  }
} 