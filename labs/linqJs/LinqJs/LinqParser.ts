import {
  LinqQueryExpr,
  LinqFromExpr,
  LinqSelectExpr,
  LinqJoinExpr,
  LinqIdentifierExpr,
  LinqMemberAccessExpr,
  LinqNewExpr,
  LinqPropertyExpr
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
  // 解析 LINQ 查詢字串，回傳 AST
  public parse(query: string): LinqQueryExpr {
    const tokenizer = new LinqTokenizer();
    this._tokens = tokenizer.tokenize(query);
    this._i = 0;
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
    // where ... (暫不實作)
    if (this.peekToken() === 'where') {
      while (this.peekToken() !== 'select' && this._i < this._tokens.length) this.nextToken();
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