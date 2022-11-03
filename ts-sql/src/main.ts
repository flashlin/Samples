import P from "parsimmon";

const seq = P.seq;
const alt = P.alt;
const regex = P.regex;
const string = P.string;
const optWhitespace = P.optWhitespace;
const whitespace = P.whitespace;
const lazy = P.lazy;

// const CrazyPointParser = P.createLanguage({
//    Num: () => P.regexp(/[0-9]+/).map(Number),
//    QuoteString: () => P.seq(P.string("'"), P.),
//    Point: (r) => P.seq(P.string('['), r.Num, P.string(' '), r.Num, P.string(']')).map(([_open, x, _space, y, _close]) => [x, y]),
//    PointSet: (r) => P.seq(P.string('('), r.Point.sepBy(r.Sep), P.string(')')).map(([_open, points, _close]) => points),
//    PointSetArray: (r) => P.seq(P.string('.:{'), r.PointSet.sepBy(r.Sep), P.string('}:.')).map(([_open, pointSets, _close]) => pointSets)
// })


// getValue(node[0], x=>x.xxx, "")
function getValue(node, fn, defaultValue) {
  if( node == null ) {
    return defaultValue;
  }
  return fn(node);
}

function opt(parser, empty?: any) {
  if (typeof empty == "undefined") return parser.or(P.succeed([]));
  return parser.or(P.succeed(empty));
}

function getPos(parser) {
  return seq(P.index, parser, P.index).map((node) => {
    let pos: any = {
      start: node[0],
      end: node[2],
    };
    if (typeof node[1] == "object") {
      let n = node[1];
      n.position = pos;
      return n;
    }
    pos.out = node[1];
    return pos;
  });
}

function removeQuotes(str) {
  return str.replace(/^([`'"])(.*)\1$/, "$2");
}

function mkString(node) {
	return node.join('');
}

const str = alt(
  regex(/"[^"\\]*(?:\\.[^"\\]*)*"/),
  regex(/'[^'\\]*(?:\\.[^'\\]*)*'/)
);

function mergeOptionnalList(node) {
  node[0].push(node[1]);
  return node[0];
}

function optionnalList(parser) {
  return seq(
    parser.skip(optWhitespace).skip(string(",")).skip(optWhitespace).many(),
    parser.skip(optWhitespace)
  ).map(mergeOptionnalList);
}

const Identifier = regex(/[a-zA-Z_]?[a-zA-Z0-9_]+/);
const Number = regex(/[0-9]+/);

const AS = regex(/AS\s/i);
const SELECT = regex(/SELECT/i);
const FROM = regex(/FROM/i);

const colName = alt(
  Identifier
  //regex(/(?!(FROM|WHERE|GROUP BY|ORDER BY|LIMIT|INNER|LEFT|RIGHT|JOIN|ON|VALUES|SET)\s)[a-z*][a-z0-9_]*/i),
  //regex(/`[^`\\]*(?:\\.[^`\\]*)*`/)
);

const expression = seq(Number);

const colListExpression = seq(
  expression,
  opt(
    // Alias
    seq(optWhitespace, opt(AS), alt(colName, str)).map(function (node) {
      let n = {
        alias: node[2],
        expression: node.join(""),
      };
      return n;
    })
    //null
  )
).map((node) => {
  let n = node[0];
  n.alias = node[1] !== null ? node[1].alias : null;
  n.expression = node[0] + (node[1] !== null ? node[1].expression : "");
  return n;
});


const colList = optionnalList(getPos(colListExpression));


const subTable = seq(
  string('('),
  lazy(_ => selectParser),
  string(')')
);

const tableAndColumn = seq(
	colName,
	string('.'),
	colName
);
const tableListExpression = seq(
	alt(
    tableAndColumn.map(mkString),
		colName,
    subTable,
	),
	opt(	// Alias
		seq(
			optWhitespace,
			opt(AS),
			alt(colName, str)
		).map(function(node) {
			return {
				alias: removeQuotes(node[2]),
				expression: node.join(''),
			};
		}),
		null
	)
).map(function(node) {
	let n: any = {};
	n.table = node[0];
	n.alias = (node[1] !== null) ? node[1].alias : null;
	n.expression = node[0] + ((node[1] !== null) ? node[1].expression : '');
	return n;
});
const tableList = optionnalList(getPos(tableListExpression));

// const select1Parser = seq(
//   SELECT.skip(optWhitespace).then(opt(colList))
// ).map(function (node) {
//   return {
//     type: "select",
//     select: node[0],
//   };
// });

const selectParser = seq(
  SELECT.skip(optWhitespace).then(opt(colList)),
  opt(FROM).skip(optWhitespace).then(opt(tableList)),
  //opt(joinList),
  //opt(regex(/WHERE/i).skip(optWhitespace).then(opt(whereExpression)), null),
  //opt(regex(/\s?GROUP BY/i).skip(optWhitespace).then(opt(groupList))),
  //opt(regex(/\s?ORDER BY/i).skip(optWhitespace).then(opt(orderList))),
).map(function (node) {
  return {
    type: "select",
    select: node[0],
    from: node[1],
    //join: node[2],
    //where: node[3],
    //group: node[4],
    //order: node[5],
    //limit: node[6],
  };
});

const p = alt(selectParser); //, insertParser, updateParser, deleteParser)

function sql2ast(sql: string) {
  var result = p.parse(sql);
  if (result.status === false) result.error = P.formatError(sql, result);
  return result;
}

interface IResult {
  status: boolean;
  //index: { offset: 7, line: 1, column: 8 },
  //expected: [ 'EOF' ],
  error: string;
  value: any;
}

let rc = sql2ast("select 123 from (select name from user)");
if (rc.status == false) {
  console.log("test", rc.error);
} else {
  console.log("raw", rc.value);
  console.log("json", JSON.stringify(rc.value));
}
