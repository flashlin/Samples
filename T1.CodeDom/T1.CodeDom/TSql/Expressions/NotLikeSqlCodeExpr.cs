﻿using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class NotLikeSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            Left.WriteToStream(stream);
            stream.Write(" NOT LIKE ");
            Right.WriteToStream(stream);
        }

        public SqlCodeExpr Left { get; set; }
        public SqlCodeExpr Right { get; set; }
    }
}