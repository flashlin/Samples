// See https://aka.ms/new-console-template for more information

using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore;

Console.WriteLine("Hello, World!");

var db = new MyDbContext();
Expression<Func<MyDbContext, IQueryable<User>>> q1 = (db) => from tb1 in db.Users select tb1;
var parser = new LinqLexer(q1.Body);
parser.Parse();
var t = parser.Result;

public abstract class CodeExpr
{
}

public class MetaData
{
    public string TableName { get; set; }
    public string FieldName { get; set; }
    public bool IsSelect { get; set; }
    public bool IsJoin { get; set; }
    public string JoinCondition { get; set; }
}


// install System.Linq.Dynamic
public class LinqLexer
{
    private Expression _expression;
    private MetaData _metaData;

    public string Result { get; private set; }
    
    
    public LinqLexer(Expression expression)
    {
        _expression = expression;
        _metaData = new MetaData();
        Result = string.Empty;
    }

    public string Parse()
    {
        if (_expression is MethodCallExpression methodCallExpression)
        {
            var methodName = methodCallExpression.Method.Name;
            
            if (methodName == "Select" || methodName == "SelectMany")
            {
                _metaData.IsSelect = true;
            }
            // 如果方法名為 Join 或 GroupJoin，則設置 IsJoin 屬性為 true
            else if (methodName == "Join" || methodName == "GroupJoin")
            {
                _metaData.IsJoin = true;
            }
            
            // 取得方法的第一個參數的元數據
            var arg = new LinqLexer(methodCallExpression.Arguments[0]);
            arg.Parse();
            
            // 如果方法名為 Select 或 SelectMany，則解析 Select 語句中的字段名
            if (methodName == "Select" || methodName == "SelectMany")
            {
                // 取得 Lambda 表達式的右側
                var lambda = (LambdaExpression) ((UnaryExpression) methodCallExpression.Arguments[1]).Operand;
                
                // 取得 Lambda 表達式的右側的 Body
                var body = lambda.Body;

                // 如果 Body 是二元運算表達式，則進行遞歸解析
                if (body is BinaryExpression binaryExpression)
                {
                    var binary = new LinqLexer(binaryExpression);
                    binary.Parse();
                    // 取得字段名
                    _metaData.FieldName = binary.Result;
                }
                // 如果 Body 是屬性表達式，則直接取得屬性名
                else if (body is MemberExpression memberExpression)
                {
                    _metaData.FieldName = memberExpression.Member.Name;
                }
            }
            
            
            if (methodName == "Join" || methodName == "GroupJoin")
            {
                // 取得 Join 語句中的表名
                _metaData.TableName = methodCallExpression.Arguments[1].Type.Name;

                // 取得 Join 語句中的關聯條件
                var lambda = (LambdaExpression)((UnaryExpression)methodCallExpression.Arguments[2]).Operand;
                var body = lambda.Body;

                if (body is BinaryExpression binaryExpression)
                {
                    var binary = new LinqLexer(binaryExpression);
                    binary.Parse();

                    // 取得關聯條件
                    _metaData.JoinCondition = binary.Result;
                }
                else if (body is MethodCallExpression methodCallExpression1)
                {
                    var method = new LinqLexer(methodCallExpression1);
                    method.Parse();
                    // 取得關聯條件
                    _metaData.JoinCondition = method.Result;
                }
            }

        }
        
        return null;
    }

    private string GetOperator(ExpressionType nodeType)
    {
        return null;
    }
}

public class MyDbContext : DbContext
{
    public DbSet<User> Users { get; set; }
    public DbSet<Home> Homes { get; set; }
}

public class User
{
    public int Id { get; set; }
    public string Name { get; set; }
    public DateTime Birth { get; set; }
    public decimal Price { get; set; }
}

public class Home
{
    public int Id { get; set; }
    public string Name { get; set; }
    public string Address { get; set; }
}