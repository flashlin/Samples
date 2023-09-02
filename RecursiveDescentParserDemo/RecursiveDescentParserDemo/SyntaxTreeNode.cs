using System.Text;

public class SyntaxTreeNode
{
    public string Token { get; set; }
    public SyntaxTreeNode Left { get; set; }
    public SyntaxTreeNode Right { get; set; }

    public override string ToString()
    {
        var sb = new StringBuilder();
        if (Left != null)
        {
            sb.Append(Left.ToString());
        }

        sb.Append(Token);
        if (Right != null)
        {
            sb.Append(Right.ToString());
        }

        return sb.ToString();
    }
}