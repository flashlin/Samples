namespace T1.SourceGenerator.Utils;

public class Either<TLeft, TRight>
{
    private readonly bool _isLeft;
    private readonly TLeft? _left;
    private readonly TRight? _right;

    public Either(TLeft left)
    {
        _isLeft = true;
        _left = left;
    }

    public Either(TRight right)
    {
        _isLeft = false;
        _right = right;
    }

    public T Match<T>(Func<TLeft, T> left, Func<TRight, T> right)
    {
        return _isLeft ? left(_left!) : right(_right!);
    }
}