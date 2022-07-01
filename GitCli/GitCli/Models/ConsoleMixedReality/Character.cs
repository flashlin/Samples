using System.Collections;
using System.Drawing;
using T1.Standard.DynamicCode;

namespace GitCli.Models.ConsoleMixedReality;

public struct Color
{
    public Color(byte red, byte green, byte blue)
    {
        Red = red;
        Green = green;
        Blue = blue;
    }

    public readonly byte Red;
    public readonly byte Green;
    public readonly byte Blue;

    public static implicit operator Color(ConsoleColor color) => GetColor(color);
    
    public static Color GetColor(ConsoleColor color)
    {
        if (color == ConsoleColor.DarkGray) return new Color(128, 128, 128);
        if (color == ConsoleColor.Gray) return new Color(192, 192, 192);
        var index = (int)color;
        var d = ((index & 8) != 0) ? (byte)255 : (byte)128;
        return new Color(
            ((index & 4) != 0) ? d : (byte)0,
            ((index & 2) != 0) ? d : (byte)0,
            ((index & 1) != 0) ? d : (byte)0);
    }

    public static Color White => new Color(255, 255, 255);
    public static Color Black => new Color(0, 0, 0);

    public static bool operator ==(in Color lhs, in Color rhs)
    {
        return
            lhs.Red == rhs.Red &&
            lhs.Green == rhs.Green &&
            lhs.Blue == rhs.Blue;
    }

    public static bool operator !=(Color a, Color b) => !(a == b);

    public static Color operator *(Color color, float factor) => new Color((byte) (color.Red * factor),
        (byte) (color.Green * factor), (byte) (color.Blue * factor));

    public static Color operator +(in Color lhs, in Color rhs) => new Color(
        (byte) Math.Min(byte.MaxValue, lhs.Red + rhs.Red),
        (byte) Math.Min(byte.MaxValue, lhs.Green + rhs.Green),
        (byte) Math.Min(byte.MaxValue, lhs.Blue + rhs.Blue));

    public override bool Equals(object obj)
    {
        return obj is Color color && this == color;
    }

    public override int GetHashCode()
    {
        var hashCode = -1058441243;
        hashCode = hashCode * -1521134295 + Red.GetHashCode();
        hashCode = hashCode * -1521134295 + Green.GetHashCode();
        hashCode = hashCode * -1521134295 + Blue.GetHashCode();
        return hashCode;
    }
}

public static class HashCodeCalculator
{
    public static int GetHashCodeOf(Type objType, object obj)
    {
        var fieldInfo = typeof(EqualityComparer<>)
            .MakeGenericType(objType)
            .GetField("Default");
        var getEqualityComparerDefault = DynamicField.GetField(typeof(EqualityComparer<>), fieldInfo);
        var equalityComparerDefault = (IEqualityComparer) getEqualityComparerDefault(null);
        return equalityComparerDefault.GetHashCode(obj);
    }
    
    public static int GetHashCode(IEnumerable<(Type type, object value)> objList)
    {
        var hashCode = -1058441243;
        foreach (var obj in objList)
        {
            hashCode = hashCode * -1521134295 + GetHashCodeOf(obj.type, obj.value);
        }
        return hashCode;
    }
}

public readonly struct Character
{
    public Character(char content, Color? foreground = null, Color? background = null)
    {
        Content = content;
        Foreground = foreground ?? Color.White;
        Background = background ?? Color.Black;
    }

    public Character(Color background)
    {
        Content = ' ';
        Foreground = Color.White;
        Background = background;
    }
    
    public char Content { get; init; }
    public Color Foreground { get; init; } 
    public Color Background { get; init; }


    public static Character Empty => new Character(' ');

    public static bool operator ==(Character a, Character b)
    {
        return a.Content == b.Content &&
               a.Foreground == b.Foreground &&
               a.Background == b.Background;
    }

    public static bool operator !=(Character a, Character b) => !(a == b);

    public override bool Equals(object obj)
    {
        return obj is Character character && this == character;
    }

    public override int GetHashCode()
    {
        var hashCode = -1661473088;
        hashCode = hashCode * -1521134295 + EqualityComparer<char>.Default.GetHashCode(Content);
        hashCode = hashCode * -1521134295 + EqualityComparer<Color>.Default.GetHashCode(Foreground);
        hashCode = hashCode * -1521134295 + EqualityComparer<Color>.Default.GetHashCode(Background);
        return hashCode;
    }
}