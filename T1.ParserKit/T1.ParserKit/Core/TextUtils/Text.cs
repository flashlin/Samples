using System.Collections;
using T1.ParserKit.Core.Utilities;

namespace T1.ParserKit.Core.TextUtils
{
    public abstract class Text : IEnumerable<char>
    {
        #region Fields

        internal string value;
        private bool isEvaluated;

        #endregion //Fields

        #region Properties

        public abstract int Length { get; }
        public abstract char this[int index] { get; }

        #endregion //Properties

        #region Base Class Overrides

        public override string ToString()
        {
            if (this.isEvaluated)
                return this.value;

            this.value = Evaluate();
            this.isEvaluated = true;
            return this.value;
        }

        #endregion //Base Class Overrides

        #region IEnumerable Members

        public abstract IEnumerator<char> GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((Text)this).GetEnumerator();
        }

        #endregion //IEnumerable Members

        #region Methods

        #region Append

        public Text Append(Text tail)
        {
            tail.AssertNotNull();
            if (this.Length == 0)
                return tail;
            if (tail.Length == 0)
                return this;

            var simpleTail = tail as SimpleText;
            if (simpleTail != null)
                return this.AppendSimpleText(simpleTail);

            return this.AppendComplexText((ComplexText)tail);
        }

        #endregion //Append

        #region StartsWith

        public bool StartsWith(string text)
        {
            text.AssertNotNull();
            if (this.Length < text.Length)
                return false;

            var index = 0;
            foreach (var character in text)
            {
                if (this[index++] != character)
                    return false;
            }

            return true;
        }

        #endregion //StartsWith

        #region Split

        public abstract SplitText Split(int index);

        #endregion //Split

        #region Skip

        public Text Skip(int count)
        {
            count.AssertNotNegative();
            if (this.Length <= count)
                return Text.Empty;

            return Split(count).Tail;
        }

        #endregion //Skip

        #region SkipWhile

        public Text SkipWhile(Func<char, bool> predicate)
        {
            var index = 0;
            while (index < this.Length && predicate(this[index]))
                index = index + 1;

            if (index >= this.Length)
                return Text.Empty;
            return Split(index).Tail;
        }

        #endregion //SkipWhile

        #region Take

        public Text Take(int count)
        {
            count.AssertNotNegative();
            if (this.Length <= count)
                return this;

            return Split(count).Head;
        }

        #endregion //Take

        #region TakeWhile

        public Text TakeWhile(Func<char, bool> predicate)
        {
            var index = 0;
            while (index < this.Length && predicate(this[index]))
                index = index + 1;

            if (index >= this.Length)
                return this;
            return Split(index).Head;
        }

        #endregion //TakeWhile

        #endregion //Methods

        #region Internal Methods

        internal abstract bool IsSimpleTextAppendableTo(SimpleText tail);
        internal abstract bool IsComplexTextAppendableTo(ComplexText tail);
        internal abstract Text AppendSimpleText(SimpleText tail);
        internal abstract Text AppendComplexText(ComplexText tail);
        internal abstract string Evaluate();

        #endregion //Internal Methods

        #region Static Members

        #region Fields

        private static Text empty = new SimpleText(string.Empty, 0, 0);

        #endregion //Fields

        #region Properties

        public static Text Empty
        {
            get { return empty; }
        }

        #endregion //Properties

        #region Methods

        #region Create

        public static Text Create(string value)
        {
            value.AssertNotNull();
            return new SimpleText(value, 0, value.Length);
        }

        public static Text Create(char value)
        {
            return new SimpleText(value.ToString(), 0, 1);
        }

        #endregion //Create

        #region Join

        public static Text Join(params Text[] texts)
        {
            return Join((IEnumerable<Text>)texts);
        }

        public static Text Join(IEnumerable<Text> texts)
        {
            texts.AssertNotNull();
            var result = new ComplexText(texts);
            var resultComponents = result.texts;

            if (resultComponents.Count == 0)
                return Text.Empty;
            if (resultComponents.Count == 1)
                return resultComponents[0];

            return result;
        }

        #endregion //Join

        #region IsEmpty

        public static bool IsEmpty(Text text)
        {
            text.AssertNotNull();
            return text.Length == 0;
        }

        #endregion //IsEmpty

        #endregion //Methods

        #region Operators

        public static implicit operator Text(string text)
        {
            return Text.Create(text);
        }

        public static implicit operator Text(char character)
        {
            return Text.Create(character);
        }

        public static implicit operator string(Text text)
        {
            text.AssertNotNull();
            return text.ToString();
        }

        #endregion //Operators

        #endregion //Static Members
    }
}
