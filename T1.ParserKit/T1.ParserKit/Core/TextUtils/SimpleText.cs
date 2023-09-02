using System.Diagnostics;

namespace T1.ParserKit.Core.TextUtils
{
    internal class SimpleText : Text, IEnumerable<char>
    {
        #region Fields

        private int index;
        private int length;

        #endregion //Fields

        #region Properties

        #region Length

        public override int Length
        {
            get { return this.length; }
        }

        #endregion //Length

        #region Indexer

        public override char this[int index]
        {
            get
            {
                if (index >= this.length)
                    throw new IndexOutOfRangeException("The index is out of bounds.");

                return this.value[this.index + index];
            }
        }

        #endregion //Indexer

        #endregion //Properties

        #region Constructors

        public SimpleText(string value, int index, int length)
        {
            Debug.Assert(value != null);
            Debug.Assert(index >= 0);
            Debug.Assert(length >= 0);
            Debug.Assert(index + length <= value.Length);

            this.value = value;
            this.index = index;
            this.length = length;
        }

        #endregion //Constructors

        #region IEnumerable Members

        public override IEnumerator<char> GetEnumerator()
        {
            for (int i = 0, length = this.length; i < length; i++)
                yield return this[i];
        }

        #endregion //IEnumerable Members

        #region Methods

        #region Split

        public override SplitText Split(int index)
        {
            if (index < 0 || index > this.length)
                throw new ArgumentOutOfRangeException("The index is out of the bounds of the Text.");

            var head = new SimpleText(this.value, this.index, index);
            var tail = new SimpleText(this.value, this.index + index, this.length - index);

            return new SplitText(head, tail);
        }

        #endregion //Split

        #region IsSimpleTextAppendableTo

        internal override bool IsSimpleTextAppendableTo(SimpleText tail)
        {
            Debug.Assert(tail != null);
            Debug.Assert(tail.Length > 0);
            Debug.Assert(this.Length > 0);

            return
                object.ReferenceEquals(this.value, tail.value) &&
                this.index + this.length == tail.index;
        }

        #endregion //IsSimpleTextAppendableTo

        #region IsComplexTextAppendableTo

        internal override bool IsComplexTextAppendableTo(ComplexText tail)
        {
            Debug.Assert(tail != null);
            Debug.Assert(tail.Length > 0);
            Debug.Assert(this.Length > 0);

            var firstTailComponent = tail.texts[0] as SimpleText;
            return
                firstTailComponent != null &&
                IsSimpleTextAppendableTo(firstTailComponent);
        }

        #endregion //IsComplexTextAppendableTo

        #region AppendSimpleText

        internal override Text AppendSimpleText(SimpleText tail)
        {
            Debug.Assert(tail != null);
            Debug.Assert(tail.Length > 0);
            Debug.Assert(this.Length > 0);

            if (IsSimpleTextAppendableTo(tail))
                return new SimpleText(this.value, this.index, this.length + tail.length);

            return Text.Join(this, tail);
        }

        #endregion AppendSimpleText

        #region AppendComplexText

        internal override Text AppendComplexText(ComplexText tail)
        {
            Debug.Assert(tail != null);
            Debug.Assert(tail.Length > 0);
            Debug.Assert(this.Length > 0);

            var tailComponents = tail.texts;
            var textComponents = new List<Text>();

            if (IsComplexTextAppendableTo(tail))
            {
                textComponents.Add(this.AppendSimpleText((SimpleText)tailComponents[0]));
                textComponents.AddRange(tailComponents.Skip(1));
            }
            else
            {
                textComponents.Add(this);
                textComponents.AddRange(tailComponents);
            }

            if (textComponents.Count == 1)
                return textComponents[0];

            return Text.Join(textComponents);
        }

        #endregion //AppendComplexText

        #region Evaluate

        internal override string Evaluate()
        {
            var value = this.value.Substring(this.index, this.length);
            this.index = 0;
            return value;
        }

        #endregion //Evaluate

        #endregion //Methods
    }
}
