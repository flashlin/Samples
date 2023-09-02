namespace T1.ParserKit.Core.TextUtils
{
    public struct SplitText
    {
        private Text head;
        private Text tail;

        public Text Head { get { return this.head; } }
        public Text Tail { get { return this.tail; } }

        public SplitText(Text head, Text tail)
        {
            this.head = head;
            this.tail = tail;
        }

        public override string ToString()
        {
            return string.Format(
                "Head: \"{0}\", Tail: \"{1}\"",
                this.Head == null ? string.Empty : this.Head.ToString(),
                this.Tail == null ? string.Empty : this.Tail.ToString());
        }
    }
}
