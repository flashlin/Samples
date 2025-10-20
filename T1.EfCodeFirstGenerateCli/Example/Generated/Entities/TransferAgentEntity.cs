using System;

namespace Generated
{
    public class TransferAgentEntity
    {
        public int custid { get; set; }
        public required string username { get; set; }
        public required string currencystr { get; set; }
        public int currency { get; set; }
        public int recommend { get; set; }
        public int mrecommend { get; set; }
        public int srecommend { get; set; }
    }
}
