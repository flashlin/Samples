using System;

namespace Generated
{
    public class CashSettledLogSumEntity
    {
        public int CustomerId { get; set; }
        public DateTime UpdateTime { get; set; }
        public int ProductType { get; set; }
        public decimal? SettledAmount { get; set; }
        public decimal? ReturnAmount { get; set; }
    }
}
