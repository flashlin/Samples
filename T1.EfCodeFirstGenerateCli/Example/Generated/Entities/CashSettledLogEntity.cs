using System;

namespace Generated
{
    public class CashSettledLogEntity
    {
        public DateTime UpdateTime { get; set; }
        public int CustomerId { get; set; }
        public int ProductType { get; set; }
        public decimal? SettledAmount { get; set; }
        public decimal? ReturnAmount { get; set; }
    }
}
