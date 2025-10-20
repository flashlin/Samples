using System;

namespace Generated
{
    public class CashSettledUpsertEntity
    {
        public int Id { get; set; }
        public int CustomerId { get; set; }
        public decimal CashSettled { get; set; }
        public decimal CashReturn { get; set; }
        public decimal AgtCashSettled { get; set; }
        public decimal AgtCashReturn { get; set; }
        public decimal MaCashSettled { get; set; }
        public decimal MaCashReturn { get; set; }
        public decimal SmaCashSettled { get; set; }
        public decimal SmaCashReturn { get; set; }
        public DateTime TransDate { get; set; }
        public bool IsProcessed { get; set; }
    }
}
