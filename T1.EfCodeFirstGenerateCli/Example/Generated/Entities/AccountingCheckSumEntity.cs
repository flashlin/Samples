using System;

namespace Generated
{
    public class AccountingCheckSumEntity
    {
        public DateTime StatementDate { get; set; }
        public int StatementType { get; set; }
        public byte ProductType { get; set; }
        public decimal DailyTotalRaw { get; set; }
        public decimal DailyTotalSma { get; set; }
        public decimal DailyTotalCash { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
    }
}
