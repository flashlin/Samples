using System;

namespace Generated
{
    public class SpecialSalaryTransactionsEntity
    {
        public int Id { get; set; }
        public DateTime TransDate { get; set; }
        public required string FromAccount { get; set; }
        public required string ToAccount { get; set; }
        public decimal Amount { get; set; }
        public decimal ExchangeRate { get; set; }
        public required string TransDesc { get; set; }
        public required string TransRemark { get; set; }
        public required string ExchangeDescription { get; set; }
        public required string ExchangeRemark { get; set; }
        public int Status { get; set; }
        public int StatementType { get; set; }
        public int FromCurrency { get; set; }
        public required string Remark { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public int ToCurrency { get; set; }
    }
}
