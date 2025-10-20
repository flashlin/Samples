using System;

namespace Generated
{
    public class MnlStatementForIomCustomerEntity
    {
        public int Id { get; set; }
        public int CustomerID { get; set; }
        public DateTime WinLostDate { get; set; }
        public byte StatementType { get; set; }
        public decimal TotalCashIn { get; set; }
        public decimal TotalCashOut { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
    }
}
