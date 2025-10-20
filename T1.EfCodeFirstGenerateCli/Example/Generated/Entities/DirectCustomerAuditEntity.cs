using System;

namespace Generated
{
    public class DirectCustomerAuditEntity
    {
        public int DirectCustID { get; set; }
        public required string Auditor { get; set; }
        public decimal CashBalance { get; set; }
        public decimal TotalBalance { get; set; }
        public DateTime AuditDate { get; set; }
        public required string Remark { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
    }
}
