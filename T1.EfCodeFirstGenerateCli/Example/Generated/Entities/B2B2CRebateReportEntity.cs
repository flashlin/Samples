using System;

namespace Generated
{
    public class B2B2CRebateReportEntity
    {
        public int Id { get; set; }
        public int Batch { get; set; }
        public required string Username { get; set; }
        public required string AgentId { get; set; }
        public required string MaId { get; set; }
        public required string SmaId { get; set; }
        public DateTime? CreditDate { get; set; }
        public decimal Amount { get; set; }
        public decimal AgentPt { get; set; }
        public decimal MaPt { get; set; }
        public decimal SmaPt { get; set; }
        public decimal AgentAmount { get; set; }
        public decimal MaAmount { get; set; }
        public decimal SmaAmount { get; set; }
        public required string Currency { get; set; }
        public required string Status { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public required string Remark { get; set; }
        public Guid GUID { get; set; }
        public DateTime? WinloseDate { get; set; }
    }
}
