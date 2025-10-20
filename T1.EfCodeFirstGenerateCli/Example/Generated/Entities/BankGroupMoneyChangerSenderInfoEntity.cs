using System;

namespace Generated
{
    public class BankGroupMoneyChangerSenderInfoEntity
    {
        public int Id { get; set; }
        public int MoneyChangerGroupId { get; set; }
        public required string SenderName { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string UpdatedBy { get; set; }
        public DateTime? UpdatedOn { get; set; }
        public int? Status { get; set; }
        public DateTime? ClosedDate { get; set; }
        public int? TargetGroupId { get; set; }
        public bool IsRequireTargetSender { get; set; }
        public bool IsTargetSender { get; set; }
    }
}
