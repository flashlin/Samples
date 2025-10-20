using System;

namespace Generated
{
    public class MonthlyTransferConfigEntity
    {
        public int Id { get; set; }
        public int CustomerId { get; set; }
        public required string AccountId { get; set; }
        public int TargetCustomerId { get; set; }
        public required string TargetAccountId { get; set; }
        public required string Description { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public bool IsDeleted { get; set; }
    }
}
