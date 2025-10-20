using System;

namespace Generated
{
    public class MonthlyTransferAccountsEntity
    {
        public int Id { get; set; }
        public int CustomerId { get; set; }
        public required string AccountId { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public bool IsDeleted { get; set; }
    }
}
